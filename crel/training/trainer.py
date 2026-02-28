"""CREL trainer with alternating optimization.

Implements alternating optimization of loss-net and task-net:

  Per batch:
    1. Update loss-net with contrastive ranking loss (InfoNCE or NCE)
    2. Update task-net with -E + BCE

Energy magnitude is bounded by spectral normalization on the energy
network — no clamping, ramp-up, or adaptive loss balancing needed.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from crel.losses.nce import NCELoss, InfoNCELoss, compute_log_prob, compute_log_prob_gaussian
from crel.losses.task_loss import TaskLoss
from crel.training.ema import EMACenter
from crel.training.sampling import get_sampler
from crel.diagnostics.metrics import DiagnosticTracker, compute_f1_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the CREL trainer."""

    # Optimization
    total_epochs: int = 100
    task_net_lr: float = 1e-3
    loss_net_lr: float = 1e-3
    weight_decay: float = 1e-5

    # Loss weights
    lambda_energy: float = 0.1
    lambda_bce: float = 1.0

    # EMA centering
    ema_beta: float = 0.99
    ema_warmup_beta: float = 0.9
    ema_warmup_fraction: float = 0.05
    per_sample_ema: bool = True

    # Contrastive loss
    contrastive_loss: str = "infonce"  # "nce" or "infonce"
    nce_samples: int = 32
    nce_sampling: str = "bernoulli"
    nce_gaussian_sigma: float = 0.3
    infonce_temperature: float = 1.0
    energy_reg: float = 0.01  # λ_reg * E² regularization on contrastive loss
    stop_gradient_energy: bool = True  # detach energy in task loss

    # Gradient clipping (both nets)
    grad_clip: float = 5.0

    # Diagnostics
    log_interval: int = 50
    track_diagnostics: bool = True

    # Experiment logging
    experiment_dir: str = ""
    experiment_description: str = ""

    # Device
    device: str = "cuda"


class CRELTrainer:
    """Trainer implementing alternating optimization for CREL.

    Per batch:
      1. Update loss-net with contrastive ranking loss
      2. Update task-net with -E + BCE (energy optionally stop-gradient)

    Manages EMA centering updates, NCE sampling, and diagnostic tracking.
    """

    def __init__(
        self,
        task_net: nn.Module,
        loss_net: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        config: TrainerConfig | None = None,
    ):
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        self.task_net = task_net.to(self.device)
        self.loss_net = loss_net.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Mixed precision
        self.use_amp = self.device.type == "cuda"
        self.task_scaler = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)
        self.loss_scaler = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)

        # Optimizers
        self.task_opt = torch.optim.Adam(
            self.task_net.parameters(),
            lr=self.config.task_net_lr,
            weight_decay=self.config.weight_decay,
        )
        self.loss_opt = torch.optim.Adam(
            self.loss_net.parameters(),
            lr=self.config.loss_net_lr,
            weight_decay=self.config.weight_decay,
        )

        # Losses
        self.task_loss_fn = TaskLoss(
            lambda_energy=self.config.lambda_energy,
            lambda_bce=self.config.lambda_bce,
        )
        if self.config.contrastive_loss == "infonce":
            self.contrastive_loss_fn = InfoNCELoss(
                num_samples=self.config.nce_samples,
                temperature=self.config.infonce_temperature,
            )
        else:
            self.contrastive_loss_fn = NCELoss(num_samples=self.config.nce_samples)

        # Sampling
        self.sampler = get_sampler(self.config.nce_sampling)

        # EMA centering
        dataset_size = len(train_loader.dataset)
        num_labels = self._get_num_labels()
        total_steps = self.config.total_epochs * len(train_loader)
        ema_warmup_steps = int(self.config.ema_warmup_fraction * total_steps)

        self.ema = EMACenter(
            num_labels=num_labels,
            dataset_size=dataset_size,
            beta=self.config.ema_beta,
            warmup_beta=self.config.ema_warmup_beta,
            warmup_steps=ema_warmup_steps,
            per_sample=self.config.per_sample_ema,
        ).to(self.device)

        # Diagnostics
        self.diagnostics = DiagnosticTracker(log_interval=self.config.log_interval)

        self.total_steps = total_steps
        self.global_step = 0

        # Experiment file logging
        self._exp_file = None
        self._exp_path = None
        if self.config.experiment_dir:
            from datetime import datetime
            exp_dir = Path(self.config.experiment_dir)
            exp_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_path = exp_dir / f"epoch_results_{timestamp}.txt"
            self._exp_path = exp_path
            self._exp_file = open(exp_path, "w", encoding="utf-8")
            self._write_experiment_header(timestamp)
            logger.info("Logging experiment results to %s", exp_path)

    def _log_to_file(self, line: str) -> None:
        if self._exp_file is not None:
            self._exp_file.write(line + "\n")
            self._exp_file.flush()
            os.fsync(self._exp_file.fileno())

    def _write_experiment_header(self, timestamp: str) -> None:
        from datetime import datetime
        lines = [
            f"# CREL Experiment Log",
            f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Description: {self.config.experiment_description or 'N/A'}",
            f"#",
            f"# --- Configuration ---",
            f"# Device: {self.device}",
            f"# Total epochs: {self.config.total_epochs}",
            f"# Training protocol: alternating (1 contrastive + 1 task step per batch)",
            f"# Task-net LR: {self.config.task_net_lr}",
            f"# Loss-net LR: {self.config.loss_net_lr}",
            f"# Lambda energy: {self.config.lambda_energy}",
            f"# Lambda BCE: {self.config.lambda_bce}",
            f"# EMA beta: {self.config.ema_beta}",
            f"# Contrastive loss: {self.config.contrastive_loss}",
            f"# InfoNCE temperature: {self.config.infonce_temperature}",
            f"# NCE samples: {self.config.nce_samples}",
            f"# Energy reg: {self.config.energy_reg}",
            f"# Stop-gradient energy: {self.config.stop_gradient_energy}",
            f"# Grad clip: {self.config.grad_clip}",
            f"# Optimizer: Adam(weight_decay={self.config.weight_decay})",
            f"# Energy bounding: spectral normalization on all energy network layers",
            f"#",
            f"# --- Dataset ---",
            f"# Train samples: {len(self.train_loader.dataset)}",
            f"# Batch size: {self.train_loader.batch_size}",
            f"# Steps/epoch: {len(self.train_loader)}",
            f"# Total steps: {self.total_steps}",
            f"#",
            f"# --- Model ---",
            f"# TaskNet params: {sum(p.numel() for p in self.task_net.parameters())}",
            f"# LossNet params: {sum(p.numel() for p in self.loss_net.parameters())}",
            f"#",
        ]
        for line in lines:
            self._log_to_file(line)

    def _close_exp_file(self) -> None:
        if self._exp_file is not None:
            self._exp_file.close()
            self._exp_file = None

    def _get_num_labels(self) -> int:
        sample = next(iter(self.train_loader))
        return sample[1].shape[1]

    def train(self, callbacks: list | None = None) -> dict:
        callbacks = callbacks or []
        history = {"train_loss": [], "val_f1": [], "test_f1": []}

        header = (
            "epoch\ttrain_loss\tenergy_loss\tbce_loss\tnce_loss\t"
            "val_micro_f1\tval_macro_f1\tval_sample_f1\t"
            "test_micro_f1\ttest_macro_f1\ttest_sample_f1\t"
            "task_lr\telapsed_sec"
        )
        self._log_to_file(header)

        train_start = time.time()

        for epoch in range(self.config.total_epochs):
            train_metrics = self._train_epoch(epoch)
            history["train_loss"].append(train_metrics["total_loss"])

            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._evaluate(self.val_loader)

            test_metrics = {}
            if self.test_loader is not None:
                test_metrics = self._evaluate(self.test_loader)
                history["test_f1"].append(test_metrics.get("micro_f1", 0))

            current_lr = self.task_opt.param_groups[0]["lr"]
            elapsed = time.time() - train_start

            # Console logging
            msg = (
                f"Epoch {epoch+1}/{self.config.total_epochs} "
                f"loss={train_metrics['total_loss']:.4f}"
                f" energy={train_metrics['energy_loss']:.4f}"
                f" bce={train_metrics['bce_loss']:.4f}"
                f" lr={current_lr:.1e}"
            )
            if val_metrics:
                msg += (
                    f" | val_sF1={val_metrics.get('sample_f1', 0):.4f}"
                    f" micro={val_metrics.get('micro_f1', 0):.4f}"
                )
            if test_metrics:
                msg += (
                    f" | test_sF1={test_metrics.get('sample_f1', 0):.4f}"
                    f" micro={test_metrics.get('micro_f1', 0):.4f}"
                )
            logger.info(msg)

            # File logging
            row = (
                f"{epoch+1}\t"
                f"{train_metrics['total_loss']:.6f}\t"
                f"{train_metrics['energy_loss']:.6f}\t"
                f"{train_metrics['bce_loss']:.6f}\t"
                f"{train_metrics['nce_loss']:.6f}\t"
                f"{val_metrics.get('micro_f1', 0):.6f}\t"
                f"{val_metrics.get('macro_f1', 0):.6f}\t"
                f"{val_metrics.get('sample_f1', 0):.6f}\t"
                f"{test_metrics.get('micro_f1', 0):.6f}\t"
                f"{test_metrics.get('macro_f1', 0):.6f}\t"
                f"{test_metrics.get('sample_f1', 0):.6f}\t"
                f"{current_lr:.6e}\t"
                f"{elapsed:.1f}"
            )
            self._log_to_file(row)

            for cb in callbacks:
                cb(self, epoch, {**train_metrics, **val_metrics, **test_metrics})

        self._close_exp_file()
        return history

    def _train_epoch(self, epoch: int) -> dict:
        self.task_net.train()
        self.loss_net.train()

        total_loss = 0
        total_energy = 0
        total_bce = 0
        total_nce = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            metrics = self._train_step(batch)
            total_loss += metrics["total"]
            total_energy += metrics["energy"]
            total_bce += metrics["bce"]
            total_nce += metrics["nce"]
            num_batches += 1

            pbar.set_postfix(loss=f"{metrics['total']:.4f}")

            self.global_step += 1
            self.diagnostics.increment_step()

        return {
            "total_loss": total_loss / num_batches,
            "energy_loss": total_energy / num_batches,
            "bce_loss": total_bce / num_batches,
            "nce_loss": total_nce / num_batches,
        }

    def _train_step(self, batch: tuple) -> dict:
        if len(batch) == 3:
            features, labels, indices = batch
        else:
            features, labels = batch
            indices = None

        features = features.to(self.device)
        labels = labels.to(self.device)
        if indices is not None:
            indices = indices.to(self.device)

        # Get current marginals for centering
        with torch.no_grad():
            if self.config.per_sample_ema and indices is not None:
                marginals = self.ema.get_marginals(indices)
            else:
                marginals = self.ema.get_marginals()
                if marginals.dim() == 1:
                    marginals = marginals.unsqueeze(0).expand(features.shape[0], -1)

        # Step 1: Update loss-net with contrastive ranking loss
        with torch.no_grad():
            y_pred_detached = self.task_net(features)

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            nce_loss = self._compute_nce_loss(
                features, labels, y_pred_detached, marginals
            )
        self.loss_opt.zero_grad()
        self.loss_scaler.scale(nce_loss).backward()
        self.loss_scaler.unscale_(self.loss_opt)
        nn.utils.clip_grad_norm_(
            self.loss_net.parameters(), self.config.grad_clip
        )
        self.loss_scaler.step(self.loss_opt)
        self.loss_scaler.update()

        # Step 2: Update task-net with -E + BCE
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            y_pred = self.task_net(features)
            energy = self.loss_net(features, y_pred, marginals)
            if self.config.stop_gradient_energy:
                energy = energy.detach()

            loss_dict = self.task_loss_fn(
                energy=energy,
                y_pred=y_pred,
                y_true=labels,
                phase="dynamic",
            )

        self.task_opt.zero_grad()
        self.task_scaler.scale(loss_dict["total"]).backward()
        self.task_scaler.unscale_(self.task_opt)
        nn.utils.clip_grad_norm_(
            self.task_net.parameters(), self.config.grad_clip
        )
        self.task_scaler.step(self.task_opt)
        self.task_scaler.update()

        # Update EMA centering with final predictions
        with torch.no_grad():
            final_pred = self.task_net(features)
            self.ema.update(final_pred.detach(), indices)

        metrics = {
            "total": loss_dict["total"].item(),
            "energy": loss_dict["energy"].item(),
            "bce": loss_dict["bce"].item(),
            "nce": nce_loss.item(),
        }

        if self.config.track_diagnostics and self.diagnostics.should_log():
            self._run_diagnostics(features, labels, final_pred.detach(), marginals)

        return metrics

    def _compute_nce_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        y_pred: torch.Tensor,
        marginals: torch.Tensor,
        log_prob_gt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Compute feature embeddings once
        input_features = self.loss_net.feature_network(features)

        # Precompute energy cache (label embeddings A for CREL, local scores for SEAL).
        # These depend only on x, so they are shared across GT and all K negatives.
        cache = self.loss_net.energy.precompute(input_features)

        # GT energy using precomputed cache (B samples)
        energy_gt = self.loss_net.energy.energy_from_precomputed(
            cache, labels.float(), marginals
        )

        # Sample negatives: (B, K, L)
        neg_samples = self.sampler(y_pred, num_samples=self.config.nce_samples)

        # Negative energy using batched einsum — avoids expanding A from (B,L,r) to (B*K,L,r)
        energy_neg = self.loss_net.energy.energy_neg_from_precomputed(
            cache, neg_samples, marginals
        )

        # Energy regularization: penalize large energy magnitudes
        if self.config.energy_reg > 0:
            all_energies = torch.cat([energy_gt.unsqueeze(1), energy_neg], dim=1)
            e_reg = self.config.energy_reg * (all_energies ** 2).mean()
        else:
            e_reg = 0.0

        # InfoNCE uses raw energy scores; NCE needs log-prob correction
        if self.config.contrastive_loss == "infonce":
            return self.contrastive_loss_fn(energy_gt, energy_neg) + e_reg

        # NCE: compute log-prob of GT and negatives under proposal
        if log_prob_gt is None:
            if self.config.nce_sampling == "gaussian_noise":
                sigma = self.config.nce_gaussian_sigma
                log_prob_gt = compute_log_prob_gaussian(labels.float(), y_pred, sigma)
            else:
                log_prob_gt = compute_log_prob(labels.float(), y_pred)

        if self.config.nce_sampling == "gaussian_noise":
            sigma = self.config.nce_gaussian_sigma
            log_prob_neg = compute_log_prob_gaussian(
                neg_samples, y_pred.unsqueeze(1).expand_as(neg_samples), sigma
            )
        else:
            log_prob_neg = compute_log_prob(
                neg_samples, y_pred.unsqueeze(1).expand_as(neg_samples)
            )

        return self.contrastive_loss_fn(energy_gt, energy_neg, log_prob_gt, log_prob_neg) + e_reg

    def _run_diagnostics(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        y_pred: torch.Tensor,
        marginals: torch.Tensor,
    ) -> None:
        energy_module = self.loss_net.get_energy_module()

        with torch.no_grad():
            input_features = self.loss_net.feature_network(features[:8])

        cos_sim = self.diagnostics.compute_gradient_cosine(
            y_pred=y_pred[:8],
            y_true=labels[:8],
            energy_fn=energy_module,
            input_features=input_features,
            y_marginals=marginals[:8],
        )
        if self.diagnostics.step % (self.config.log_interval * 10) == 0:
            logger.info(f"  [Diagnostic] gradient cosine similarity: {cos_sim:.4f}")

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> dict:
        self.task_net.eval()

        all_preds = []
        all_labels = []

        for batch in loader:
            if len(batch) == 3:
                features, labels, _ = batch
            else:
                features, labels = batch

            features = features.to(self.device)
            labels = labels.to(self.device)

            y_pred = self.task_net(features)
            all_preds.append(y_pred)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self.task_net.train()
        return compute_f1_metrics(all_preds, all_labels)

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "global_step": self.global_step,
                "task_net": self.task_net.state_dict(),
                "loss_net": self.loss_net.state_dict(),
                "task_opt": self.task_opt.state_dict(),
                "loss_opt": self.loss_opt.state_dict(),
                "ema": self.ema.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.global_step = ckpt["global_step"]
        self.task_net.load_state_dict(ckpt["task_net"])
        self.loss_net.load_state_dict(ckpt["loss_net"])
        self.task_opt.load_state_dict(ckpt["task_opt"])
        self.loss_opt.load_state_dict(ckpt["loss_opt"])
        self.ema.load_state_dict(ckpt["ema"])
