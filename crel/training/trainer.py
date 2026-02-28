"""CREL trainer with SEAL-dynamic minimax protocol.

Implements alternating optimization of loss-net and task-net from step 1,
matching the SEAL paper's minimax training structure:

  Per batch, for each outer step (num_steps_task_net times):
    - Inner loop: update loss-net num_steps_loss_net times with NCE
    - Outer step: update task-net once with -E + BCE

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

from crel.losses.nce import NCELoss, compute_log_prob, compute_log_prob_gaussian
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

    # Multi-step minimax (matching SEAL protocol)
    num_steps_loss_net: int = 12  # inner NCE steps for loss-net per outer step
    num_steps_task_net: int = 5   # outer task-net steps per batch

    # Loss weights
    lambda_energy: float = 1.0
    lambda_bce: float = 1.0

    # EMA centering
    ema_beta: float = 0.99
    ema_warmup_beta: float = 0.9
    ema_warmup_fraction: float = 0.05  # fraction of total steps for EMA beta annealing
    per_sample_ema: bool = True

    # NCE
    nce_samples: int = 32
    nce_sampling: str = "bernoulli"
    nce_gaussian_sigma: float = 0.3

    # Gradient clipping (task-net only, matching SEAL)
    grad_clip: float = 10.0

    # LR scheduling
    lr_patience: int = 5      # epochs without improvement before reducing LR
    lr_factor: float = 0.5    # factor to reduce LR by

    # Diagnostics
    log_interval: int = 50
    track_diagnostics: bool = True

    # Experiment logging
    experiment_dir: str = ""
    experiment_description: str = ""

    # Device
    device: str = "cuda"


class CRELTrainer:
    """Trainer implementing SEAL-dynamic minimax optimization.

    Per batch, runs a nested loop matching the SEAL protocol:
      - Outer loop (num_steps_task_net): update task-net with -E + BCE
        - Inner loop (num_steps_loss_net): update loss-net with NCE

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

        # Optimizers (AdamW matching SEAL)
        self.task_opt = torch.optim.AdamW(
            self.task_net.parameters(),
            lr=self.config.task_net_lr,
            weight_decay=self.config.weight_decay,
        )
        self.loss_opt = torch.optim.AdamW(
            self.loss_net.parameters(),
            lr=self.config.loss_net_lr,
            weight_decay=self.config.weight_decay,
        )

        # LR scheduler (ReduceOnPlateau on task-net, matching SEAL)
        self.task_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.task_opt,
            mode="max",
            factor=self.config.lr_factor,
            patience=self.config.lr_patience,
        )

        # Losses
        self.task_loss_fn = TaskLoss(
            lambda_energy=self.config.lambda_energy,
            lambda_bce=self.config.lambda_bce,
        )
        self.nce_loss_fn = NCELoss(num_samples=self.config.nce_samples)

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
            f"# Training protocol: SEAL-dynamic minimax",
            f"# Steps per batch: {self.config.num_steps_task_net} outer x {self.config.num_steps_loss_net} inner",
            f"# Task-net LR: {self.config.task_net_lr}",
            f"# Loss-net LR: {self.config.loss_net_lr}",
            f"# Lambda energy: {self.config.lambda_energy}",
            f"# Lambda BCE: {self.config.lambda_bce}",
            f"# EMA beta: {self.config.ema_beta}",
            f"# NCE samples: {self.config.nce_samples}",
            f"# NCE sampling: {self.config.nce_sampling}",
            f"# Grad clip (task-net): {self.config.grad_clip}",
            f"# LR scheduler: ReduceOnPlateau(patience={self.config.lr_patience}, factor={self.config.lr_factor})",
            f"# Optimizer: AdamW(weight_decay={self.config.weight_decay})",
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

            # Step LR scheduler based on validation metric
            scheduler_metric = val_metrics.get("sample_f1", test_metrics.get("sample_f1", 0))
            self.task_scheduler.step(scheduler_metric)
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
                    f" | val_instF1={val_metrics.get('sample_f1', 0):.4f}"
                    f" micro={val_metrics.get('micro_f1', 0):.4f}"
                )
            if test_metrics:
                msg += (
                    f" | test_instF1={test_metrics.get('sample_f1', 0):.4f}"
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

        nce_loss_accum = 0.0
        last_loss_dict = None

        # Minimax loop: outer = task-net steps, inner = loss-net steps
        for _outer in range(self.config.num_steps_task_net):

            # Get frozen task-net predictions for this outer step
            with torch.no_grad():
                y_pred_detached = self.task_net(features)

            # Inner loop: multiple loss-net (NCE) updates
            for _inner in range(self.config.num_steps_loss_net):
                nce_loss = self._compute_nce_loss(
                    features, labels, y_pred_detached, marginals
                )
                self.loss_opt.zero_grad()
                nce_loss.backward()
                self.loss_opt.step()
                nce_loss_accum += nce_loss.item()

            # Outer step: update task-net with -E + BCE
            y_pred = self.task_net(features)
            energy = self.loss_net(features, y_pred, marginals)

            loss_dict = self.task_loss_fn(
                energy=energy,
                y_pred=y_pred,
                y_true=labels,
                phase="dynamic",
            )

            self.task_opt.zero_grad()
            loss_dict["total"].backward()
            nn.utils.clip_grad_norm_(
                self.task_net.parameters(), self.config.grad_clip
            )
            self.task_opt.step()
            last_loss_dict = loss_dict

        # Update EMA centering with final predictions
        with torch.no_grad():
            final_pred = self.task_net(features)
            self.ema.update(final_pred.detach(), indices)

        total_inner = self.config.num_steps_task_net * self.config.num_steps_loss_net
        metrics = {
            "total": last_loss_dict["total"].item(),
            "energy": last_loss_dict["energy"].item(),
            "bce": last_loss_dict["bce"].item(),
            "nce": nce_loss_accum / max(total_inner, 1),
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
    ) -> torch.Tensor:
        energy_gt = self.loss_net(features, labels.float(), marginals)

        neg_samples = self.sampler(y_pred, num_samples=self.config.nce_samples)
        batch_size = features.shape[0]
        K = self.config.nce_samples

        features_expanded = features.unsqueeze(1).expand(batch_size, K, -1)
        features_flat = features_expanded.reshape(batch_size * K, -1)
        neg_flat = neg_samples.reshape(batch_size * K, -1)
        marginals_expanded = marginals.unsqueeze(1).expand(batch_size, K, -1)
        marginals_flat = marginals_expanded.reshape(batch_size * K, -1)

        energy_neg_flat = self.loss_net(features_flat, neg_flat, marginals_flat)
        energy_neg = energy_neg_flat.reshape(batch_size, K)

        # Use the correct log-prob for the sampling method
        if self.config.nce_sampling == "gaussian_noise":
            sigma = self.config.nce_gaussian_sigma
            log_prob_gt = compute_log_prob_gaussian(labels.float(), y_pred, sigma)
            log_prob_neg = compute_log_prob_gaussian(
                neg_samples, y_pred.unsqueeze(1).expand_as(neg_samples), sigma
            )
        else:
            log_prob_gt = compute_log_prob(labels.float(), y_pred)
            log_prob_neg = compute_log_prob(
                neg_samples, y_pred.unsqueeze(1).expand_as(neg_samples)
            )

        return self.nce_loss_fn(energy_gt, energy_neg, log_prob_gt, log_prob_neg)

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
                "task_scheduler": self.task_scheduler.state_dict(),
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
        if "task_scheduler" in ckpt:
            self.task_scheduler.load_state_dict(ckpt["task_scheduler"])
