"""CREL trainer with cooperative energy training.

Works with any dataset provided by create_data_loaders(), including MEKA-fold
datasets: bibtex, delicious, genbase (train=folds 1–6, val=7–8, test=9–10).

Per batch:
  1. Update loss-net with contrastive ranking loss (InfoNCE or NCE)
  2. Update task-net with BCE + cooperative energy refinement:
     - Compute energy-refined target via gradient ascent on E(x,y) w.r.t. y
     - Train task-net to match refined target with MSE

Both networks agree on the direction — no adversarial instability.
"""

import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from crel.losses.nce import NCELoss, InfoNCELoss, compute_log_prob, compute_log_prob_gaussian
from crel.training.ema import EMACenter
from crel.training.sampling import get_sampler
from crel.diagnostics.metrics import DiagnosticTracker, compute_f1_metrics, optimize_thresholds

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
    lambda_energy: float = 1.0  # weight on cooperative energy refinement loss
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

    # Inference-time energy refinement
    refinement_steps: int = 5
    refinement_step_size: float = 0.1

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
    """Trainer implementing cooperative energy training for CREL.

    Per batch:
      1. Update loss-net with contrastive ranking loss
      2. Compute energy-refined target: ŷ_ref = clamp(ŷ + ∇_ŷ E, 0, 1)
      3. Update task-net with BCE(ŷ,y) + λ_E·MSE(ŷ, ŷ_ref)
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

        # Optimizers (AdamW for decoupled weight decay)
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

        # Cosine annealing LR schedulers
        # Filter harmless PyTorch warning about scheduler.step() ordering.
        # Our schedulers step after _train_epoch which calls optimizer.step().
        warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step\\(\\)` before")
        self.task_scheduler = CosineAnnealingLR(
            self.task_opt,
            T_max=self.config.total_epochs,
            eta_min=self.config.task_net_lr * 0.01,
        )
        self.loss_scheduler = CosineAnnealingLR(
            self.loss_opt,
            T_max=self.config.total_epochs,
            eta_min=self.config.loss_net_lr * 0.01,
        )

        # Contrastive loss for loss-net
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

        # Best model tracking
        self.best_val_metric = -float("inf")
        self.best_epoch = -1
        self.best_state = None

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
            f"# Training protocol: cooperative energy refinement",
            f"# Task-net LR: {self.config.task_net_lr}",
            f"# Loss-net LR: {self.config.loss_net_lr}",
            f"# Lambda energy (coop): {self.config.lambda_energy}",
            f"# Lambda BCE: {self.config.lambda_bce}",
            f"# EMA beta: {self.config.ema_beta}",
            f"# Contrastive loss: {self.config.contrastive_loss}",
            f"# InfoNCE temperature: {self.config.infonce_temperature}",
            f"# NCE samples: {self.config.nce_samples}",
            f"# Energy reg: {self.config.energy_reg}",
            f"# Grad clip: {self.config.grad_clip}",
            f"# Optimizer: AdamW(weight_decay={self.config.weight_decay})",
            f"# LR scheduler: CosineAnnealing(T_max={self.config.total_epochs})",
            f"# Best model: tracked on val sample_f1",
            f"# Refinement steps: {self.config.refinement_steps}",
            f"# Refinement step size: {self.config.refinement_step_size}",
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
            "epoch\ttrain_loss\tbce_loss\tcoop_loss\tnce_loss\t"
            "val_micro_f1\tval_macro_f1\tval_sample_f1\t"
            "test_micro_f1\ttest_macro_f1\ttest_sample_f1\t"
            "task_lr\telapsed_sec"
        )
        self._log_to_file(header)

        train_start = time.time()

        for epoch in range(self.config.total_epochs):
            train_metrics = self._train_epoch(epoch)
            history["train_loss"].append(train_metrics["total_loss"])

            # Step LR schedulers
            self.task_scheduler.step()
            self.loss_scheduler.step()

            # Evaluate with threshold optimization on val, applied to test
            val_metrics = {}
            val_thresholds = 0.5
            if self.val_loader is not None:
                val_preds, val_labels = self._collect_predictions(self.val_loader)
                val_thresholds = optimize_thresholds(val_preds, val_labels)
                val_metrics = compute_f1_metrics(val_preds, val_labels, threshold=val_thresholds)

            test_metrics = {}
            if self.test_loader is not None:
                test_metrics = self._evaluate(self.test_loader, threshold=val_thresholds)
                history["test_f1"].append(test_metrics.get("micro_f1", 0))

            # Best model tracking on val sample_f1
            if val_metrics:
                current_metric = val_metrics.get("sample_f1", 0)
                if current_metric > self.best_val_metric:
                    self.best_val_metric = current_metric
                    self.best_epoch = epoch + 1
                    self.best_state = {
                        "task_net": {k: v.cpu().clone() for k, v in self.task_net.state_dict().items()},
                        "loss_net": {k: v.cpu().clone() for k, v in self.loss_net.state_dict().items()},
                    }

            current_lr = self.task_opt.param_groups[0]["lr"]
            elapsed = time.time() - train_start

            # Console logging
            msg = (
                f"Epoch {epoch+1}/{self.config.total_epochs} "
                f"loss={train_metrics['total_loss']:.4f}"
                f" bce={train_metrics['bce_loss']:.4f}"
                f" coop={train_metrics['coop_loss']:.4f}"
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
            if self.best_epoch > 0:
                msg += f" | best@{self.best_epoch}={self.best_val_metric:.4f}"
            logger.info(msg)

            # File logging
            row = (
                f"{epoch+1}\t"
                f"{train_metrics['total_loss']:.6f}\t"
                f"{train_metrics['bce_loss']:.6f}\t"
                f"{train_metrics['coop_loss']:.6f}\t"
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

        # Restore best model
        if self.best_state is not None:
            self.task_net.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_state["task_net"].items()}
            )
            self.loss_net.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_state["loss_net"].items()}
            )
            logger.info(
                "Restored best model from epoch %d (val_sF1=%.4f)",
                self.best_epoch, self.best_val_metric,
            )

            # Final eval with best model — optimize thresholds on val, apply to test
            if self.test_loader is not None:
                best_threshold = 0.5
                if self.val_loader is not None:
                    val_preds, val_labels = self._collect_predictions(self.val_loader)
                    best_threshold = optimize_thresholds(val_preds, val_labels)
                best_test = self._evaluate(self.test_loader, threshold=best_threshold)
                logger.info(
                    "Best model test: sF1=%.4f micro=%.4f macro=%.4f",
                    best_test["sample_f1"], best_test["micro_f1"], best_test["macro_f1"],
                )
                self._log_to_file(
                    f"# Best model (epoch {self.best_epoch}): "
                    f"test_sF1={best_test['sample_f1']:.6f} "
                    f"test_micro={best_test['micro_f1']:.6f}"
                )

        self._close_exp_file()
        return history

    def _train_epoch(self, epoch: int) -> dict:
        self.task_net.train()
        self.loss_net.train()

        total_loss = 0
        total_bce = 0
        total_coop = 0
        total_nce = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            metrics = self._train_step(batch)
            total_loss += metrics["total"]
            total_bce += metrics["bce"]
            total_coop += metrics["coop"]
            total_nce += metrics["nce"]
            num_batches += 1

            pbar.set_postfix(loss=f"{metrics['total']:.4f}")

            self.global_step += 1
            self.diagnostics.increment_step()

        return {
            "total_loss": total_loss / num_batches,
            "bce_loss": total_bce / num_batches,
            "coop_loss": total_coop / num_batches,
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

        # Step 2: Update task-net with BCE + cooperative energy refinement
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            y_pred = self.task_net(features)

            # BCE loss
            with torch.amp.autocast(device_type=self.device.type, enabled=False):
                bce_loss = F.binary_cross_entropy(
                    y_pred.float(), labels.float(), reduction="mean"
                )

            # Cooperative energy refinement: one gradient ascent step on E w.r.t. y
            y_refined = self._compute_cooperative_target(
                features, y_pred, marginals
            )
            coop_loss = F.mse_loss(y_pred.float(), y_refined)

            total_loss = (
                self.config.lambda_bce * bce_loss
                + self.config.lambda_energy * coop_loss
            )

        self.task_opt.zero_grad()
        self.task_scaler.scale(total_loss).backward()
        self.task_scaler.unscale_(self.task_opt)
        nn.utils.clip_grad_norm_(
            self.task_net.parameters(), self.config.grad_clip
        )
        self.task_scaler.step(self.task_opt)
        self.task_scaler.update()

        # Update EMA centering with fresh predictions
        with torch.no_grad():
            fresh_pred = self.task_net(features)
            self.ema.update(fresh_pred.float(), indices)

        metrics = {
            "total": total_loss.item(),
            "bce": bce_loss.item(),
            "coop": coop_loss.item(),
            "nce": nce_loss.item(),
        }

        if self.config.track_diagnostics and self.diagnostics.should_log():
            self._run_diagnostics(features, labels, fresh_pred.detach(), marginals)

        return metrics

    def _compute_cooperative_target(
        self,
        features: torch.Tensor,
        y_pred: torch.Tensor,
        marginals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy-refined prediction target via one gradient ascent step.

        ŷ_ref = clamp(ŷ + ∇_ŷ E(x, ŷ), 0, 1)

        The energy gradient tells the task-net which direction makes predictions
        more GT-like according to the energy function. Uses precomputed cache
        and torch.autograd.grad (no grad flow to loss-net params).

        Returns:
            y_refined: detached refined target, shape (batch, L).
        """
        y = y_pred.detach().float().requires_grad_(True)

        # Precompute input-dependent cache (shared, does not depend on y)
        with torch.no_grad():
            input_features = self.loss_net.feature_network(features)
            cache = self.loss_net.energy.precompute(input_features)

        energy = self.loss_net.energy.energy_from_precomputed(
            cache, y, marginals.detach()
        )
        # Gradient of energy w.r.t. y only (not loss-net params)
        grad_y = torch.autograd.grad(energy.sum(), y)[0]
        y_refined = (y.detach() + grad_y).clamp(0, 1)

        return y_refined.detach()

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

        # Precompute energy cache (depends only on x)
        cache = self.loss_net.energy.precompute(input_features)

        # GT energy
        energy_gt = self.loss_net.energy.energy_from_precomputed(
            cache, labels.float(), marginals
        )

        # Sample negatives: (B, K, L)
        neg_samples = self.sampler(y_pred, num_samples=self.config.nce_samples)

        # Negative energy using batched einsum
        energy_neg = self.loss_net.energy.energy_neg_from_precomputed(
            cache, neg_samples, marginals
        )

        # --- Root-cause diagnostic: log once on first few steps ---
        if self.global_step < 3:
            def _stats(t: torch.Tensor, name: str) -> None:
                with torch.no_grad():
                    t = t.float()
                    nan_c = torch.isnan(t).sum().item()
                    inf_c = torch.isinf(t).sum().item()
                    ok = t[~(torch.isnan(t) | torch.isinf(t))]
                    if ok.numel() > 0:
                        logger.info(
                            "[nce_diagnostic] %s: min=%.4f max=%.4f mean=%.4f nan=%d inf=%d",
                            name, ok.min().item(), ok.max().item(), ok.mean().item(), nan_c, inf_c,
                        )
                    else:
                        logger.info("[nce_diagnostic] %s: all nan/inf (nan=%d inf=%d)", name, nan_c, inf_c)
            _stats(input_features, "input_features")
            _stats(energy_gt, "energy_gt")
            _stats(energy_neg, "energy_neg")
            with torch.no_grad():
                infonce_only = self.contrastive_loss_fn(energy_gt, energy_neg)
                logger.info("[nce_diagnostic] infonce_loss (no e_reg)=%.6f", infonce_only.item())

        # Energy regularization
        if self.config.energy_reg > 0:
            all_energies = torch.cat([energy_gt.unsqueeze(1), energy_neg], dim=1)
            e_reg = self.config.energy_reg * (all_energies ** 2).mean()
        else:
            e_reg = 0.0

        if self.global_step < 3:
            with torch.no_grad():
                if isinstance(e_reg, torch.Tensor):
                    logger.info("[nce_diagnostic] e_reg=%.6f", e_reg.item())
                else:
                    logger.info("[nce_diagnostic] e_reg=%.6f", float(e_reg))
                total_nce = self.contrastive_loss_fn(energy_gt, energy_neg)
                total_nce = total_nce + (e_reg if isinstance(e_reg, torch.Tensor) else 0.0)
                logger.info("[nce_diagnostic] total_nce_loss=%.6f", total_nce.item())

        # InfoNCE or NCE
        if self.config.contrastive_loss == "infonce":
            return self.contrastive_loss_fn(energy_gt, energy_neg) + e_reg

        # NCE: compute log-prob
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

    def _collect_predictions(
        self, loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference with optional energy refinement. Returns (preds, labels)."""
        self.task_net.eval()
        self.loss_net.eval()

        all_preds = []
        all_labels = []
        T = self.config.refinement_steps
        alpha = self.config.refinement_step_size

        for batch in loader:
            if len(batch) == 3:
                features, labels, indices = batch
            else:
                features, labels = batch
                indices = None

            features = features.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                y_pred = self.task_net(features)

                # Get marginals for centering
                if self.config.per_sample_ema and indices is not None:
                    indices = indices.to(self.device)
                    marginals = self.ema.get_marginals(indices)
                else:
                    marginals = self.ema.get_marginals()
                    if marginals.dim() == 1:
                        marginals = marginals.unsqueeze(0).expand(features.shape[0], -1)

                feat = self.loss_net.feature_network(features.float())
                cache = self.loss_net.energy.precompute(feat)

            # Multi-step gradient ascent to refine predictions using learned energy
            if T > 0:
                y = y_pred.detach().clone()
                for _ in range(T):
                    y.requires_grad_(True)
                    energy = self.loss_net.energy.energy_from_precomputed(
                        cache, y.float(), marginals.float()
                    )
                    grad_y = torch.autograd.grad(energy.sum(), y)[0]
                    y = (y.detach() + alpha * grad_y).clamp(0, 1)
                y_pred = y.detach()

            all_preds.append(y_pred)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        self.task_net.train()
        self.loss_net.train()
        return all_preds, all_labels

    def _evaluate(
        self,
        loader: DataLoader,
        threshold: float | torch.Tensor = 0.5,
    ) -> dict:
        preds, labels = self._collect_predictions(loader)
        return compute_f1_metrics(preds, labels, threshold=threshold)

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
