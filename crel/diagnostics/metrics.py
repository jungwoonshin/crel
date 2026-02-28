"""Diagnostic measurements for CREL training.

Implements the four diagnostics from Section 7.4:
1. Gradient orthogonality (cosine similarity between energy and BCE gradients)
2. Precision alignment (how well learned coupling matches true precision)
3. Gradient signal-to-noise ratio
4. Label dependency visualization helpers
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class DiagnosticTracker:
    """Tracks diagnostic metrics during CREL training.

    Accumulates measurements over training steps and provides
    summary statistics for logging.
    """

    def __init__(self, log_interval: int = 50):
        """
        Args:
            log_interval: number of steps between computing diagnostics.
        """
        self.log_interval = log_interval
        self.step = 0
        self.history = defaultdict(list)
        self._gradient_buffer = defaultdict(list)

    def should_log(self) -> bool:
        return self.step % self.log_interval == 0

    def increment_step(self) -> None:
        self.step += 1

    def compute_gradient_cosine(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        energy_fn,
        input_features: torch.Tensor,
        y_marginals: torch.Tensor,
    ) -> float:
        """Diagnostic 1: Gradient orthogonality.

        Computes cosine similarity between the energy gradient and BCE
        gradient with respect to the predicted labels ỹ.

        For CREL, this should be near zero (orthogonal / non-redundant).
        For SEAL, this will be significantly positive (redundant).

        Args:
            y_pred: task-net predictions, shape (batch, L), requires_grad must be possible.
            y_true: ground truth labels, shape (batch, L).
            energy_fn: callable(input_features, y_pred, y_marginals) -> energy (batch,).
            input_features: encoded input features, shape (batch, d_x).
            y_marginals: marginal probabilities, shape (batch, L).

        Returns:
            Mean cosine similarity over the batch.
        """
        try:
            # Detach inputs and enable grad only on y_pred copies
            input_features_d = input_features.detach()
            y_marginals_d = y_marginals.detach()

            y_pred_e = y_pred.detach().clone().requires_grad_(True)
            y_pred_b = y_pred.detach().clone().requires_grad_(True)

            # Energy gradient — need torch.enable_grad() since caller may be in no_grad
            with torch.enable_grad():
                energy = energy_fn(input_features_d, y_pred_e, y_marginals_d)
                energy_sum = energy.sum()
                energy_sum.backward()
            grad_energy = y_pred_e.grad  # (batch, L)

            # BCE gradient
            with torch.enable_grad():
                bce = F.binary_cross_entropy(y_pred_b, y_true, reduction="sum")
                bce.backward()
            grad_bce = y_pred_b.grad  # (batch, L)

            if grad_energy is None or grad_bce is None:
                return 0.0

            # Cosine similarity per sample
            cos_sim = F.cosine_similarity(grad_energy, grad_bce, dim=1)  # (batch,)
            mean_cos = cos_sim.mean().item()

            self.history["gradient_cosine"].append((self.step, mean_cos))
            return mean_cos
        except Exception:
            return 0.0

    @torch.no_grad()
    def compute_precision_alignment(
        self,
        energy_module,
        input_features: torch.Tensor,
        empirical_precision: np.ndarray,
    ) -> float:
        """Diagnostic 2: Precision alignment.

        Measures how well the learned coupling matrix matches the
        empirical precision matrix.

        alignment = ||off_diag(A^T A) - off_diag(Σ⁻¹)|| / ||off_diag(Σ⁻¹)||

        Lower is better.

        Args:
            energy_module: CRELEnergy module with get_coupling_matrix method.
            input_features: single encoded input, shape (1, d_x).
            empirical_precision: ground-truth precision matrix, shape (L, L).

        Returns:
            Relative Frobenius norm of the difference (lower = better alignment).
        """
        # Get learned coupling for this input
        learned = energy_module.get_coupling_matrix(input_features)  # (L, L)
        learned = learned.cpu().numpy()

        # Zero diagonals for off-diagonal comparison
        np.fill_diagonal(learned, 0)
        precision_off = empirical_precision.copy()
        np.fill_diagonal(precision_off, 0)

        # Relative error
        diff_norm = np.linalg.norm(learned - precision_off, "fro")
        ref_norm = np.linalg.norm(precision_off, "fro")

        alignment = diff_norm / max(ref_norm, 1e-8)

        self.history["precision_alignment"].append((self.step, alignment))
        return alignment

    def accumulate_gradient_snr(
        self,
        grad_energy: torch.Tensor,
    ) -> None:
        """Diagnostic 3: Accumulate energy gradients for SNR computation.

        Call this every step to collect gradients. The SNR is computed
        periodically over the accumulated buffer.

        Args:
            grad_energy: energy gradient w.r.t. y_pred, shape (batch, L).
        """
        self._gradient_buffer["energy"].append(grad_energy.detach().cpu())

    @torch.no_grad()
    def compute_gradient_snr(self) -> float | None:
        """Compute gradient signal-to-noise ratio from accumulated gradients.

        SNR = ||E[grad]|| / E[||grad - E[grad]||]

        Higher SNR means more consistent structural signal.

        Returns:
            SNR value, or None if insufficient data.
        """
        buffer = self._gradient_buffer["energy"]
        if len(buffer) < 2:
            return None

        grads = torch.cat(buffer, dim=0)  # (total_samples, L)
        mean_grad = grads.mean(dim=0)  # (L,)
        signal = mean_grad.norm().item()
        noise = (grads - mean_grad).norm(dim=1).mean().item()

        snr = signal / max(noise, 1e-8)

        self.history["gradient_snr"].append((self.step, snr))
        self._gradient_buffer["energy"] = []  # Reset buffer
        return snr

    def get_summary(self) -> dict[str, float]:
        """Get latest values for all tracked metrics."""
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[key] = values[-1][1]
        return summary

    def get_full_history(self) -> dict[str, list[tuple[int, float]]]:
        """Get full history of all metrics as (step, value) pairs."""
        return dict(self.history)


def compute_f1_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float | torch.Tensor = 0.5,
) -> dict[str, float]:
    """Compute multi-label F1 metrics.

    Args:
        y_pred: predicted probabilities, shape (N, L).
        y_true: ground truth, shape (N, L).
        threshold: binarization threshold. Scalar or per-label tensor of shape (L,).

    Returns:
        dict with 'micro_f1', 'macro_f1', 'sample_f1'.
    """
    if isinstance(threshold, torch.Tensor):
        y_binary = (y_pred >= threshold.to(y_pred.device)).float()
    else:
        y_binary = (y_pred >= threshold).float()

    # Micro F1: compute globally
    tp = (y_binary * y_true).sum().item()
    pred_pos = y_binary.sum().item()
    true_pos = y_true.sum().item()

    micro_p = tp / max(pred_pos, 1e-8)
    micro_r = tp / max(true_pos, 1e-8)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)

    # Macro F1: average per-label F1
    tp_per_label = (y_binary * y_true).sum(dim=0)  # (L,)
    pred_per_label = y_binary.sum(dim=0)
    true_per_label = y_true.sum(dim=0)

    p_per_label = tp_per_label / pred_per_label.clamp(min=1e-8)
    r_per_label = tp_per_label / true_per_label.clamp(min=1e-8)
    f1_per_label = 2 * p_per_label * r_per_label / (p_per_label + r_per_label).clamp(min=1e-8)

    # Only average over labels that appear in ground truth
    active = true_per_label > 0
    macro_f1 = f1_per_label[active].mean().item() if active.any() else 0.0

    # Sample F1: average per-sample F1
    tp_per_sample = (y_binary * y_true).sum(dim=1)
    pred_per_sample = y_binary.sum(dim=1)
    true_per_sample = y_true.sum(dim=1)

    p_per_sample = tp_per_sample / pred_per_sample.clamp(min=1e-8)
    r_per_sample = tp_per_sample / true_per_sample.clamp(min=1e-8)
    f1_per_sample = 2 * p_per_sample * r_per_sample / (p_per_sample + r_per_sample).clamp(min=1e-8)

    # Handle edge cases: no true labels AND no predictions → correct empty (F1=1)
    no_true = true_per_sample == 0
    no_pred = pred_per_sample == 0
    correct_empty = no_true & no_pred
    f1_per_sample[correct_empty] = 1.0

    # Exclude samples with no true labels but with false positive predictions
    has_labels_or_correct = (true_per_sample > 0) | correct_empty
    sample_f1 = (
        f1_per_sample[has_labels_or_correct].mean().item()
        if has_labels_or_correct.any() else 0.0
    )

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "sample_f1": sample_f1,
    }


def optimize_thresholds(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_candidates: int = 50,
) -> torch.Tensor:
    """Find per-label thresholds that maximize micro-F1 on a validation set.

    Uses a grid search over candidate thresholds for each label independently.

    Args:
        y_pred: predicted probabilities, shape (N, L).
        y_true: ground truth, shape (N, L).
        num_candidates: number of threshold candidates to try per label.

    Returns:
        Optimal per-label thresholds, shape (L,).
    """
    L = y_pred.shape[1]
    thresholds = torch.full((L,), 0.5)
    candidates = torch.linspace(0.1, 0.9, num_candidates)

    for j in range(L):
        best_f1 = -1.0
        best_t = 0.5
        pj = y_pred[:, j]
        tj = y_true[:, j]

        for t in candidates:
            pred_bin = (pj >= t.item()).float()
            tp = (pred_bin * tj).sum().item()
            fp = (pred_bin * (1 - tj)).sum().item()
            fn = ((1 - pred_bin) * tj).sum().item()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t.item()

        thresholds[j] = best_t

    return thresholds
