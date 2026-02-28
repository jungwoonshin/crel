"""Task-net loss combining CREL energy with BCE.

L_F(φ; θ) = -λ_1 · E_CREL(x, F_φ(x); θ) + λ_2 · Σ_j BCE(y_j, F_φ(x)_j)

The NCE ranking loss trains E to be HIGH for ground-truth labels.
Negating E in the task loss means minimising L_F MAXIMISES the energy,
pushing predictions toward configurations the energy rates highly
(i.e. ground-truth-like).  BCE provides marginal (per-label) supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskLoss(nn.Module):
    """Combined loss for training the task network.

    Combines the energy term (from the loss-net) with binary cross-entropy.
    The energy term provides structural supervision via label coupling,
    while BCE provides marginal supervision.
    """

    def __init__(
        self,
        lambda_energy: float = 1.0,
        lambda_bce: float = 1.0,
    ):
        """
        Args:
            lambda_energy: λ_1, weight on the energy term.
            lambda_bce: λ_2, weight on the BCE term.
        """
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_bce = lambda_bce

    def forward(
        self,
        energy: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        phase: str = "dynamic",
    ) -> dict[str, torch.Tensor]:
        """Compute the combined task-net loss.

        Args:
            energy: E_CREL(x, F_φ(x); θ), shape (batch,).
            y_pred: task-net predictions F_φ(x), shape (batch, L), in [0,1].
            y_true: ground truth labels, shape (batch, L), binary.
            phase: training phase. 'warmup' uses BCE only, 'dynamic' uses both.

        Returns:
            dict with keys:
                'total': total loss (scalar)
                'energy': energy term (scalar)
                'bce': BCE term (scalar)
        """
        # BCE loss (always computed)
        bce = F.binary_cross_entropy(y_pred, y_true, reduction="mean")

        if phase == "warmup":
            return {
                "total": self.lambda_bce * bce,
                "energy": torch.tensor(0.0, device=y_pred.device),
                "bce": bce,
            }

        # Energy loss: mean over batch
        energy_loss = energy.mean()

        # Negate energy: NCE trains E to be high for ground truth,
        # so maximising E (minimising -E) produces ground-truth-like predictions.
        total = -self.lambda_energy * energy_loss + self.lambda_bce * bce

        return {
            "total": total,
            "energy": energy_loss,
            "bce": bce,
        }
