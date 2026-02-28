"""SEAL Energy Module: Structured Energy-based Adversarial Learning baseline.

Implements the original SEAL energy function consisting of a local
(per-label bilinear) term and a global (softplus) term.  This module
serves as the baseline against which CREL is compared.

All linear layers are spectrally normalized to bound energy magnitude.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class SEALEnergy(nn.Module):
    """SEAL energy function for structured multi-label prediction.

    Parameters
    ----------
    num_labels : int
        Number of labels (L).
    input_dim : int
        Dimension of the input feature vector T_E(x) (d_x).
    global_hidden_dim : int
        Hidden dimension of the global softplus block (h).
    """

    def __init__(
        self,
        num_labels: int,
        input_dim: int,
        global_hidden_dim: int = 150,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.input_dim = input_dim
        self.global_hidden_dim = global_hidden_dim

        # --- Local energy: E_local = sum_i y_i * (b_i^T T_E(x)) ---
        self.local_weight = nn.Linear(input_dim, num_labels, bias=False)

        # --- Global energy: E_global = v^T softplus(M y) ---
        self.global_linear = nn.Linear(num_labels, global_hidden_dim)
        self.global_weight = nn.Linear(global_hidden_dim, 1, bias=False)

        # Initialize then apply spectral norm
        self._init_weights()
        self._apply_spectral_norm()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.local_weight.weight)
        nn.init.xavier_uniform_(self.global_linear.weight)
        nn.init.zeros_(self.global_linear.bias)
        nn.init.xavier_uniform_(self.global_weight.weight)

    def _apply_spectral_norm(self) -> None:
        self.local_weight = spectral_norm(self.local_weight)
        self.global_linear = spectral_norm(self.global_linear)
        self.global_weight = spectral_norm(self.global_weight)

    def forward(
        self,
        input_features: torch.Tensor,
        y_pred: torch.Tensor,
        y_marginals: torch.Tensor,
    ) -> torch.Tensor:
        # --- Local energy ---
        scores = self.local_weight(input_features)  # (batch, L)
        e_local = (y_pred * scores).sum(dim=-1)     # (batch,)

        # --- Global energy ---
        h = F.softplus(self.global_linear(y_pred))          # (batch, h)
        e_global = self.global_weight(h).squeeze(-1)        # (batch,)

        energy = e_local + e_global  # (batch,)
        return energy

    # ------------------------------------------------------------------
    # Precompute / batched API (used by NCE training for efficiency)
    # ------------------------------------------------------------------

    def precompute(self, input_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Precompute local scores that depend only on x."""
        scores = self.local_weight(input_features)  # (B, L)
        return {"scores": scores}

    def energy_from_precomputed(
        self,
        cache: dict[str, torch.Tensor],
        y_pred: torch.Tensor,
        y_marginals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy using precomputed local scores. Returns (B,)."""
        scores = cache["scores"]
        e_local = (y_pred * scores).sum(dim=-1)
        h = F.softplus(self.global_linear(y_pred))
        e_global = self.global_weight(h).squeeze(-1)
        return e_local + e_global

    def energy_neg_from_precomputed(
        self,
        cache: dict[str, torch.Tensor],
        neg_samples: torch.Tensor,
        y_marginals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy for K negatives using precomputed scores. Returns (B, K)."""
        scores = cache["scores"]  # (B, L)
        e_local = torch.einsum("bl,bkl->bk", scores, neg_samples)  # (B, K)
        B, K, L = neg_samples.shape
        neg_flat = neg_samples.reshape(B * K, L)
        h = F.softplus(self.global_linear(neg_flat))
        e_global = self.global_weight(h).squeeze(-1).reshape(B, K)
        return e_local + e_global
