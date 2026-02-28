"""CREL Energy Module: Centered Residual Energy Loss.

Computes a low-rank quadratic energy over centered label predictions,
optionally augmented with a higher-order softplus term.  The forward
pass is O(L * r) -- the L x L coupling matrix is never materialized.

All linear layers are spectrally normalized to bound the energy output
magnitude, preventing NCE scale degeneracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


def _sn(module: nn.Module) -> nn.Module:
    """Apply spectral normalization to a module with a weight parameter."""
    return spectral_norm(module)


class CRELEnergy(nn.Module):
    """Centered Residual Energy Loss for structured multi-label prediction.

    Parameters
    ----------
    num_labels : int
        Number of labels (L).
    input_dim : int
        Dimension of the input feature vector g(x) (d_x).
    rank : int
        Rank of the low-rank factorisation (r).
    label_embed_dim : int
        Dimension of each learnable label embedding e_i (d_e).
    proj_hidden_dim : int
        Hidden dimension of the two-layer projection MLP (d_h).
    higher_order : bool
        Whether to include the higher-order softplus energy term.
    higher_proj_dim : int
        Projection dimension for the centered predictions in the
        higher-order term (r').
    higher_hidden_dim : int
        Hidden dimension inside the higher-order softplus block (h').
    """

    def __init__(
        self,
        num_labels: int,
        input_dim: int,
        rank: int = 32,
        label_embed_dim: int = 32,
        proj_hidden_dim: int = 64,
        higher_order: bool = True,
        higher_proj_dim: int = 64,
        higher_hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.input_dim = input_dim
        self.rank = rank
        self.label_embed_dim = label_embed_dim
        self.proj_hidden_dim = proj_hidden_dim
        self.higher_order = higher_order
        self.higher_proj_dim = higher_proj_dim
        self.higher_hidden_dim = higher_hidden_dim

        # --- Label embeddings e_i, i = 1..L ---
        self.label_embeddings = nn.Embedding(num_labels, label_embed_dim)

        # --- Two-layer projection: [e_i ; g(x)] -> a_i(x) in R^r ---
        concat_dim = label_embed_dim + input_dim
        self.proj_w1 = nn.Linear(concat_dim, proj_hidden_dim)
        self.proj_w2 = nn.Linear(proj_hidden_dim, rank)

        # --- Higher-order term (optional) ---
        if self.higher_order:
            self.higher_proj = nn.Linear(num_labels, higher_proj_dim, bias=False)
            self.higher_linear = nn.Linear(higher_proj_dim, higher_hidden_dim)
            self.higher_weight = nn.Linear(higher_hidden_dim, 1, bias=False)

        # Initialize weights first, then apply spectral norm
        self._init_weights()
        self._apply_spectral_norm()

    def _init_weights(self) -> None:
        """Xavier / Glorot initialisation for all learnable parameters."""
        nn.init.xavier_uniform_(self.label_embeddings.weight)
        nn.init.xavier_uniform_(self.proj_w1.weight)
        nn.init.zeros_(self.proj_w1.bias)
        nn.init.xavier_uniform_(self.proj_w2.weight)
        nn.init.zeros_(self.proj_w2.bias)

        if self.higher_order:
            nn.init.xavier_uniform_(self.higher_proj.weight)
            nn.init.xavier_uniform_(self.higher_linear.weight)
            nn.init.zeros_(self.higher_linear.bias)
            nn.init.xavier_uniform_(self.higher_weight.weight)

    def _apply_spectral_norm(self) -> None:
        """Apply spectral normalization to all layers in the energy network."""
        self.label_embeddings = _sn(self.label_embeddings)
        self.proj_w1 = _sn(self.proj_w1)
        self.proj_w2 = _sn(self.proj_w2)

        if self.higher_order:
            self.higher_proj = _sn(self.higher_proj)
            self.higher_linear = _sn(self.higher_linear)
            self.higher_weight = _sn(self.higher_weight)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _compute_label_embeddings(
        self, input_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute all L label-conditioned embeddings a_i(x).

        Parameters
        ----------
        input_features : Tensor, shape (batch, d_x)

        Returns
        -------
        A : Tensor, shape (batch, L, r)
            A[b, i, :] = a_i(x_b) = W2 * ReLU(W1 * [e_i ; g(x_b)])
        """
        batch_size = input_features.size(0)

        label_indices = torch.arange(
            self.num_labels, device=input_features.device
        )
        e = self.label_embeddings(label_indices)  # (L, d_e)

        e_expanded = e.unsqueeze(0).expand(batch_size, -1, -1)
        g_expanded = input_features.unsqueeze(1).expand(-1, self.num_labels, -1)

        concat = torch.cat([e_expanded, g_expanded], dim=-1)

        hidden = F.relu(self.proj_w1(concat))   # (batch, L, d_h)
        A = self.proj_w2(hidden)                 # (batch, L, r)

        return A

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_features: torch.Tensor,
        y_pred: torch.Tensor,
        y_marginals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the CREL energy for a batch.

        Parameters
        ----------
        input_features : Tensor, shape (batch, d_x)
            Encoded input features g(x) from the feature network.
        y_pred : Tensor, shape (batch, L)
            Relaxed predicted labels in [0, 1].
        y_marginals : Tensor, shape (batch, L)
            Marginal probabilities mu(x) for centering (from EMA).

        Returns
        -------
        energy : Tensor, shape (batch,)
            CREL energy per sample.
        """
        y_bar = y_pred - y_marginals  # (batch, L)

        A = self._compute_label_embeddings(input_features)  # (batch, L, r)

        # Quadratic energy (O(Lr), never form L x L matrix)
        z = torch.einsum("blr,bl->br", A, y_bar)  # (batch, r)
        z_sq = 0.5 * (z * z).sum(dim=-1)  # (batch,)

        # Diagonal correction
        a_sq = (A * A).sum(dim=-1)               # (batch, L)
        y_bar_sq = y_bar * y_bar                  # (batch, L)
        diag_correction = 0.5 * (a_sq * y_bar_sq).sum(dim=-1)  # (batch,)

        e_quad = -z_sq + diag_correction  # (batch,)

        # Higher-order energy (optional)
        if self.higher_order:
            y_pooled = self.higher_proj(y_bar)              # (batch, r')
            h = F.softplus(self.higher_linear(y_pooled))    # (batch, h')
            e_higher = self.higher_weight(h).squeeze(-1)    # (batch,)
        else:
            e_higher = torch.zeros(
                y_pred.size(0), device=y_pred.device, dtype=y_pred.dtype
            )

        energy = e_quad + e_higher  # (batch,)
        return energy

    # ------------------------------------------------------------------
    # Precompute / batched API (used by NCE training for efficiency)
    # ------------------------------------------------------------------

    def precompute(self, input_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Precompute label embeddings and derived quantities that depend only on x.

        These are shared across the ground-truth label vector and all K negative
        samples, avoiding redundant computation in the NCE inner loop.

        Parameters
        ----------
        input_features : Tensor, shape (batch, d_x)

        Returns
        -------
        dict with keys ``'A'`` (batch, L, r) and ``'a_sq'`` (batch, L).
        """
        A = self._compute_label_embeddings(input_features)  # (batch, L, r)
        a_sq = (A * A).sum(dim=-1)  # (batch, L)
        return {"A": A, "a_sq": a_sq}

    def energy_from_precomputed(
        self,
        cache: dict[str, torch.Tensor],
        y_pred: torch.Tensor,
        y_marginals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy using precomputed label embeddings A.

        Parameters
        ----------
        cache : dict from :meth:`precompute`.
        y_pred : Tensor, shape (batch, L)
        y_marginals : Tensor, shape (batch, L)

        Returns
        -------
        energy : Tensor, shape (batch,)
        """
        A = cache["A"]
        a_sq = cache["a_sq"]
        y_bar = y_pred - y_marginals

        z = torch.einsum("blr,bl->br", A, y_bar)
        z_sq = 0.5 * (z * z).sum(dim=-1)
        y_bar_sq = y_bar * y_bar
        diag_correction = 0.5 * (a_sq * y_bar_sq).sum(dim=-1)
        e_quad = -z_sq + diag_correction

        if self.higher_order:
            y_pooled = self.higher_proj(y_bar)
            h = F.softplus(self.higher_linear(y_pooled))
            e_higher = self.higher_weight(h).squeeze(-1)
        else:
            e_higher = torch.zeros(
                y_pred.size(0), device=y_pred.device, dtype=y_pred.dtype
            )

        return e_quad + e_higher

    def energy_neg_from_precomputed(
        self,
        cache: dict[str, torch.Tensor],
        neg_samples: torch.Tensor,
        y_marginals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy for K negative samples per batch element using precomputed A.

        Uses batched einsum to avoid expanding A from (B, L, r) to (B*K, L, r).

        Parameters
        ----------
        cache : dict from :meth:`precompute`.
        neg_samples : Tensor, shape (batch, K, L)
        y_marginals : Tensor, shape (batch, L)

        Returns
        -------
        energy : Tensor, shape (batch, K)
        """
        A = cache["A"]      # (B, L, r)
        a_sq = cache["a_sq"]  # (B, L)
        y_bar = neg_samples - y_marginals.unsqueeze(1)  # (B, K, L)

        # Quadratic: z_k = A^T y_bar_k
        z = torch.einsum("blr,bkl->bkr", A, y_bar)  # (B, K, r)
        z_sq = 0.5 * (z * z).sum(dim=-1)  # (B, K)

        # Diagonal correction
        y_bar_sq = y_bar * y_bar  # (B, K, L)
        diag_correction = 0.5 * torch.einsum("bl,bkl->bk", a_sq, y_bar_sq)  # (B, K)

        e_quad = -z_sq + diag_correction  # (B, K)

        if self.higher_order:
            B, K, L = neg_samples.shape
            y_bar_flat = y_bar.reshape(B * K, L)
            y_pooled = self.higher_proj(y_bar_flat)
            h = F.softplus(self.higher_linear(y_pooled))
            e_higher = self.higher_weight(h).squeeze(-1).reshape(B, K)
        else:
            e_higher = torch.zeros(
                neg_samples.shape[:2], device=neg_samples.device, dtype=neg_samples.dtype
            )

        return e_quad + e_higher

    # ------------------------------------------------------------------
    # Diagnostic helpers (NOT used during training)
    # ------------------------------------------------------------------
    def get_label_embeddings(
        self, input_features: torch.Tensor
    ) -> torch.Tensor:
        """Return all L embeddings a_i(x) for each sample in the batch."""
        return self._compute_label_embeddings(input_features)

    def get_coupling_matrix(
        self, input_features: torch.Tensor
    ) -> torch.Tensor:
        """Return the full L x L coupling matrix A A^T for diagnostics.

        WARNING: O(L^2 r), never call during training.
        """
        if input_features.dim() == 1:
            input_features = input_features.unsqueeze(0)
        assert input_features.size(0) == 1
        A = self._compute_label_embeddings(input_features)  # (1, L, r)
        A = A.squeeze(0)  # (L, r)
        coupling = A @ A.t()  # (L, L)
        return coupling
