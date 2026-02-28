"""Task Network: multi-label classification MLP used at train and inference time."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class TaskNet(nn.Module):
    """MLP that maps input features to L sigmoid outputs (multi-label classification).

    The network is split into two logical parts:
      * **feature extractor** -- all hidden layers (Linear -> BatchNorm -> Activation -> Dropout)
      * **classifier head** -- final Linear -> Sigmoid

    Parameters
    ----------
    input_dim : int
        Dimension of raw input features.
    num_labels : int
        Number of output labels *L*.
    hidden_dims : list[int]
        Sizes of hidden layers.  Default ``[512, 512]``.
    dropout : float
        Dropout probability applied after each hidden activation.  Default ``0.3``.
    activation : str
        ``'relu'`` or ``'leaky_relu'``.  Default ``'relu'``.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dims: Sequence[int] = (512, 512),
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_labels = num_labels
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.activation = activation

        # ---- build activation factory ----
        act_fn = self._make_activation(activation)

        # ---- feature extractor (hidden layers) ----
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)

        # ---- classifier head ----
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, num_labels),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict label probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted label probabilities of shape ``(batch, num_labels)``.
        """
        features = self.feature_extractor(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return hidden representation before the classifier head.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Hidden representation of shape ``(batch, hidden_dims[-1])``.
        """
        return self.feature_extractor(x)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_activation(name: str):
        """Return a *callable* that creates the requested activation module."""
        name_lower = name.lower()
        if name_lower == "relu":
            return nn.ReLU
        if name_lower in ("leaky_relu", "leakyrelu"):
            return nn.LeakyReLU
        raise ValueError(
            f"Unsupported activation '{name}'. Choose 'relu' or 'leaky_relu'."
        )
