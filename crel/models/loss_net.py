"""Loss Network: wraps a feature encoder and an energy function for training."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from crel.energy import CRELEnergy, SEALEnergy


class FeatureNetwork(nn.Module):
    """MLP that encodes raw inputs into a fixed-size feature vector g(x).

    The architecture mirrors :class:`TaskNet`'s feature extractor
    (Linear -> BatchNorm -> ReLU -> Dropout for each hidden layer)
    but has no classification head.

    Parameters
    ----------
    input_dim : int
        Raw input dimension.
    output_dim : int
        Feature encoding dimension *d_x* (size of the last hidden layer's
        output, i.e. the final ``hidden_dims`` entry equals ``output_dim``
        when default settings are used, but the last Linear maps to
        ``output_dim`` explicitly).
    hidden_dims : list[int]
        Hidden-layer sizes *before* the final projection.  Default ``[512, 512]``.
    dropout : float
        Dropout probability.  Default ``0.3``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (512, 512),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in list(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final projection to the desired output dimension.
        # Spectrally normalized so that the feature network output is bounded,
        # completing the Lipschitz chain into the energy module.
        final_linear = nn.Linear(in_dim, output_dim)
        layers.append(spectral_norm(final_linear))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw inputs.

        Parameters
        ----------
        x : torch.Tensor
            Raw input of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Encoded features g(x) of shape ``(batch, output_dim)``.
        """
        return self.net(x)


class LossNet(nn.Module):
    """Combines a :class:`FeatureNetwork` with an energy function.

    Used **only** during training to compute the energy term of the loss.

    Parameters
    ----------
    input_dim : int
        Raw input dimension.
    num_labels : int
        Number of labels *L*.
    energy_type : str
        ``'crel'`` or ``'seal'``.
    feature_hidden_dims : list[int]
        Hidden-layer sizes for the feature network.  Default ``[512, 512]``.
    feature_dropout : float
        Dropout probability for the feature network.  Default ``0.3``.
    **energy_kwargs
        Additional keyword arguments forwarded to the energy constructor.

        For CREL (``CRELEnergy``): ``rank``, ``label_embed_dim``,
        ``proj_hidden_dim``, ``higher_order``, ``higher_proj_dim``,
        ``higher_hidden_dim``.

        For SEAL (``SEALEnergy``): ``global_hidden_dim``.
    """

    _ENERGY_REGISTRY: dict[str, type[nn.Module]] = {
        "crel": CRELEnergy,
        "seal": SEALEnergy,
    }

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        energy_type: str = "crel",
        feature_hidden_dims: Sequence[int] = (512, 512),
        feature_dropout: float = 0.3,
        **energy_kwargs: Any,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_labels = num_labels
        self.energy_type = energy_type.lower()

        if self.energy_type not in self._ENERGY_REGISTRY:
            raise ValueError(
                f"Unknown energy_type '{energy_type}'. "
                f"Choose from {list(self._ENERGY_REGISTRY.keys())}."
            )

        # Determine the feature dimension expected by the energy module.
        # Convention: the last hidden dim is also used as the feature-network
        # output dimension (d_x) that feeds into the energy function.
        feature_output_dim = list(feature_hidden_dims)[-1] if feature_hidden_dims else input_dim

        # ---- Feature encoder g(x) ----
        self.feature_network = FeatureNetwork(
            input_dim=input_dim,
            output_dim=feature_output_dim,
            hidden_dims=feature_hidden_dims,
            dropout=feature_dropout,
        )

        # ---- Energy function E(g(x), y_pred, y_marginals) ----
        energy_cls = self._ENERGY_REGISTRY[self.energy_type]
        self.energy = energy_cls(
            input_dim=feature_output_dim,
            num_labels=num_labels,
            **energy_kwargs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_marginals: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the energy for the given inputs and predictions.

        Parameters
        ----------
        x : torch.Tensor
            Raw input of shape ``(batch, input_dim)``.
        y_pred : torch.Tensor
            Predicted labels of shape ``(batch, L)``.
        y_marginals : torch.Tensor
            Marginal probabilities used for centering, shape ``(batch, L)``.

        Returns
        -------
        torch.Tensor
            Energy values of shape ``(batch,)``.
        """
        input_features = self.feature_network(x)
        return self.energy(input_features, y_pred, y_marginals)

    def get_energy_module(self) -> nn.Module:
        """Return the underlying energy module (``CRELEnergy`` or ``SEALEnergy``)."""
        return self.energy
