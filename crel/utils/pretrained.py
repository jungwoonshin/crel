"""Utilities for loading pretrained weights (e.g. SEAL) into CREL models."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _set_if_present(
    sd: dict[str, torch.Tensor],
    keys: str | tuple[str, ...],
    value: torch.Tensor | None,
    device: torch.device | str,
) -> None:
    """Set sd[key] = value.to(device) for the first key that exists in sd. value may be None."""
    if value is None:
        return
    keys_tuple = (keys,) if isinstance(keys, str) else keys
    for k in keys_tuple:
        if k in sd:
            sd[k] = value.to(device)
            return


def load_seal_pretrained_into_crel(
    ckpt_path: str | None,
    task_net: nn.Module,
    loss_net: nn.Module,
    device: torch.device | str,
    ckpt: dict[str, Any] | None = None,
) -> None:
    """Load SEAL-format checkpoint (task_nn, score_nn) into CREL task_net and loss_net.

    Expects CREL models built with architecture compatible with SEAL bibtex:
    - task_net: hidden_dims [400, 400], so feature_extractor has Linear(1836,400), Linear(400,400), classifier Linear(400, 159).
    - loss_net: feature_network with hidden_dims [400, 400], output 400; energy_type 'seal' with global_hidden_dim 200.

    SEAL checkpoint keys: task_nn (feature_network.0/3, label_embeddings), score_nn (same + global_score_ff, global_projection).

    If ckpt is provided, ckpt_path is only used for logging; otherwise the checkpoint is loaded from ckpt_path.
    """
    if ckpt is None:
        if not ckpt_path:
            raise ValueError("Either ckpt_path or ckpt must be provided")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "task_nn" not in ckpt or "score_nn" not in ckpt:
        raise ValueError(
            f"Checkpoint does not contain 'task_nn' and 'score_nn' (SEAL format). "
            "Keys found: " + ", ".join(ckpt.keys())
        )

    task_sd = ckpt["task_nn"]
    score_sd = ckpt["score_nn"]

    # ---- Task net: SEAL task_nn -> CREL task_net ----
    # CREL: feature_extractor.0 (Linear), .1 (BN), .2 ReLU, .3 Dropout, .4 (Linear), .5 (BN), .6 ReLU, .7 Dropout; classifier.0 (Linear), .1 Sigmoid
    task_mapping = {
        "feature_network.0.weight": "feature_extractor.0.weight",
        "feature_network.0.bias": "feature_extractor.0.bias",
        "feature_network.3.weight": "feature_extractor.4.weight",
        "feature_network.3.bias": "feature_extractor.4.bias",
    }
    crel_task_sd = task_net.state_dict()
    for seal_k, crel_k in task_mapping.items():
        if seal_k in task_sd and crel_k in crel_task_sd:
            crel_task_sd[crel_k] = task_sd[seal_k].to(device)
    # label_embeddings (159, 400) -> classifier.0.weight (159, 400) [PyTorch Linear(400, 159)]
    if "label_embeddings.weight" in task_sd and "classifier.0.weight" in crel_task_sd:
        crel_task_sd["classifier.0.weight"] = task_sd["label_embeddings.weight"].to(device)
    if "classifier.0.bias" in crel_task_sd:
        crel_task_sd["classifier.0.bias"] = torch.zeros_like(crel_task_sd["classifier.0.bias"], device=device)
    missing_task, unexpected_task = task_net.load_state_dict(crel_task_sd, strict=False)
    logger.info(
        "Loaded SEAL task_nn into task_net. Missing: %s; Unexpected: %s",
        len(missing_task),
        len(unexpected_task),
    )
    if missing_task:
        logger.debug("Task net missing keys: %s", missing_task[:10])  # BN, etc.
    if unexpected_task:
        logger.debug("Task net unexpected keys: %s", unexpected_task[:5])

    # ---- Loss net: SEAL score_nn -> CREL loss_net (feature_network + energy) ----
    # Feature network: score_nn.feature_network.0/3 -> feature_network.net.0, .4; last layer net.8 (spectral_norm) from score_nn feature output 400
    loss_mapping = {
        "feature_network.0.weight": "feature_network.net.0.weight",
        "feature_network.0.bias": "feature_network.net.0.bias",
        "feature_network.3.weight": "feature_network.net.4.weight",
        "feature_network.3.bias": "feature_network.net.4.bias",
    }
    crel_loss_sd = loss_net.state_dict()
    for seal_k, crel_k in loss_mapping.items():
        if seal_k in score_sd and crel_k in crel_loss_sd:
            crel_loss_sd[crel_k] = score_sd[seal_k].to(device)
    # Last feature layer: score_nn has 400-dim features; CREL feature_network.net.8 is (400, output_dim). Spectral norm uses parametrizations.weight.original.
    _set_if_present(
        crel_loss_sd,
        "feature_network.net.8.parametrizations.weight.original",
        score_sd.get("feature_network.3.weight"),
        device,
    )
    if "feature_network.3.bias" in score_sd and "feature_network.net.8.bias" in crel_loss_sd:
        crel_loss_sd["feature_network.net.8.bias"] = score_sd["feature_network.3.bias"].to(device)
    # Energy (SEAL only): load score_nn energy into loss_net.energy when it is SEALEnergy.
    # When training CREL (energy_type 'crel'), only the feature_network above is loaded; CREL energy stays random.
    # SEALEnergy.local_weight is Linear(400, 159) so weight shape (159, 400); SEAL label_embeddings (159, 400).
    if getattr(loss_net, "energy_type", "") == "seal":
        label_emb = score_sd.get("label_embeddings.weight")
        if label_emb is not None:
            _set_if_present(
                crel_loss_sd,
                ("energy.local_weight.parametrizations.weight.original", "energy.local_weight.weight"),
                label_emb,
                device,
            )
        _set_if_present(
            crel_loss_sd,
            ("energy.global_linear.parametrizations.weight.original", "energy.global_linear.weight"),
            score_sd.get("global_score_ff.0.weight"),
            device,
        )
        if "global_score_ff.0.bias" in score_sd and "energy.global_linear.bias" in crel_loss_sd:
            crel_loss_sd["energy.global_linear.bias"] = score_sd["global_score_ff.0.bias"].to(device)
        _set_if_present(
            crel_loss_sd,
            ("energy.global_weight.parametrizations.weight.original", "energy.global_weight.weight"),
            score_sd.get("global_projection.weight"),
            device,
        )

    missing_loss, unexpected_loss = loss_net.load_state_dict(crel_loss_sd, strict=False)
    logger.info(
        "Loaded SEAL score_nn into loss_net. Missing: %s; Unexpected: %s",
        len(missing_loss),
        len(unexpected_loss),
    )
    if missing_loss:
        logger.debug("Loss net missing keys: %s", missing_loss[:10])
    if unexpected_loss:
        logger.debug("Loss net unexpected keys: %s", unexpected_loss[:5])
