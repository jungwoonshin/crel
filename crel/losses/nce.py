"""Contrastive losses for energy-based model training.

Provides NCE and InfoNCE ranking losses that train the energy network
to assign higher energy to ground-truth label configurations than to
negative samples from the task-net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCELoss(nn.Module):
    """NCE ranking loss for training the loss-net (energy network).

    The loss is:
        L_NCE = -log[ exp(s(x, y⁰)) / Σ_{k=0}^{K} exp(s(x, y^k)) ]

    where:
        s(x, y) = E(x, y) - Σ_i log P(y_i | x)
        y⁰ = ground truth
        y^k ~ task-net's independent distribution (negative samples)
    """

    def __init__(self, num_samples: int = 32):
        """
        Args:
            num_samples: K, number of negative samples.
        """
        super().__init__()
        self.num_samples = num_samples

    def forward(
        self,
        energy_gt: torch.Tensor,
        energy_neg: torch.Tensor,
        log_prob_gt: torch.Tensor,
        log_prob_neg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NCE loss.

        Args:
            energy_gt: energy of ground truth, shape (batch,).
            energy_neg: energy of negative samples, shape (batch, K).
            log_prob_gt: log probability of ground truth under task-net,
                shape (batch,). Computed as Σ_i log P(y_i^gt | x).
            log_prob_neg: log probability of negative samples,
                shape (batch, K).

        Returns:
            loss: scalar, mean NCE loss over batch.
        """
        # Score = energy - log_prob (following SEAL convention)
        score_gt = energy_gt - log_prob_gt  # (batch,)
        score_neg = energy_neg - log_prob_neg  # (batch, K)

        # Concatenate: [gt_score, neg_scores] along dim 1
        # gt_score is index 0
        scores = torch.cat([score_gt.unsqueeze(1), score_neg], dim=1)  # (batch, K+1)

        # NCE loss: -log softmax at index 0
        log_probs = F.log_softmax(scores, dim=1)
        loss = -log_probs[:, 0].mean()

        return loss


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for training the energy network.

    Unlike NCE, InfoNCE uses raw energy as scores without subtracting
    the proposal log-probability. This makes the loss simpler and avoids
    the adversarial scaling dynamic where the log-prob correction can
    amplify instability.

    The loss is:
        L = -log[ exp(E(x, y⁰)) / Σ_{k=0}^{K} exp(E(x, y^k)) ]

    where y⁰ is the ground truth and y^k are negative samples.
    """

    def __init__(self, num_samples: int = 32, temperature: float = 1.0):
        """
        Args:
            num_samples: K, number of negative samples.
            temperature: temperature scaling for the softmax.
        """
        super().__init__()
        self.num_samples = num_samples
        self.temperature = temperature

    def forward(
        self,
        energy_gt: torch.Tensor,
        energy_neg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            energy_gt: energy of ground truth, shape (batch,).
            energy_neg: energy of negative samples, shape (batch, K).

        Returns:
            loss: scalar, mean InfoNCE loss over batch.
        """
        # Scores are raw energies (no log-prob correction). Use float32 and
        # subtract max before log_softmax to avoid overflow/NaN when energies
        # are large (e.g. delicious with 983 labels) or under AMP (float16).
        scores = torch.cat(
            [energy_gt.unsqueeze(1).float(), energy_neg.float()], dim=1
        )
        scores = scores / self.temperature
        scores = scores - scores.max(dim=1, keepdim=True).values

        log_probs = F.log_softmax(scores, dim=1)
        loss = -log_probs[:, 0].mean()

        return loss


def compute_log_prob(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute log probability of labels under independent Bernoulli model.

    Args:
        labels: binary labels, shape (..., L).
        predictions: predicted probabilities, shape (..., L), in [0, 1].
        eps: small constant for numerical stability.

    Returns:
        log_prob: sum of per-label log probabilities, shape (...).
            i.e., Σ_i [y_i log p_i + (1 - y_i) log(1 - p_i)]
    """
    predictions = predictions.clamp(eps, 1 - eps)
    log_prob = (
        labels * predictions.log() + (1 - labels) * (1 - predictions).log()
    )
    return log_prob.sum(dim=-1)


def compute_log_prob_gaussian(
    samples: torch.Tensor,
    means: torch.Tensor,
    sigma: float = 0.3,
) -> torch.Tensor:
    """Compute log probability of samples under independent Gaussian model.

    Returns the unnormalized log-density (the constant -½L·log(2πσ²) is
    omitted because it cancels in the NCE softmax).

    Args:
        samples: continuous values, shape (..., L).
        means: predicted means (task-net outputs), shape (..., L).
        sigma: standard deviation of the Gaussian proposal.

    Returns:
        log_prob: unnormalized sum of per-label log densities, shape (...).
            i.e., Σ_i [-½(y_i - μ_i)² / σ²]
    """
    return -0.5 * ((samples - means) ** 2).sum(dim=-1) / (sigma ** 2)
