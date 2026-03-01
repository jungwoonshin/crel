"""Sampling strategies for NCE training.

Provides negative samples for the NCE ranking loss used to train
the CREL (or SEAL) energy network.
"""

import torch


def sample_bernoulli(
    predictions: torch.Tensor,
    num_samples: int = 32,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample from independent per-label Bernoulli distributions.

    Each label is sampled independently using the task-net's predicted
    probability as the Bernoulli parameter. This is cheap but produces
    samples without label correlations.

    Args:
        predictions: task-net outputs, shape (batch, L), in [0, 1].
        num_samples: K, number of negative samples per input.
        labels: ignored (accepted for API compatibility).

    Returns:
        samples: shape (batch, K, L), binary samples.
    """
    batch_size, num_labels = predictions.shape
    # Expand predictions for K samples: (batch, 1, L) -> (batch, K, L)
    probs = predictions.unsqueeze(1).expand(batch_size, num_samples, num_labels)
    samples = torch.bernoulli(probs)
    return samples


def sample_gaussian_noise(
    predictions: torch.Tensor,
    num_samples: int = 32,
    labels: torch.Tensor | None = None,
    sigma: float = 0.3,
) -> torch.Tensor:
    """Sample by adding Gaussian noise to task-net predictions.

    Args:
        predictions: task-net outputs, shape (batch, L), in [0, 1].
        num_samples: K, number of negative samples.
        labels: ignored (accepted for API compatibility).
        sigma: noise standard deviation.

    Returns:
        samples: shape (batch, K, L), in [0, 1].
    """
    batch_size, num_labels = predictions.shape
    probs = predictions.unsqueeze(1).expand(batch_size, num_samples, num_labels)
    noise = torch.randn_like(probs) * sigma
    samples = (probs + noise).clamp(0, 1)
    return samples


def sample_corruption(
    predictions: torch.Tensor,
    num_samples: int = 32,
    labels: torch.Tensor | None = None,
    flip_fraction: float = 0.1,
) -> torch.Tensor:
    """Sample by corrupting ground-truth labels.

    Randomly flips a fraction of GT labels per sample. This preserves
    most of the real correlation structure while creating "near-miss"
    configurations that differ from GT in a small number of labels.

    Unlike Bernoulli sampling (which destroys all correlations by sampling
    independently), corruption-based negatives retain enough structure for
    the energy to learn meaningful discrimination.

    Args:
        predictions: task-net outputs, shape (batch, L), in [0, 1].
            Used as fallback if labels is None.
        num_samples: K, number of negative samples per input.
        labels: ground-truth labels, shape (batch, L), binary.
        flip_fraction: fraction of labels to flip per sample.

    Returns:
        samples: shape (batch, K, L), binary samples.
    """
    if labels is None:
        # Fallback to Bernoulli if no GT available (e.g., at inference)
        return sample_bernoulli(predictions, num_samples)

    batch_size, num_labels = labels.shape
    # Expand GT for K samples: (batch, 1, L) -> (batch, K, L)
    gt_expanded = labels.unsqueeze(1).expand(batch_size, num_samples, num_labels)

    # Random flip mask: each label has flip_fraction probability of flipping
    flip_mask = torch.rand_like(gt_expanded.float()) < flip_fraction

    # Flip: 0 -> 1, 1 -> 0 where mask is True
    samples = gt_expanded.clone().float()
    samples[flip_mask] = 1.0 - samples[flip_mask]

    return samples


def get_sampler(method: str = "bernoulli"):
    """Get sampling function by name.

    Args:
        method: 'bernoulli', 'gaussian_noise', or 'corruption'.

    Returns:
        Sampling function with signature (predictions, num_samples, labels) -> samples.
    """
    samplers = {
        "bernoulli": sample_bernoulli,
        "gaussian_noise": sample_gaussian_noise,
        "corruption": sample_corruption,
    }
    if method not in samplers:
        raise ValueError(f"Unknown sampling method: {method}. Choose from {list(samplers)}")
    return samplers[method]
