"""Sampling strategies for NCE training.

Provides negative samples for the NCE ranking loss used to train
the CREL (or SEAL) energy network.
"""

import torch


def sample_bernoulli(
    predictions: torch.Tensor,
    num_samples: int = 32,
) -> torch.Tensor:
    """Sample from independent per-label Bernoulli distributions.

    Each label is sampled independently using the task-net's predicted
    probability as the Bernoulli parameter. This is cheap but produces
    samples without label correlations.

    Args:
        predictions: task-net outputs, shape (batch, L), in [0, 1].
        num_samples: K, number of negative samples per input.

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
    sigma: float = 0.3,
) -> torch.Tensor:
    """Sample by adding Gaussian noise to task-net predictions.

    Better than Bernoulli for structured outputs (e.g., image segmentation)
    where independent per-pixel sampling produces unrealistic masks.

    Args:
        predictions: task-net outputs, shape (batch, L), in [0, 1].
        num_samples: K, number of negative samples.
        sigma: noise standard deviation.

    Returns:
        samples: shape (batch, K, L), in [0, 1].
    """
    batch_size, num_labels = predictions.shape
    probs = predictions.unsqueeze(1).expand(batch_size, num_samples, num_labels)
    noise = torch.randn_like(probs) * sigma
    samples = (probs + noise).clamp(0, 1)
    return samples


def get_sampler(method: str = "bernoulli"):
    """Get sampling function by name.

    Args:
        method: 'bernoulli' or 'gaussian_noise'.

    Returns:
        Sampling function.
    """
    samplers = {
        "bernoulli": sample_bernoulli,
        "gaussian_noise": sample_gaussian_noise,
    }
    if method not in samplers:
        raise ValueError(f"Unknown sampling method: {method}. Choose from {list(samplers)}")
    return samplers[method]
