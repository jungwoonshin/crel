"""EMA centering for CREL energy.

Maintains an exponential moving average of the task-net's predictions
to provide stable centering targets μ(x) for the CREL energy.
"""

import torch
import torch.nn as nn


class EMACenter(nn.Module):
    """Exponential Moving Average tracker for label marginals.

    Tracks per-sample marginals using an EMA of the task-net's predictions.
    During early training, uses a faster beta (ema_warmup_beta) to quickly
    adapt, then anneals to the slower beta for stability.

    The EMA is stored as a running buffer indexed by sample ID. For datasets
    that don't fit per-sample tracking, falls back to batch-level statistics.
    """

    def __init__(
        self,
        num_labels: int,
        dataset_size: int,
        beta: float = 0.99,
        warmup_beta: float = 0.9,
        warmup_steps: int = 1000,
        per_sample: bool = True,
    ):
        """
        Args:
            num_labels: L, number of labels.
            dataset_size: N, number of training samples. Used for per-sample tracking.
            beta: target EMA coefficient (used after warmup).
            warmup_beta: initial EMA coefficient (faster tracking during warmup).
            warmup_steps: number of steps to anneal from warmup_beta to beta.
            per_sample: if True, track per-sample marginals. If False, use
                global marginals (single vector for all samples).
        """
        super().__init__()
        self.num_labels = num_labels
        self.dataset_size = dataset_size
        self.beta = beta
        self.warmup_beta = warmup_beta
        self.warmup_steps = warmup_steps
        self.per_sample = per_sample

        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

        if per_sample:
            # Per-sample marginal estimates: shape (N, L)
            # Initialize to 0.5 (uninformative prior)
            self.register_buffer(
                "marginals", torch.full((dataset_size, num_labels), 0.5)
            )
        else:
            # Global marginal estimate: shape (L,)
            self.register_buffer("marginals", torch.full((num_labels,), 0.5))

        # Track whether marginals have been initialized from real data
        self.register_buffer("initialized", torch.tensor(False))

    def current_beta(self) -> float:
        """Compute the current EMA coefficient with warmup annealing."""
        step = self.step_count.item()
        if step >= self.warmup_steps:
            return self.beta
        # Linear annealing from warmup_beta to beta
        t = step / max(self.warmup_steps, 1)
        return self.warmup_beta + t * (self.beta - self.warmup_beta)

    @torch.no_grad()
    def update(
        self,
        predictions: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Update EMA with new predictions from the task-net.

        Args:
            predictions: task-net outputs, shape (batch, L), in [0, 1].
            indices: sample indices in the dataset, shape (batch,).
                Required if per_sample=True.
        """
        beta = self.current_beta()

        if self.per_sample:
            if indices is None:
                raise ValueError("indices required for per-sample EMA tracking")

            if not self.initialized:
                # First update: initialize directly instead of EMA
                self.marginals[indices] = predictions
                self.initialized.fill_(True)
            else:
                self.marginals[indices] = (
                    beta * self.marginals[indices] + (1 - beta) * predictions
                )
        else:
            # Global: EMA of batch means
            batch_mean = predictions.mean(dim=0)
            if not self.initialized:
                self.marginals.copy_(batch_mean)
                self.initialized.fill_(True)
            else:
                self.marginals.mul_(beta).add_(batch_mean, alpha=1 - beta)

        self.step_count += 1

    def get_marginals(
        self,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get current marginal estimates.

        Args:
            indices: sample indices, shape (batch,). Required if per_sample=True.

        Returns:
            marginals: shape (batch, L) if per_sample and indices given,
                shape (L,) if global.
        """
        if self.per_sample:
            if indices is None:
                raise ValueError("indices required for per-sample EMA")
            return self.marginals[indices]
        return self.marginals

    @torch.no_grad()
    def initialize_from_labels(self, labels: torch.Tensor) -> None:
        """Initialize global marginals from training label statistics.

        Args:
            labels: full training label matrix, shape (N, L).
        """
        global_means = labels.float().mean(dim=0)
        if self.per_sample:
            self.marginals.fill_(0)
            self.marginals.add_(global_means.unsqueeze(0))
        else:
            self.marginals.copy_(global_means)
        self.initialized.fill_(True)
