"""Label covariance analysis utilities for CREL.

Implements the pre-experiment analysis described in Section 6.2 of the CREL
research plan.  The main purpose is to characterize the label covariance
structure of a multi-label dataset and recommend an effective rank *r* for
the low-rank precision factorisation used by the CREL energy function.

Typical usage:
    >>> import numpy as np
    >>> from crel.utils.covariance import label_covariance_analysis
    >>> labels = np.random.randint(0, 2, size=(1000, 50)).astype(float)
    >>> info = label_covariance_analysis(labels, variance_threshold=0.9)
    >>> print(info["effective_rank"], info["suggested_rank"])
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label covariance analysis
# ---------------------------------------------------------------------------


def label_covariance_analysis(
    labels: np.ndarray,
    variance_threshold: float = 0.9,
) -> Dict[str, object]:
    """Analyse the label covariance structure to determine effective rank.

    Given the binary label matrix **Y** of shape ``(N, L)``, this function
    computes the empirical covariance, its eigenspectrum, and derives an
    *effective rank* -- the minimum number of principal components needed to
    capture at least ``variance_threshold`` of the total variance.

    Parameters
    ----------
    labels : np.ndarray, shape (N, L)
        Binary (or continuous) label matrix from the training set.
    variance_threshold : float
        Fraction of total variance to capture when computing the effective
        rank.  Must be in ``(0, 1]``.  Default is ``0.9``.

    Returns
    -------
    dict
        ``'covariance'``
            ``(L, L)`` empirical covariance matrix.
        ``'eigenvalues'``
            ``(L,)`` eigenvalues sorted in descending order.
        ``'effective_rank'``
            Minimum *r* such that the top-*r* eigenvalues capture at least
            ``variance_threshold`` of the total variance.
        ``'suggested_rank'``
            Practical rank recommendation:
            ``min(effective_rank, L // 4, 64)``.
        ``'variance_explained'``
            ``(L,)`` cumulative proportion of variance explained.

    Raises
    ------
    ValueError
        If *labels* is not 2-D or *variance_threshold* is out of range.
    """
    # ---- Input validation --------------------------------------------------
    if labels.ndim != 2:
        raise ValueError(
            f"labels must be a 2-D array of shape (N, L), got shape {labels.shape}"
        )
    if not (0.0 < variance_threshold <= 1.0):
        raise ValueError(
            f"variance_threshold must be in (0, 1], got {variance_threshold}"
        )

    labels = np.asarray(labels, dtype=np.float64)
    N, L = labels.shape

    if N == 0:
        raise ValueError("labels array has zero samples (N=0)")

    # ---- Compute empirical covariance  Sigma = Y^T Y / N  -  mu mu^T ------
    mu = labels.mean(axis=0)  # (L,)
    # Using the unbiased-style formula centred on the sample mean:
    #   Sigma_ij = (1/N) sum_n Y_ni Y_nj  -  mu_i mu_j
    # which is equivalent to np.cov with bias=True (population covariance).
    cov = (labels.T @ labels) / N - np.outer(mu, mu)  # (L, L)

    # ---- Eigenvalue decomposition ------------------------------------------
    # eigvalsh is faster than eigh when eigenvectors are not needed, and it
    # guarantees real eigenvalues for symmetric matrices.
    eigenvalues = np.linalg.eigvalsh(cov)  # ascending order

    # Sort descending.
    eigenvalues = eigenvalues[::-1].copy()

    # Clamp tiny negative eigenvalues that arise from floating-point error.
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # ---- Cumulative variance explained -------------------------------------
    total_var = eigenvalues.sum()
    if total_var < 1e-15:
        # Degenerate case: all labels are constant.
        warnings.warn(
            "Total label variance is near zero; all labels may be constant.",
            stacklevel=2,
        )
        cumulative = np.ones(L, dtype=np.float64)
        effective_rank = 1
    else:
        cumulative = np.cumsum(eigenvalues) / total_var
        # effective_rank: smallest r such that cumulative[r-1] >= threshold.
        indices = np.where(cumulative >= variance_threshold)[0]
        if len(indices) > 0:
            effective_rank = int(indices[0]) + 1
        else:
            effective_rank = L

    # ---- Suggested practical rank ------------------------------------------
    suggested_rank = min(effective_rank, L // 4, 64)
    # Ensure at least 1.
    suggested_rank = max(suggested_rank, 1)

    logger.info(
        "Label covariance analysis  (N=%d, L=%d): "
        "effective_rank=%d (%.0f%% variance), suggested_rank=%d",
        N,
        L,
        effective_rank,
        variance_threshold * 100,
        suggested_rank,
    )

    return {
        "covariance": cov,
        "eigenvalues": eigenvalues,
        "effective_rank": effective_rank,
        "suggested_rank": suggested_rank,
        "variance_explained": cumulative,
    }


# ---------------------------------------------------------------------------
# Precision matrix (inverse covariance)
# ---------------------------------------------------------------------------


def compute_precision_matrix(
    labels: np.ndarray,
    regularization: float = 1e-4,
) -> np.ndarray:
    """Compute the regularised empirical precision matrix (inverse covariance).

    This is intended for **diagnostics only** -- e.g. comparing the learned
    low-rank precision with the empirical one (Diagnostic 2 in the CREL plan).

    The precision matrix is defined as::

        Sigma_reg  = Sigma + regularization * I
        Precision  = Sigma_reg^{-1}

    where Sigma is the population covariance of the label matrix.

    .. warning::
        For large label spaces (*L* > 1000), the inversion is O(L^3) and may
        be slow or memory-intensive.  Use this function for analysis and
        diagnostics, not in a training loop.

    Parameters
    ----------
    labels : np.ndarray, shape (N, L)
        Binary (or continuous) label matrix.
    regularization : float
        Ridge regularisation added to the diagonal for numerical stability.
        Must be non-negative.

    Returns
    -------
    precision : np.ndarray, shape (L, L)
        The precision matrix Sigma^{-1}.

    Raises
    ------
    ValueError
        If *labels* is not 2-D or *regularization* is negative.
    numpy.linalg.LinAlgError
        If the regularised covariance matrix is still singular (very unlikely
        with positive regularisation).
    """
    # ---- Input validation --------------------------------------------------
    if labels.ndim != 2:
        raise ValueError(
            f"labels must be a 2-D array of shape (N, L), got shape {labels.shape}"
        )
    if regularization < 0:
        raise ValueError(
            f"regularization must be non-negative, got {regularization}"
        )

    labels = np.asarray(labels, dtype=np.float64)
    N, L = labels.shape

    if N == 0:
        raise ValueError("labels array has zero samples (N=0)")

    if L > 1000:
        warnings.warn(
            f"Computing a {L}x{L} precision matrix is O(L^3) and may be "
            f"slow / memory-intensive.  Consider using this function only "
            f"for diagnostics, not during training.",
            stacklevel=2,
        )

    # ---- Covariance --------------------------------------------------------
    mu = labels.mean(axis=0)
    cov = (labels.T @ labels) / N - np.outer(mu, mu)

    # ---- Regularise and invert ---------------------------------------------
    cov_reg = cov + regularization * np.eye(L, dtype=np.float64)
    precision = np.linalg.inv(cov_reg)

    logger.info(
        "Precision matrix computed (L=%d, reg=%.2e, cond=%.2e)",
        L,
        regularization,
        np.linalg.cond(cov_reg),
    )

    return precision
