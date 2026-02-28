"""Analyze label covariance structure for a dataset.

Usage:
    python scripts/analyze_labels.py --dataset bibtex --data-dir ./data
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crel.data.datasets import load_dataset, DATASET_INFO
from crel.utils.covariance import label_covariance_analysis, compute_precision_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Analyze label covariance structure")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_INFO))
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Variance threshold for effective rank")
    parser.add_argument("--compute-precision", action="store_true",
                        help="Also compute precision matrix (slow for large L)")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, data_dir=args.data_dir, split="train")
    labels = ds._labels

    print(f"\nDataset: {args.dataset}")
    print(f"  Samples: {labels.shape[0]}")
    print(f"  Labels:  {labels.shape[1]}")
    print(f"  Mean labels per sample: {labels.sum(axis=1).mean():.1f}")
    print(f"  Label density: {labels.mean():.4f}")

    analysis = label_covariance_analysis(labels, variance_threshold=args.threshold)

    print(f"\nCovariance Analysis:")
    print(f"  Effective rank (90% variance): {analysis['effective_rank']}")
    print(f"  Suggested CREL rank: {analysis['suggested_rank']}")

    evals = analysis["eigenvalues"]
    print(f"  Top 5 eigenvalues: {evals[:5]}")
    print(f"  Eigenvalue ratio (1st/10th): {evals[0]/max(evals[9], 1e-10):.1f}")

    cum_var = analysis["variance_explained"]
    for frac in [0.5, 0.8, 0.9, 0.95, 0.99]:
        idx = np.searchsorted(cum_var, frac)
        print(f"  Rank for {frac*100:.0f}% variance: {idx + 1}")

    if args.compute_precision:
        print(f"\nComputing precision matrix (L={labels.shape[1]})...")
        precision = compute_precision_matrix(labels)
        np.fill_diagonal(precision, 0)
        print(f"  Off-diagonal Frobenius norm: {np.linalg.norm(precision, 'fro'):.4f}")
        print(f"  Max off-diagonal: {np.abs(precision).max():.4f}")
        print(f"  Sparsity (|val|<0.01): {(np.abs(precision) < 0.01).mean():.2%}")


if __name__ == "__main__":
    main()
