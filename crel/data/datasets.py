"""Multi-label classification dataset loading for CREL.

Supports benchmark datasets from the SEAL paper (Bai et al., 2021):
  - Mulan ARFF format: bibtex, delicious, genbase, cal500
  - Sparse/LibSVM format and pre-processed NumPy archives
  - Additional datasets: eurlex_ev, expr_fun, spo_fun

Typical usage:
    >>> from crel.data.datasets import load_dataset, create_data_loaders
    >>> ds = load_dataset("bibtex", data_dir="./data", split="train")
    >>> loaders = create_data_loaders("bibtex", data_dir="./data", batch_size=64)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASET_INFO: Dict[str, Dict] = {
    "bibtex": {
        "num_labels": 159,
        "input_dim": 1836,
        "description": "Bibtex (binary input)",
    },
    "delicious": {
        "num_labels": 983,
        "input_dim": 500,
        "description": "Delicious (binary input)",
    },
    "genbase": {
        "num_labels": 27,
        "input_dim": 1186,
        "description": "Genbase (binary input)",
    },
    "cal500": {
        "num_labels": 174,
        "input_dim": 68,
        "description": "Cal500 (continuous input)",
    },
    "eurlex_ev": {
        "num_labels": 3993,
        "input_dim": 5000,
        "description": "Eurlex-ev (continuous input)",
    },
    "expr_fun": {
        "num_labels": 540,
        "input_dim": 651,
        "description": "Expr_FUN (continuous, taxonomy)",
    },
    "spo_fun": {
        "num_labels": 455,
        "input_dim": 651,
        "description": "Spo_FUN (continuous, taxonomy)",
    },
}


# ---------------------------------------------------------------------------
# MultiLabelDataset
# ---------------------------------------------------------------------------


class MultiLabelDataset(Dataset):
    """PyTorch dataset for multi-label classification.

    Stores dense feature and label matrices and yields (feature, label)
    tuples as float32 tensors.

    Parameters
    ----------
    features : np.ndarray
        Input feature matrix of shape ``(N, D)``.
    labels : np.ndarray
        Binary label matrix of shape ``(N, L)``.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D (N, D), got shape {features.shape}"
            )
        if labels.ndim != 2:
            raise ValueError(
                f"labels must be 2-D (N, L), got shape {labels.shape}"
            )
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Number of samples in features ({features.shape[0]}) and "
                f"labels ({labels.shape[0]}) must match"
            )

        # Store as contiguous float32 arrays for efficient tensor conversion.
        self._features = np.ascontiguousarray(features, dtype=np.float32)
        self._labels = np.ascontiguousarray(labels, dtype=np.float32)

    # -- Sequence protocol ---------------------------------------------------

    def __len__(self) -> int:
        return self._features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        feat = torch.from_numpy(self._features[idx])
        lab = torch.from_numpy(self._labels[idx])
        return feat, lab, idx

    # -- Convenience properties ----------------------------------------------

    @property
    def num_labels(self) -> int:
        """Number of labels (L)."""
        return self._labels.shape[1]

    @property
    def input_dim(self) -> int:
        """Dimensionality of input features (D)."""
        return self._features.shape[1]


# ---------------------------------------------------------------------------
# ARFF parsing
# ---------------------------------------------------------------------------


def _parse_arff(
    filepath: str,
    num_labels_hint: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parse a Mulan-style ARFF file (dense or sparse) into feature and label arrays.

    Mulan ARFF files store multi-label data with labels occupying the *last*
    ``num_labels`` attributes.  The number of labels is typically encoded in a
    companion XML file, but here we infer it from the ``@attribute`` declarations
    that have values ``{0,1}`` (or ``{0, 1}``) at the end of the attribute list.

    Parameters
    ----------
    filepath : str
        Path to the ``.arff`` file.
    num_labels_hint : int, optional
        If provided, use this as the number of label attributes (e.g. for MEKA
        fold files like Bibtex-fold1.arff where the filename does not match
        DATASET_INFO).

    Returns
    -------
    features : np.ndarray, shape (N, D)
    labels : np.ndarray, shape (N, L)
    num_label_attrs : int
        The number of trailing attributes treated as labels.
    """
    attributes: list[str] = []
    is_sparse = False
    data_lines: list[str] = []
    in_data = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lower = line.lower()

            if lower.startswith("@attribute"):
                attributes.append(line)
            elif lower.startswith("@data"):
                in_data = True
                continue

            if in_data and line:
                if line.startswith("{"):
                    is_sparse = True
                data_lines.append(line)

    total_attrs = len(attributes)
    if total_attrs == 0:
        raise ValueError(f"No @attribute declarations found in {filepath}")

    # Identify trailing binary label attributes.
    # Labels are typically declared as  @attribute <name> {0,1}  or  {0, 1}.
    label_pattern = re.compile(r"\{\s*0\s*,\s*1\s*\}")
    num_label_attrs = 0
    for attr in reversed(attributes):
        if label_pattern.search(attr):
            num_label_attrs += 1
        else:
            break

    # Try to find dataset name from filepath for DATASET_INFO lookup.
    stem = Path(filepath).stem  # e.g. "bibtex_train" or "train"
    # Try both "bibtex_train" -> "bibtex" and parent dir name
    dataset_name = stem.rsplit("_", 1)[0]
    if dataset_name not in DATASET_INFO:
        dataset_name = Path(filepath).parent.name

    if num_label_attrs == 0 or num_label_attrs == total_attrs:
        # Fallback: heuristic failed (either no binary attrs, or ALL are binary
        # like in Bibtex where features and labels are both {0,1}).
        if num_labels_hint is not None:
            num_label_attrs = num_labels_hint
            logger.info(
                "Using num_labels_hint=%d for %s",
                num_label_attrs, filepath,
            )
        elif dataset_name in DATASET_INFO:
            num_label_attrs = DATASET_INFO[dataset_name]["num_labels"]
            logger.info(
                "Using DATASET_INFO to set num_labels=%d for %s",
                num_label_attrs, filepath,
            )
        elif num_label_attrs == 0:
            raise ValueError(
                f"Cannot determine label count from ARFF file: {filepath}"
            )

    num_feature_attrs = total_attrs - num_label_attrs

    # Parse data section.
    N = len(data_lines)
    features = np.zeros((N, num_feature_attrs), dtype=np.float64)
    labels = np.zeros((N, num_label_attrs), dtype=np.float64)

    if is_sparse:
        for i, line in enumerate(data_lines):
            # Sparse ARFF: { idx val, idx val, ... }
            content = line.strip().strip("{}")
            if not content:
                continue
            for token in content.split(","):
                token = token.strip()
                if not token:
                    continue
                parts = token.split()
                if len(parts) != 2:
                    continue
                col_idx = int(parts[0])
                value = float(parts[1])
                if col_idx < num_feature_attrs:
                    features[i, col_idx] = value
                else:
                    labels[i, col_idx - num_feature_attrs] = value
    else:
        for i, line in enumerate(data_lines):
            vals = line.split(",")
            for j, v in enumerate(vals):
                v = v.strip()
                if not v:
                    continue
                fval = float(v)
                if j < num_feature_attrs:
                    features[i, j] = fval
                else:
                    labels[i, j - num_feature_attrs] = fval

    return features, labels, num_label_attrs


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def load_dataset(
    name: str,
    data_dir: str = "./data",
    split: str = "train",
) -> MultiLabelDataset:
    """Load a multi-label classification dataset.

    The function searches for data files in the following order:

    1. **Pre-processed NumPy archive** (fastest):
       ``{data_dir}/{name}/{split}.npz`` with keys ``'features'`` and
       ``'labels'``.
    2. **ARFF format** (Mulan datasets):
       ``{data_dir}/{name}/{name}_{split}.arff`` or
       ``{data_dir}/{name}/{split}.arff``.
    3. If nothing is found, an informative ``FileNotFoundError`` is raised
       with download instructions.

    Parameters
    ----------
    name : str
        Dataset identifier.  One of ``bibtex``, ``delicious``, ``genbase``,
        ``cal500``, ``eurlex_ev``, ``expr_fun``, ``spo_fun``.
    data_dir : str
        Root data directory.  Each dataset is expected in a subdirectory
        ``{data_dir}/{name}/``.
    split : str
        One of ``'train'``, ``'val'``, or ``'test'``.

    Returns
    -------
    MultiLabelDataset

    Raises
    ------
    FileNotFoundError
        If no data file can be located.
    ValueError
        If the split name is invalid or parsing fails.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

    base_dir = Path(data_dir) / name

    # ------------------------------------------------------------------
    # Strategy 1: pre-processed .npz
    # ------------------------------------------------------------------
    npz_path = base_dir / f"{split}.npz"
    if npz_path.is_file():
        logger.info("Loading pre-processed NPZ: %s", npz_path)
        data = np.load(str(npz_path), allow_pickle=False)
        features = data["features"]
        labels = data["labels"]

        # Handle sparse matrices stored inside npz (CSR components).
        if features.ndim == 0:
            # scipy sparse was saved; reload via scipy.
            loaded = sp.load_npz(str(npz_path))
            features = loaded.toarray()

        if sp.issparse(features):
            features = features.toarray()  # type: ignore[union-attr]
        if sp.issparse(labels):
            labels = labels.toarray()  # type: ignore[union-attr]

        return MultiLabelDataset(features, labels)

    # Also try scipy-style sparse npz (saved with scipy.sparse.save_npz).
    sparse_npz_features = base_dir / f"{split}_features.npz"
    sparse_npz_labels = base_dir / f"{split}_labels.npz"
    if sparse_npz_features.is_file() and sparse_npz_labels.is_file():
        logger.info(
            "Loading sparse NPZ: %s, %s", sparse_npz_features, sparse_npz_labels
        )
        features = sp.load_npz(str(sparse_npz_features)).toarray()
        labels = sp.load_npz(str(sparse_npz_labels)).toarray()
        return MultiLabelDataset(features, labels)

    # ------------------------------------------------------------------
    # Strategy 2: ARFF
    # ------------------------------------------------------------------
    arff_candidates = [
        base_dir / f"{name}_{split}.arff",
        base_dir / f"{split}.arff",
        base_dir / f"{name}-{split}.arff",
    ]
    for arff_path in arff_candidates:
        if arff_path.is_file():
            logger.info("Loading ARFF: %s", arff_path)
            features, labels, _ = _parse_arff(str(arff_path))
            return MultiLabelDataset(features, labels)

    # ------------------------------------------------------------------
    # Nothing found — provide a helpful error message.
    # ------------------------------------------------------------------
    tried_paths = [str(npz_path)] + [str(p) for p in arff_candidates]
    tried_str = "\n  ".join(tried_paths)

    info_msg = ""
    if name in DATASET_INFO:
        info_msg = (
            f"\nDataset '{name}': {DATASET_INFO[name]['description']}\n"
            f"  Expected labels: {DATASET_INFO[name]['num_labels']}, "
            f"features: {DATASET_INFO[name]['input_dim']}\n"
        )

    raise FileNotFoundError(
        f"Could not find data files for dataset '{name}', split '{split}'.\n"
        f"Searched the following paths:\n  {tried_str}\n"
        f"{info_msg}"
        f"\nTo prepare the data, place files in one of these formats:\n"
        f"  1. Pre-processed:  {base_dir}/{split}.npz  (keys: 'features', 'labels')\n"
        f"  2. ARFF (Mulan):   {base_dir}/{name}_{split}.arff\n"
        f"\nMulan datasets can be downloaded from:\n"
        f"  http://mulan.sourceforge.net/datasets-mlc.html\n"
        f"Pre-processed splits used in the SEAL paper are available at:\n"
        f"  https://github.com/Thartvigsen/SEAL"
    )


# ---------------------------------------------------------------------------
# MEKA / SEAL fold-based split (e.g. bibtex 10-fold stratified)
# ---------------------------------------------------------------------------

# SEAL bibtex split: train = folds 1-6, val = 7-8, test = 9-10
BIBTEX_SEAL_TRAIN_FOLDS = [1, 2, 3, 4, 5, 6]
BIBTEX_SEAL_VAL_FOLDS = [7, 8]
BIBTEX_SEAL_TEST_FOLDS = [9, 10]


def _load_meka_folds(
    folds_dir: Path,
    fold_indices: list[int],
    num_labels: int,
) -> MultiLabelDataset:
    """Load MEKA-style fold ARFFs (e.g. Bibtex-fold1.arff) and return one dataset."""
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for fold in fold_indices:
        path = folds_dir / f"Bibtex-fold{fold}.arff"
        if not path.is_file():
            raise FileNotFoundError(
                f"MEKA fold file not found: {path}. "
                "Use SEAL's data layout (e.g. data/bibtex_stratified10folds_meka/)."
            )
        features, labels, _ = _parse_arff(str(path), num_labels_hint=num_labels)
        all_features.append(features)
        all_labels.append(labels)
    features = np.concatenate(all_features, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.float32)
    return MultiLabelDataset(features, labels)


def create_data_loaders(
    name: str,
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    bibtex_folds_dir: Optional[str] = None,
) -> Dict[str, Optional[DataLoader]]:
    """Create train / val / test :class:`DataLoader` instances for a dataset.

    Parameters
    ----------
    name : str
        Dataset name (see :data:`DATASET_INFO`).
    data_dir : str
        Root data directory.
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of workers for parallel data loading.
    bibtex_folds_dir : str, optional
        If set and name is ``'bibtex'``, load SEAL's split from MEKA folds:
        train = folds 1-6, val = 7-8, test = 9-10 (Bibtex-fold1.arff ... in this dir).

    Returns
    -------
    dict[str, DataLoader | None]
        Dictionary with keys ``'train'``, ``'val'``, ``'test'``.
        ``'val'`` may be ``None`` if no validation split is available.
    """
    loaders: Dict[str, Optional[DataLoader]] = {}

    if name == "bibtex" and bibtex_folds_dir is not None:
        folds_path = Path(bibtex_folds_dir)
        info = DATASET_INFO["bibtex"]
        num_labels = info["num_labels"]
        logger.info("Loading bibtex from SEAL MEKA folds: %s", folds_path)
        train_ds = _load_meka_folds(
            folds_path, BIBTEX_SEAL_TRAIN_FOLDS, num_labels
        )
        val_ds = _load_meka_folds(
            folds_path, BIBTEX_SEAL_VAL_FOLDS, num_labels
        )
        test_ds = _load_meka_folds(
            folds_path, BIBTEX_SEAL_TEST_FOLDS, num_labels
        )
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        logger.info(
            "Train: %d samples, %d features, %d labels",
            len(train_ds),
            train_ds.input_dim,
            train_ds.num_labels,
        )
        logger.info("Val:   %d samples", len(val_ds))
        logger.info("Test:  %d samples", len(test_ds))
        return loaders

    # Train split (required).
    train_ds = load_dataset(name, data_dir=data_dir, split="train")
    loaders["train"] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    logger.info(
        "Train: %d samples, %d features, %d labels",
        len(train_ds),
        train_ds.input_dim,
        train_ds.num_labels,
    )

    # Validation split (optional).
    try:
        val_ds = load_dataset(name, data_dir=data_dir, split="val")
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        logger.info("Val:   %d samples", len(val_ds))
    except FileNotFoundError:
        loaders["val"] = None
        logger.info("Val:   not available (no val split found)")

    # Test split (required).
    test_ds = load_dataset(name, data_dir=data_dir, split="test")
    loaders["test"] = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    logger.info("Test:  %d samples", len(test_ds))

    return loaders
