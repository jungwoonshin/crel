"""Optuna hyperparameter tuning for CREL.

Usage:
    python scripts/optuna_tune.py --dataset expr_fun --n-trials 10
    python scripts/optuna_tune.py --dataset bibtex --n-trials 20 --epochs 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import optuna
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crel.data.datasets import create_data_loaders
from crel.models.task_net import TaskNet
from crel.models.loss_net import LossNet
from crel.training.trainer import CRELTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def objective(trial: optuna.Trial, args: argparse.Namespace, loaders: dict) -> float:
    """Single Optuna trial: build models, train, return best val sample_f1."""

    train_ds = loaders["train"].dataset
    input_dim = train_ds.input_dim
    num_labels = train_ds.num_labels

    # --- Search space ---
    task_net_lr = trial.suggest_float("task_net_lr", 1e-4, 1e-2, log=True)
    loss_net_lr = trial.suggest_float("loss_net_lr", 1e-4, 1e-2, log=True)
    task_net_hidden_dim = trial.suggest_categorical("task_net_hidden_dim", [256, 512, 768])
    task_net_n_layers = trial.suggest_int("task_net_n_layers", 1, 3)
    feature_net_hidden_dim = trial.suggest_categorical("feature_net_hidden_dim", [256, 512, 768])
    energy_rank = trial.suggest_categorical("energy_rank", [32, 64, 128])
    energy_label_embed_dim = trial.suggest_categorical("energy_label_embed_dim", [32, 64, 128])
    energy_proj_hidden_dim = trial.suggest_categorical("energy_proj_hidden_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Build hidden dims lists
    task_hidden_dims = [task_net_hidden_dim] * task_net_n_layers
    feature_hidden_dims = [feature_net_hidden_dim, feature_net_hidden_dim]

    # Build task-net
    task_net = TaskNet(
        input_dim=input_dim,
        num_labels=num_labels,
        hidden_dims=task_hidden_dims,
        dropout=dropout,
    )

    # Build loss-net
    loss_net = LossNet(
        input_dim=input_dim,
        num_labels=num_labels,
        energy_type="crel",
        feature_hidden_dims=feature_hidden_dims,
        feature_dropout=dropout,
        rank=energy_rank,
        label_embed_dim=energy_label_embed_dim,
        proj_hidden_dim=energy_proj_hidden_dim,
    )

    logger.info(
        "Trial %d: task_hidden=%s, feat_hidden=%s, rank=%d, lr=(%.1e, %.1e), dropout=%.2f",
        trial.number, task_hidden_dims, feature_hidden_dims,
        energy_rank, task_net_lr, loss_net_lr, dropout,
    )

    # Build trainer config (reduced epochs for tuning)
    epochs = args.epochs
    trainer_config = TrainerConfig(
        total_epochs=epochs,
        task_net_lr=task_net_lr,
        loss_net_lr=loss_net_lr,
        weight_decay=1e-5,
        lambda_bce=1.0,
        ema_beta=0.99,
        ema_warmup_beta=0.9,
        ema_warmup_fraction=0.05,
        contrastive_loss="infonce",
        nce_samples=32,
        nce_sampling="adversarial",
        infonce_temperature=1.0,
        grad_clip=5.0,
        log_interval=999999,  # suppress per-step diagnostics
        track_diagnostics=False,
        experiment_dir="",  # no file logging per trial
        device=args.device,
    )

    trainer = CRELTrainer(
        task_net=task_net,
        loss_net=loss_net,
        train_loader=loaders["train"],
        val_loader=loaders.get("val"),
        test_loader=loaders.get("test"),
        config=trainer_config,
    )

    trainer.train()

    # best_val_metric is tracked internally by CRELTrainer on val sample_f1
    best_val_f1 = trainer.best_val_metric
    logger.info("Trial %d finished: best val sample_f1 = %.4f", trial.number, best_val_f1)

    return best_val_f1


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for CREL")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default="expr_fun")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial (reduced for tuning)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Load config for dataset settings
    cfg = load_config(args.config)
    data_dir = cfg["dataset"]["data_dir"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataset"].get("num_workers", 4)

    logger.info("Loading dataset: %s", args.dataset)
    loaders = create_data_loaders(
        name=args.dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Create Optuna study with SQLite storage for crash recovery
    out_dir = Path(f"experiment_result/optuna_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_dir / 'study.db'}"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name=f"crel_{args.dataset}",
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, args, loaders),
        n_trials=args.n_trials,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"OPTUNA TUNING COMPLETE: {args.dataset}")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best val sample_f1: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"{'='*60}\n")

    # Save best params
    best_params_path = out_dir / "best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_value": study.best_value,
                "best_trial": study.best_trial.number,
                "best_params": study.best_params,
            },
            f,
            indent=2,
        )
    logger.info("Best params saved to %s", best_params_path)

    # Save all trials
    trials_path = out_dir / "all_trials.json"
    trials_data = []
    for t in study.trials:
        trials_data.append({
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })
    with open(trials_path, "w", encoding="utf-8") as f:
        json.dump(trials_data, f, indent=2)
    logger.info("All trials saved to %s", trials_path)


if __name__ == "__main__":
    main()
