"""Main training script for CREL.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --dataset bibtex --energy crel
    python scripts/train.py --config configs/default.yaml --energy seal  # Run SEAL baseline
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crel.data.datasets import create_data_loaders, DATASET_INFO
from crel.models.task_net import TaskNet
from crel.models.loss_net import LossNet
from crel.training.trainer import CRELTrainer, TrainerConfig
from crel.utils.covariance import label_covariance_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train CREL / SEAL models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--energy", type=str, default=None, choices=["crel", "seal"])
    parser.add_argument("--rank", type=int, default=None, help="Override CREL rank")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--analyze-labels", action="store_true",
                        help="Run label covariance analysis before training")
    parser.add_argument("--desc", type=str, default="",
                        help="Experiment description logged to result file")
    args = parser.parse_args()

    # Load and override config
    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset"]["name"] = args.dataset
    if args.energy:
        cfg["energy"]["type"] = args.energy
    if args.rank:
        cfg["energy"]["rank"] = args.rank
    if args.epochs:
        cfg["training"]["total_epochs"] = args.epochs
    if args.seed:
        cfg["seed"] = args.seed

    # Seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    dataset_name = cfg["dataset"]["name"]
    energy_type = cfg["energy"]["type"]
    logger.info("Dataset: %s | Energy: %s | Seed: %d", dataset_name, energy_type, seed)

    # Load data
    data_dir = cfg["dataset"]["data_dir"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["dataset"].get("num_workers", 4)

    loaders = create_data_loaders(
        name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    train_ds = loaders["train"].dataset
    input_dim = train_ds.input_dim
    num_labels = train_ds.num_labels

    # Optional: label covariance analysis
    if args.analyze_labels:
        import numpy as np
        labels_np = train_ds._labels
        analysis = label_covariance_analysis(labels_np)
        logger.info(
            "Effective rank: %d | Suggested rank: %d",
            analysis["effective_rank"],
            analysis["suggested_rank"],
        )
        if args.rank is None and energy_type == "crel":
            cfg["energy"]["rank"] = analysis["suggested_rank"]
            logger.info("Auto-setting rank to %d", analysis["suggested_rank"])

    # Build task-net
    task_net_cfg = cfg["task_net"]
    task_net = TaskNet(
        input_dim=input_dim,
        num_labels=num_labels,
        hidden_dims=task_net_cfg["hidden_dims"],
        dropout=task_net_cfg["dropout"],
        activation=task_net_cfg.get("activation", "relu"),
    )
    logger.info("TaskNet params: %d", sum(p.numel() for p in task_net.parameters()))

    # Build loss-net
    energy_cfg = cfg["energy"]
    feature_cfg = cfg.get("feature_net", {})

    energy_kwargs = {}
    if energy_type == "crel":
        energy_kwargs = {
            "rank": energy_cfg.get("rank", 32),
            "label_embed_dim": energy_cfg.get("label_embed_dim", 32),
            "proj_hidden_dim": energy_cfg.get("proj_hidden_dim", 64),
            "higher_order": energy_cfg.get("higher_order", {}).get("enabled", True),
            "higher_proj_dim": energy_cfg.get("higher_order", {}).get("proj_dim", 64),
            "higher_hidden_dim": energy_cfg.get("higher_order", {}).get("hidden_dim", 128),
        }
    elif energy_type == "seal":
        energy_kwargs = {
            "global_hidden_dim": energy_cfg.get("global_hidden_dim", 150),
        }

    loss_net = LossNet(
        input_dim=input_dim,
        num_labels=num_labels,
        energy_type=energy_type,
        feature_hidden_dims=feature_cfg.get("hidden_dims", [512, 512]),
        feature_dropout=feature_cfg.get("dropout", 0.3),
        **energy_kwargs,
    )
    logger.info("LossNet params: %d", sum(p.numel() for p in loss_net.parameters()))

    # Build trainer config
    train_cfg = cfg["training"]
    device = args.device or cfg.get("device", "cuda")
    trainer_config = TrainerConfig(
        total_epochs=train_cfg["total_epochs"],
        task_net_lr=train_cfg["task_net_lr"],
        loss_net_lr=train_cfg["loss_net_lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        num_steps_loss_net=train_cfg.get("num_steps_loss_net", 12),
        num_steps_task_net=train_cfg.get("num_steps_task_net", 5),
        lambda_energy=train_cfg.get("lambda_energy", 1.0),
        lambda_bce=train_cfg.get("lambda_bce", 1.0),
        ema_beta=train_cfg.get("ema_beta", 0.99),
        ema_warmup_beta=train_cfg.get("ema_warmup_beta", 0.9),
        ema_warmup_fraction=train_cfg.get("ema_warmup_fraction", 0.05),
        nce_samples=train_cfg.get("nce_samples", 32),
        nce_sampling=train_cfg.get("nce_sampling", "bernoulli"),
        nce_gaussian_sigma=train_cfg.get("nce_gaussian_sigma", 0.3),
        grad_clip=train_cfg.get("grad_clip", 10.0),
        lr_patience=train_cfg.get("lr_patience", 5),
        lr_factor=train_cfg.get("lr_factor", 0.5),
        log_interval=cfg.get("diagnostics", {}).get("log_interval", 50),
        track_diagnostics=True,
        experiment_dir=f"./experiment_result/{dataset_name}_{energy_type}",
        experiment_description=(
            args.desc or
            f"CREL training on {dataset_name} dataset with {energy_type} energy. "
            f"Rank={energy_kwargs.get('rank', 'N/A')}, "
            f"lambda_energy={train_cfg.get('lambda_energy', 1.0)}, "
            f"lambda_bce={train_cfg.get('lambda_bce', 1.0)}, "
            f"NCE samples={train_cfg.get('nce_samples', 32)}. "
            f"SEAL-dynamic minimax: {train_cfg.get('num_steps_task_net', 5)} outer x "
            f"{train_cfg.get('num_steps_loss_net', 12)} inner steps."
        ),
        device=device,
    )

    # Train
    trainer = CRELTrainer(
        task_net=task_net,
        loss_net=loss_net,
        train_loader=loaders["train"],
        val_loader=loaders.get("val"),
        test_loader=loaders.get("test"),
        config=trainer_config,
    )

    history = trainer.train()

    # Save checkpoint
    save_dir = Path(f"./experiment_result/{dataset_name}_{energy_type}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "final_checkpoint.pt"
    trainer.save_checkpoint(str(save_path))
    logger.info("Checkpoint saved to %s", save_path)

    # Log final summary
    if history.get("test_f1"):
        best_test_f1 = max(history["test_f1"])
        best_epoch = history["test_f1"].index(best_test_f1) + 1
        final_test_f1 = history["test_f1"][-1]
        summary = (
            f"\n{'='*60}\n"
            f"EXPERIMENT COMPLETE: {dataset_name} / {energy_type}\n"
            f"  Final test micro_F1: {final_test_f1:.4f}\n"
            f"  Best test micro_F1:  {best_test_f1:.4f} (epoch {best_epoch})\n"
            f"  Results logged to:   experiment_result/{dataset_name}_{energy_type}/epoch_results.txt\n"
            f"{'='*60}"
        )
        logger.info(summary)


if __name__ == "__main__":
    main()
