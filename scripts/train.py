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
from crel.utils.pretrained import load_seal_pretrained_into_crel

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
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained checkpoint (SEAL best_model.pt or CREL checkpoint)")
    parser.add_argument("--bibtex-seal-split", action="store_true",
                        help="Use SEAL's MEKA fold split for bibtex (train 1-6, val 7-8, test 9-10)")
    parser.add_argument("--bibtex-folds-dir", type=str, default=None,
                        help="Path to MEKA fold dir (Bibtex-fold1.arff ...). Default: data_dir/bibtex_stratified10folds_meka")
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
    pretrained_path = args.pretrained or cfg.get("pretrained_path") or None
    if pretrained_path:
        cfg["pretrained_path"] = str(Path(pretrained_path).resolve())

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
    bibtex_folds_dir = getattr(args, "bibtex_folds_dir", None) or cfg["dataset"].get("bibtex_folds_dir")
    if getattr(args, "bibtex_seal_split", False) and dataset_name == "bibtex":
        bibtex_folds_dir = bibtex_folds_dir or str(Path(data_dir) / "bibtex_stratified10folds_meka")
        logger.info("Using SEAL bibtex split: %s", bibtex_folds_dir)

    loaders = create_data_loaders(
        name=dataset_name,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        bibtex_folds_dir=bibtex_folds_dir,
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

    # If using SEAL pretrained (e.g. seal_dynamic_nce best_model.pt), use SEAL-compatible architecture
    pretrained_ckpt = None
    if pretrained_path:
        pretrained_path = str(Path(pretrained_path).resolve())
        if not Path(pretrained_path).is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        pretrained_ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if isinstance(pretrained_ckpt, dict) and "task_nn" in pretrained_ckpt and "score_nn" in pretrained_ckpt:
            if dataset_name == "bibtex":
                cfg.setdefault("task_net", {})["hidden_dims"] = [400, 400]
                cfg.setdefault("feature_net", {})["hidden_dims"] = [400, 400]
                if energy_type == "seal":
                    cfg.setdefault("energy", {})["global_hidden_dim"] = 200
                logger.info(
                    "SEAL pretrained detected: using hidden_dims [400, 400] and global_hidden_dim 200 for compatibility"
                )
            logger.warning(
                "SEAL pretrained: ensure the checkpoint was trained on the SAME train/val/test split "
                "as this run (e.g. CREL data/bibtex/train.arff, val.arff, test.arff). "
                "Using a different split (e.g. MEKA folds 1-6/7-8/9-10) causes train-test leakage and overly optimistic results."
            )

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

    # Load pretrained weights if requested
    if pretrained_path and isinstance(pretrained_ckpt, dict):
        device = args.device or cfg.get("device", "cuda")
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt = pretrained_ckpt
        if "task_nn" in ckpt and "score_nn" in ckpt:
            load_seal_pretrained_into_crel(pretrained_path, task_net, loss_net, device, ckpt=ckpt)
            logger.info("Loaded SEAL pretrained weights from %s", pretrained_path)
        elif "task_net" in ckpt and "loss_net" in ckpt:
            task_net.load_state_dict(ckpt["task_net"], strict=False)
            loss_net.load_state_dict(ckpt["loss_net"], strict=False)
            logger.info("Loaded CREL checkpoint (model weights only) from %s", pretrained_path)
        else:
            raise ValueError(
                f"Unrecognized checkpoint format at {pretrained_path}. "
                "Expected SEAL (task_nn, score_nn) or CREL (task_net, loss_net)."
            )

    # Build trainer config
    train_cfg = cfg["training"]
    device = args.device or cfg.get("device", "cuda")
    trainer_config = TrainerConfig(
        total_epochs=train_cfg["total_epochs"],
        task_net_lr=train_cfg["task_net_lr"],
        loss_net_lr=train_cfg["loss_net_lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        lambda_energy=train_cfg.get("lambda_energy", 1.0),
        lambda_bce=train_cfg.get("lambda_bce", 1.0),
        ema_beta=train_cfg.get("ema_beta", 0.99),
        ema_warmup_beta=train_cfg.get("ema_warmup_beta", 0.9),
        ema_warmup_fraction=train_cfg.get("ema_warmup_fraction", 0.05),
        contrastive_loss=train_cfg.get("contrastive_loss", "infonce"),
        nce_samples=train_cfg.get("nce_samples", 32),
        nce_sampling=train_cfg.get("nce_sampling", "bernoulli"),
        nce_gaussian_sigma=train_cfg.get("nce_gaussian_sigma", 0.3),
        infonce_temperature=train_cfg.get("infonce_temperature", 1.0),
        energy_reg=train_cfg.get("energy_reg", 0.01),
        stop_gradient_energy=train_cfg.get("stop_gradient_energy", True),
        grad_clip=train_cfg.get("grad_clip", 5.0),
        log_interval=cfg.get("diagnostics", {}).get("log_interval", 50),
        track_diagnostics=True,
        experiment_dir=f"./experiment_result/{dataset_name}_{energy_type}",
        experiment_description=(
            args.desc or
            f"CREL training on {dataset_name} dataset with {energy_type} energy. "
            f"Rank={energy_kwargs.get('rank', 'N/A')}, "
            f"lambda_energy={train_cfg.get('lambda_energy', 1.0)}, "
            f"lambda_bce={train_cfg.get('lambda_bce', 1.0)}, "
            f"NCE samples={train_cfg.get('nce_samples', 32)}."
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
