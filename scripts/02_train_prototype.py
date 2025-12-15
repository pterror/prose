"""Main training script for Graph U-Net."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import DenoisingGraphDataset
from src.models.graph_unet import GraphUNet
from src.training.denoising_task import DenoisingLoss, collate_graph_pairs


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_gpu_memory_stats(device: torch.device) -> dict[str, float]:
    """Get GPU memory usage statistics."""
    if device.type == "cuda":
        return {
            "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        }
    return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_accumulation_steps: int,
    writer: SummaryWriter,
    epoch: int,
    log_every: int,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    optimizer.zero_grad()

    # Reset peak memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (corrupted, original) in enumerate(pbar):
        corrupted = corrupted.to(device)
        original = original.to(device)

        # Forward pass
        predictions = model.forward_full(corrupted)

        # Compute loss
        loss, metrics = criterion(predictions, original)

        # Backward pass with gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Compute gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norm += grad_norm.item()

            optimizer.step()
            optimizer.zero_grad()

        # Track metrics
        total_loss += metrics["loss"]
        total_accuracy += metrics["accuracy"]
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": metrics["loss"], "acc": metrics["accuracy"]})

        # Log to tensorboard
        if batch_idx % log_every == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("train/loss", metrics["loss"], global_step)
            writer.add_scalar("train/accuracy", metrics["accuracy"], global_step)

            # Log GPU memory usage
            mem_stats = get_gpu_memory_stats(device)
            writer.add_scalar("gpu/allocated_gb", mem_stats["allocated_gb"], global_step)
            writer.add_scalar("gpu/max_allocated_gb", mem_stats["max_allocated_gb"], global_step)

    avg_grad_norm = total_grad_norm / (num_batches / gradient_accumulation_steps)

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_accuracy / num_batches,
        "grad_norm": avg_grad_norm,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for corrupted, original in tqdm(val_loader, desc="Validating"):
        corrupted = corrupted.to(device)
        original = original.to(device)

        # Forward pass
        predictions = model.forward_full(corrupted)

        # Compute loss
        loss, metrics = criterion(predictions, original)

        total_loss += metrics["loss"]
        total_accuracy += metrics["accuracy"]
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_accuracy / num_batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Graph U-Net")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase1_prototype.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI args if provided
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # Set seed
    set_seed(config["seed"])

    # Device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    print("Loading datasets...")
    train_dataset = DenoisingGraphDataset(
        Path(config["data"]["train_dir"]),
        corruption_rate=config["data"]["corruption_rate"],
        mask_token_id=config["data"]["mask_token_id"],
        seed=config["seed"],
    )

    val_dataset = DenoisingGraphDataset(
        Path(config["data"]["val_dir"]),
        corruption_rate=config["data"]["corruption_rate"],
        mask_token_id=config["data"]["mask_token_id"],
        seed=config["seed"] + 1,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_graph_pairs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_graph_pairs,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    print("Creating model...")
    model = GraphUNet(
        in_channels=config["model"]["in_channels"],
        hidden_channels=config["model"]["hidden_channels"],
        out_channels=config["model"]["out_channels"],
        depth=config["model"]["depth"],
        pool_ratio=config["model"]["pool_ratio"],
        num_node_types=config["model"]["num_node_types"],
        layer_type=config["model"]["layer_type"],
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Loss and optimizer
    criterion = DenoisingLoss(num_node_types=config["model"]["num_node_types"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Learning rate scheduler with warmup
    warmup_epochs = config["training"].get("warmup_epochs", 5)
    total_epochs = config["training"]["epochs"]

    if warmup_epochs > 0:
        # Warmup: linear increase from lr/10 to lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # Main: cosine annealing after warmup
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)

        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        # No warmup, just cosine annealing
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    # Logging
    log_dir = Path(config["logging"]["log_dir"])
    checkpoint_dir = Path(config["logging"]["checkpoint_dir"])
    log_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")

    # GPU profiling data
    gpu_profile = [] if config.get("training", {}).get("profile_gpu", True) else None

    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'=' * 60}")

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config["training"]["gradient_accumulation_steps"],
            writer,
            epoch,
            config["logging"]["log_every"],
        )

        print(
            f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
            f"Accuracy: {train_metrics['accuracy']:.4f}, "
            f"Grad Norm: {train_metrics['grad_norm']:.4f}"
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        print(f"Val - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")

        # Log to tensorboard
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("train/grad_norm", train_metrics["grad_norm"], epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        # Log GPU memory stats
        mem_stats = get_gpu_memory_stats(device)
        for key, value in mem_stats.items():
            writer.add_scalar(f"gpu/{key}", value, epoch)

        print(
            f"GPU Memory: {mem_stats['allocated_gb']:.2f} GB allocated, "
            f"{mem_stats['max_allocated_gb']:.2f} GB peak"
        )

        # Save GPU profile
        if gpu_profile is not None:
            gpu_profile.append(
                {
                    "epoch": epoch,
                    **mem_stats,
                }
            )

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        if epoch % config["logging"]["save_every"] == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                },
                best_model_path,
            )
            print(f"New best model! Saved to: {best_model_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)

    # Save GPU profile
    if gpu_profile is not None:
        profile_path = Path(config["logging"]["log_dir"]) / "gpu_profile.json"
        with open(profile_path, "w") as f:
            json.dump(gpu_profile, f, indent=2)
        print(f"\nGPU profile saved to: {profile_path}")

    writer.close()


if __name__ == "__main__":
    main()
