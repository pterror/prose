"""Training script for Phase 1.5 Iterative Refinement."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.asg_builder import ASGBuilder
from src.data.dataset import IterativeRefinementDataset
from src.data.vocabulary import Vocabulary
from src.models.graph_unet import IterativeGraphUNet
from src.runtime.interpreter import MiniLispInterpreter
from src.training.denoising_task import IterativeRefinementLoss
from src.training.trajectory import TrajectoryGenerator
from src.training.denoising_metrics import IterativeRefinementMetrics


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


def get_curriculum_params(epoch: int) -> tuple[float, bool]:
    """
    Get corruption rate and structure preservation flag for curriculum.

    Curriculum stages (from phase1.5.md):
    - Stage 1 (0-5): 20% corruption, keep structure
    - Stage 2 (6-15): 50% corruption, keep structure
    - Stage 3 (16-25): 75% corruption, keep structure
    - Stage 4 (26-40): 90% corruption, no structure
    - Stage 5 (41+): 100% corruption (full generation)

    Returns:
        Tuple of (corruption_rate, keep_structure)
    """
    if epoch < 6:
        return 0.2, True
    elif epoch < 16:
        return 0.5, True
    elif epoch < 26:
        return 0.75, True
    elif epoch < 41:
        return 0.9, False
    else:
        return 1.0, False


def train_epoch(
    model: nn.Module,
    dataset: IterativeRefinementDataset,
    trajectory_gen: TrajectoryGenerator,
    criterion: IterativeRefinementLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_accumulation_steps: int,
    writer: SummaryWriter,
    epoch: int,
    log_every: int,
    batch_size: int = 1,
) -> dict[str, float]:
    """Train for one epoch using trajectory-based learning."""
    model.train()
    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    optimizer.zero_grad()

    # Reset peak memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Determine corruption rate based on curriculum
    corruption_rate, keep_structure = get_curriculum_params(epoch)

    pbar = tqdm(range(len(dataset)), desc=f"Epoch {epoch} (corruption={corruption_rate:.0%})")
    for batch_idx in pbar:
        # Get sample (dataset returns tuple: corrupted, original, tests)
        _, clean_graph, tests = dataset[batch_idx]
        clean_graph = clean_graph.to(device)

        # Generate trajectory
        trajectory = trajectory_gen.generate_trajectory(
            clean_graph=clean_graph,
            tests=tests,
            corruption_rate=corruption_rate,
            max_iterations=5,
        )

        # Train on each step in trajectory
        step_loss = 0.0
        step_metrics = {}

        for step in trajectory:
            # Prepare input features
            input_graph = step.input_graph.to(device)
            target_graph = step.target_graph.to(device)

            # Forward pass
            output = model(
                x=input_graph.x,
                edge_index=input_graph.edge_index,
                edge_type=input_graph.edge_type,
                batch=None,
                iteration=step.iteration,
            )

            # Compute loss
            loss, metrics = criterion(
                predictions=output["logits"],
                targets=target_graph.x[:, 0].long(),  # token_ids
                current=input_graph.x[:, 0].long(),
                confidence=output["confidence"],
            )

            step_loss += loss
            for key, value in metrics.items():
                step_metrics[key] = step_metrics.get(key, 0.0) + value

        # Average over trajectory steps
        step_loss = step_loss / len(trajectory)
        for key in step_metrics:
            step_metrics[key] = step_metrics[key] / len(trajectory)

        # Backward pass with gradient accumulation
        step_loss = step_loss / gradient_accumulation_steps
        step_loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

        # Track metrics
        total_loss += step_metrics["loss"]
        for key, value in step_metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value

        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": step_metrics["loss"],
            "acc": f"{step_metrics['accuracy']:.3f}",
        })

        # Log to TensorBoard
        if (batch_idx + 1) % log_every == 0:
            global_step = epoch * len(dataset) + batch_idx
            writer.add_scalar("train/batch_loss", step_metrics["loss"], global_step)
            writer.add_scalar("train/batch_accuracy", step_metrics["accuracy"], global_step)

    # Compute epoch averages
    avg_metrics = {
        "loss": total_loss / num_batches,
        **{k: v / num_batches for k, v in total_metrics.items()},
    }

    # Log GPU memory
    if device.type == "cuda":
        mem_stats = get_gpu_memory_stats(device)
        writer.add_scalar("gpu/allocated_gb", mem_stats["allocated_gb"], epoch)
        writer.add_scalar("gpu/max_allocated_gb", mem_stats["max_allocated_gb"], epoch)

    return avg_metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataset: IterativeRefinementDataset,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Validate model on dataset."""
    model.eval()
    metrics_calculator = IterativeRefinementMetrics()

    total_accuracy = 0.0
    total_confidence = 0.0
    num_samples = 0

    pbar = tqdm(range(len(dataset)), desc=f"Validation")
    for idx in pbar:
        # Get sample (dataset returns tuple: corrupted, original, tests)
        _, clean_graph, tests = dataset[idx]
        clean_graph = clean_graph.to(device)

        # Corrupt at 50% for validation
        from src.training.trajectory import corrupt_program
        corrupted = corrupt_program(clean_graph, corruption_rate=0.5, keep_structure=True)

        # Single-step prediction
        output = model(
            x=corrupted.x,
            edge_index=corrupted.edge_index,
            edge_type=corrupted.edge_type,
            batch=None,
            iteration=0,
        )

        # Compute accuracy
        predictions = output["logits"].argmax(dim=-1)
        targets = clean_graph.x[:, 0].long()
        accuracy = (predictions == targets).float().mean().item()
        confidence = output["confidence"].mean().item()

        total_accuracy += accuracy
        total_confidence += confidence
        num_samples += 1

        pbar.set_postfix({"acc": f"{accuracy:.3f}", "conf": f"{confidence:.3f}"})

    return {
        "accuracy": total_accuracy / num_samples,
        "confidence": total_confidence / num_samples,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: dict,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> None:
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Phase 1.5 Iterative Refinement Model")
    parser.add_argument("--config", type=str, default="configs/phase1_5.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Using default Phase 1 config as template...")
        config = load_config(Path("configs/phase1_prototype.yaml"))
    else:
        config = load_config(config_path)

    # Override config with CLI args
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    # Set seed
    set_seed(config.get("seed", 42))

    # Setup device
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load vocabulary
    vocab_path = Path("data/phase1_5/vocabulary.json")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}. Run scripts/build_vocabulary.py first.")

    vocab = Vocabulary.load(vocab_path)
    print(f"Loaded vocabulary: {vocab.vocab_size} tokens")

    # Load datasets
    train_dataset = IterativeRefinementDataset(
        data_dir=Path("data/phase1_5/train"),
        mask_token_id=vocab.mask_token_id,
    )
    val_dataset = IterativeRefinementDataset(
        data_dir=Path("data/phase1_5/pilot"),  # Use pilot as validation
        mask_token_id=vocab.mask_token_id,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    model = IterativeGraphUNet(
        vocab_size=vocab.vocab_size,
        hidden_channels=config["model"].get("hidden_channels", 256),
        depth=config["model"].get("depth", 3),
        pool_ratio=config["model"].get("pool_ratio", 0.5),
        max_iterations=5,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create ASGBuilder and interpreter for trajectory generation
    asg_builder = ASGBuilder()
    interpreter = MiniLispInterpreter()

    # Create trajectory generator
    trajectory_gen = TrajectoryGenerator(
        builder=asg_builder,
        interpreter=interpreter,
        mask_token_id=vocab.mask_token_id,
    )

    # Create loss function
    criterion = IterativeRefinementLoss(
        vocab_size=vocab.vocab_size,
        reconstruction_weight=1.0,
        stability_weight=0.1,
        correction_weight=0.5,
        confidence_weight=0.2,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].get("lr", 0.001),
        weight_decay=config["training"].get("weight_decay", 0.0001),
    )

    # Create scheduler
    warmup_epochs = config["training"].get("warmup_epochs", 5)
    total_epochs = config["training"]["epochs"]

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Create TensorBoard writer
    log_dir = Path(config["logging"]["log_dir"]) / "phase1_5"
    writer = SummaryWriter(log_dir=log_dir)

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_accuracy = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_accuracy = checkpoint["metrics"].get("val_accuracy", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    checkpoint_dir = Path(config["logging"]["checkpoint_dir"]) / "phase1_5"
    save_every = config["logging"].get("save_every", 5)
    log_every = config["logging"].get("log_every", 100)

    print("\nStarting training...")
    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_metrics = train_epoch(
            model=model,
            dataset=train_dataset,
            trajectory_gen=trajectory_gen,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 4),
            writer=writer,
            epoch=epoch,
            log_every=log_every,
        )

        # Validate
        val_metrics = validate_epoch(
            model=model,
            dataset=val_dataset,
            device=device,
            epoch=epoch,
        )

        # Log metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - Acc: {val_metrics['accuracy']:.4f}, Conf: {val_metrics['confidence']:.4f}")

        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/confidence", val_metrics["confidence"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Save checkpoint
        is_best = val_metrics["accuracy"] > best_val_accuracy
        if is_best:
            best_val_accuracy = val_metrics["accuracy"]

        if (epoch + 1) % save_every == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics},
                checkpoint_dir=checkpoint_dir,
                is_best=is_best,
            )

        # Step scheduler
        scheduler.step()

    writer.close()
    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
