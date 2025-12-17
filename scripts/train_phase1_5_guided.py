"""Training script for Phase 1.5 with Cross-Attention Test Feedback Guidance."""

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
from src.models.graph_unet import GuidedIterativeGraphUNet
from src.runtime.interpreter import MiniLispInterpreter
from src.training.denoising_task import IterativeRefinementLoss
from src.training.trajectory import GuidedTrajectoryGenerator
from src.training.denoising_metrics import IterativeRefinementMetrics
from src.training.curriculum import MixedCurriculumScheduler


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
    dataset: IterativeRefinementDataset,
    trajectory_gen: GuidedTrajectoryGenerator,
    criterion: IterativeRefinementLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_accumulation_steps: int,
    writer: SummaryWriter,
    epoch: int,
    log_every: int,
    batch_size: int = 1,
) -> dict[str, float]:
    """Train for one epoch using trajectory-based learning with cross-attention guidance."""
    model.train()
    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    optimizer.zero_grad()

    # Reset peak memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Create curriculum scheduler
    if not hasattr(train_epoch, '_curriculum'):
        train_epoch._curriculum = MixedCurriculumScheduler(mix_ratio=0.3)

    curriculum = train_epoch._curriculum
    stage_info = curriculum.get_current_stage_info(epoch)
    primary_rate = stage_info['primary_rate']

    pbar = tqdm(range(len(dataset)), desc=f"Epoch {epoch} (primary={primary_rate:.0%}, guided)")
    for batch_idx in pbar:
        # Get sample
        _, clean_graph, tests = dataset[batch_idx]
        clean_graph = clean_graph.to(device)

        # Sample corruption rate
        corruption_rates = curriculum.get_corruption_rate(epoch, batch_idx, batch_size=1)
        corruption_rate = corruption_rates[0]

        # Generate trajectory with test feedback
        trajectory = trajectory_gen.generate_trajectory(
            clean_graph=clean_graph,
            tests=tests,
            corruption_rate=corruption_rate,
            max_iterations=5,
            model=model,  # For scheduled sampling
        )

        # Train on each step in trajectory
        step_loss = 0.0
        step_metrics = {}

        for step in trajectory:
            # Prepare input
            input_graph = step.input_graph.to(device)
            target_graph = step.target_graph.to(device)

            # Move test feedback to device
            test_feedback = {
                'test_ids': step.test_feedback['test_ids'].to(device),
                'test_statuses': step.test_feedback['test_statuses'].to(device),
                'test_traces': step.test_feedback['test_traces'].to(device),
            }

            # Forward pass with test feedback guidance
            output = model.forward_full(
                data=input_graph,
                iteration=step.iteration,
                test_feedback=test_feedback
            )

            # Compute loss (standard token prediction loss)
            loss, metrics = criterion(
                predictions=output,
                current_graph=input_graph,
                target_graph=target_graph,
            )

            step_loss += loss
            for key, value in metrics.items():
                step_metrics[key] = step_metrics.get(key, 0.0) + value

        # Average over trajectory steps
        step_loss = step_loss / len(trajectory)
        for key in step_metrics:
            step_metrics[key] /= len(trajectory)

        # Backward pass with gradient accumulation
        (step_loss / gradient_accumulation_steps).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Update metrics
        total_loss += step_loss.item()
        for key, value in step_metrics.items():
            total_metrics[key] = total_metrics.get(key, 0.0) + value
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{step_loss.item():.4f}",
            'acc': f"{step_metrics.get('token_accuracy', 0.0):.3f}",
        })

        # Logging
        if (batch_idx + 1) % log_every == 0:
            global_step = epoch * len(dataset) + batch_idx
            writer.add_scalar('train/loss', step_loss.item(), global_step)
            for key, value in step_metrics.items():
                writer.add_scalar(f'train/{key}', value, global_step)

            # Log GPU memory
            mem_stats = get_gpu_memory_stats(device)
            for key, value in mem_stats.items():
                writer.add_scalar(f'memory/{key}', value, global_step)

    # Compute epoch averages
    avg_loss = total_loss / num_batches
    avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}

    return {'loss': avg_loss, **avg_metrics}


def validate(
    model: nn.Module,
    dataset: IterativeRefinementDataset,
    device: torch.device,
) -> dict[str, float]:
    """Validate model on dataset."""
    model.eval()
    total_accuracy = 0.0
    num_samples = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            corrupted, clean_graph, tests = dataset[idx]
            corrupted = corrupted.to(device)
            clean_graph = clean_graph.to(device)

            # One-shot prediction (no test feedback for validation)
            output = model.forward_full(corrupted, iteration=0, test_feedback=None)
            predictions = output['logits'].argmax(dim=-1)

            # Compute accuracy
            accuracy = (predictions == clean_graph.x[:, 0]).float().mean()
            total_accuracy += accuracy.item()
            num_samples += 1

    avg_accuracy = total_accuracy / num_samples
    return {'accuracy': avg_accuracy}


def main():
    parser = argparse.ArgumentParser(description="Train Phase 1.5 with Cross-Attention Guidance")
    parser.add_argument("--config", type=str, default="configs/phase1_5_guided.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    # Set seed
    set_seed(config.get('seed', 42))

    # Setup device
    device = torch.device(config.get('device', 'cuda'))

    # Load vocabulary
    vocab = Vocabulary.load(config['data']['vocabulary'])

    # Create datasets
    train_dataset = IterativeRefinementDataset(
        data_dir=config['data']['train_dir'],
        corruption_rate=0.5,
        mask_token_id=vocab.mask_token_id,
    )

    val_dataset = IterativeRefinementDataset(
        data_dir=config['data']['val_dir'],
        corruption_rate=0.5,
        mask_token_id=vocab.mask_token_id,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    model = GuidedIterativeGraphUNet(
        vocab_size=vocab.vocab_size,
        hidden_channels=config['model']['hidden_channels'],
        depth=config['model']['depth'],
        max_iterations=config['model']['max_iterations'],
        max_tests=config['model']['max_tests'],
        max_nodes=config['model']['max_nodes'],
        pool_ratio=config['model']['pool_ratio'],
        layer_type=config['model']['layer_type'],
        num_attention_heads=config['model']['num_attention_heads'],
        use_test_guidance=config['model']['use_test_guidance'],
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trajectory generator
    builder = ASGBuilder(vocab)
    interpreter = MiniLispInterpreter()

    trajectory_gen = GuidedTrajectoryGenerator(
        builder=builder,
        interpreter=interpreter,
        mask_token_id=vocab.mask_token_id,
        max_tests=config['model']['max_tests'],
        max_nodes=config['model']['max_nodes'],
        scheduled_sampling_max=config.get('scheduled_sampling_max', 0.95),
        scheduled_sampling_warmup=config.get('scheduled_sampling_warmup', 0.5),
    )

    # Create loss and optimizer
    criterion = IterativeRefinementLoss(vocab_size=vocab.vocab_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    # Scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config['training']['warmup_epochs'],
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'] - config['training']['warmup_epochs'],
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config['training']['warmup_epochs']],
    )

    # Tensorboard
    writer = SummaryWriter(log_dir=config['logging']['log_dir'])

    # Training loop
    best_val_acc = 0.0
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['training']['epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        print(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model=model,
            dataset=train_dataset,
            trajectory_gen=trajectory_gen,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            writer=writer,
            epoch=epoch,
            log_every=config['logging']['log_every'],
        )

        print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
        print(f"Train Accuracy: {train_metrics.get('token_accuracy', 0.0):.3f}")

        # Validate
        val_metrics = validate(model, val_dataset, device)
        print(f"Val Accuracy: {val_metrics['accuracy']:.3f}")

        # Log
        writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('train/epoch_loss', train_metrics['loss'], epoch)

        # Save checkpoint
        if (epoch + 1) % config['logging']['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
            }
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt')

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pt')
            print(f"✅ New best model saved (val_acc={best_val_acc:.3f})")

        scheduler.step()

    print(f"\n✅ Training complete! Best val accuracy: {best_val_acc:.3f}")
    writer.close()


if __name__ == "__main__":
    main()
