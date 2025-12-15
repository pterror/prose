"""Evaluation script for trained Graph U-Net models."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DenoisingGraphDataset
from src.models.graph_unet import GraphUNet
from src.training.denoising_task import collate_graph_pairs
from src.training.denoising_metrics import DenoisingMetrics
from src.utils.visualize import ASGVisualizer


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    metrics_calculator: DenoisingMetrics,
    device: torch.device,
) -> dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        metrics_calculator: Metrics calculator
        device: Device to run on

    Returns:
        Dictionary with aggregated metrics
    """
    model.eval()

    num_batches = 0
    all_metrics = {
        "exact_match": 0.0,
        "node_accuracy": 0.0,
        "edge_f1": 0.0,
        "edge_precision": 0.0,
        "edge_recall": 0.0,
        "syntax_valid": 0.0,
    }

    # Store samples for visualization
    viz_samples = []

    for corrupted, original in tqdm(test_loader, desc="Evaluating"):
        corrupted = corrupted.to(device)
        original = original.to(device)

        # Forward pass
        predictions = model.forward_full(corrupted)

        # Compute metrics
        batch_metrics = metrics_calculator.compute_all(predictions, original)

        # Accumulate metrics
        for key in all_metrics:
            all_metrics[key] += batch_metrics[key]

        num_batches += 1

        # Store first batch for visualization (select diverse samples)
        if len(viz_samples) < 20:
            # Get individual graphs from batch
            corrupted_graphs = corrupted.to_data_list()
            pred_graphs = predictions.to_data_list()
            gt_graphs = original.to_data_list()

            for corr, pred, gt in zip(corrupted_graphs, pred_graphs, gt_graphs):
                if len(viz_samples) >= 20:
                    break

                # Move to CPU for visualization
                viz_samples.append(
                    {
                        "corrupted": corr.cpu(),
                        "prediction": pred.cpu(),
                        "ground_truth": gt.cpu(),
                    }
                )

    # Average metrics across batches
    for key in all_metrics:
        all_metrics[key] /= num_batches if num_batches > 0 else 1

    return {
        "metrics": all_metrics,
        "num_samples": len(test_loader.dataset),
        "num_batches": num_batches,
        "viz_samples": viz_samples,
    }


def generate_visualizations(
    viz_samples: list[dict],
    output_dir: Path,
    visualizer: ASGVisualizer,
) -> None:
    """
    Generate visualizations for sample reconstructions.

    Args:
        viz_samples: List of sample dictionaries
        output_dir: Directory to save visualizations
        visualizer: ASG visualizer instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {len(viz_samples)} visualizations...")

    for i, sample in enumerate(tqdm(viz_samples, desc="Visualizing")):
        output_path = output_dir / f"reconstruction_{i:03d}.png"

        visualizer.visualize_reconstruction(
            corrupted=sample["corrupted"],
            prediction=sample["prediction"],
            ground_truth=sample["ground_truth"],
            output_path=output_path,
            title=f"Sample {i}",
        )

        # Also generate code renderings
        code_output_path = output_dir / f"code_{i:03d}.txt"
        with open(code_output_path, "w") as f:
            f.write("=== Corrupted ===\n")
            f.write(visualizer.render_mini_lisp(sample["corrupted"]))
            f.write("\n\n=== Prediction ===\n")
            f.write(visualizer.render_mini_lisp(sample["prediction"]))
            f.write("\n\n=== Ground Truth ===\n")
            f.write(visualizer.render_mini_lisp(sample["ground_truth"]))
            f.write("\n")


def save_results(
    results: dict,
    output_path: Path,
    checkpoint_path: Path,
    config: dict,
) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        results: Evaluation results
        output_path: Path to save results
        checkpoint_path: Path to checkpoint that was evaluated
        config: Model configuration
    """
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "config": config,
        "metrics": results["metrics"],
        "num_samples": results["num_samples"],
        "num_batches": results["num_batches"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def print_metrics(metrics: dict) -> None:
    """Print metrics in a formatted table."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(
        f"Exact Match Rate:    {metrics['exact_match']:.4f} ({metrics['exact_match'] * 100:.2f}%)"
    )
    print(
        f"Node Accuracy:       {metrics['node_accuracy']:.4f} ({metrics['node_accuracy'] * 100:.2f}%)"
    )
    print(f"Edge F1 Score:       {metrics['edge_f1']:.4f}")
    print(f"  - Precision:       {metrics['edge_precision']:.4f}")
    print(f"  - Recall:          {metrics['edge_recall']:.4f}")
    print(
        f"Syntax Validity:     {metrics['syntax_valid']:.4f} ({metrics['syntax_valid'] * 100:.2f}%)"
    )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Graph U-Net")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/processed/test"),
        help="Path to test data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation (faster evaluation)",
    )

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    config = checkpoint["config"]
    epoch = checkpoint["epoch"]

    print(f"Checkpoint from epoch {epoch}")
    print(f"  Val Loss: {checkpoint.get('val_metrics', {}).get('loss', 'N/A')}")
    print(f"  Val Accuracy: {checkpoint.get('val_metrics', {}).get('accuracy', 'N/A')}")

    # Set device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # Create test dataset
    print("\nLoading test dataset...")
    test_dataset = DenoisingGraphDataset(
        args.test_dir,
        corruption_rate=config["data"]["corruption_rate"],
        mask_token_id=config["data"]["mask_token_id"],
        seed=config["seed"] + 2,  # Different seed from train/val
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_graph_pairs,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Create model
    print("\nCreating model...")
    model = GraphUNet(
        in_channels=config["model"]["in_channels"],
        hidden_channels=config["model"]["hidden_channels"],
        out_channels=config["model"]["out_channels"],
        depth=config["model"]["depth"],
        pool_ratio=config["model"]["pool_ratio"],
        num_node_types=config["model"]["num_node_types"],
        layer_type=config["model"]["layer_type"],
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Create metrics calculator
    metrics_calculator = DenoisingMetrics(num_node_types=config["model"]["num_node_types"])

    # Run evaluation
    print("\n" + "=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)

    results = evaluate_model(model, test_loader, metrics_calculator, device)

    # Print metrics
    print_metrics(results["metrics"])

    # Generate visualizations
    if not args.no_viz:
        visualizer = ASGVisualizer()
        viz_dir = args.output_dir / "visualizations"
        generate_visualizations(results["viz_samples"], viz_dir, visualizer)
        print(f"\nVisualizations saved to: {viz_dir}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = args.output_dir / f"evaluation_{timestamp}.json"
    save_results(results, results_path, args.checkpoint, config)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
