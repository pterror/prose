"""Test baseline model (use_test_guidance=False) for comparison."""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.data.vocabulary import Vocabulary
from src.data.dataset import IterativeRefinementDataset
from src.data.asg_builder import ASGBuilder
from src.models.graph_unet import GuidedIterativeGraphUNet
from src.runtime.interpreter import MiniLispInterpreter
from src.training.trajectory import GuidedTrajectoryGenerator


def corrupt_graph(graph, corruption_rate, mask_token_id):
    """Corrupt a graph by masking random tokens."""
    from torch_geometric.data import Data

    # Extract only first 5 features (no test_signal)
    if graph.x.size(1) > 5:
        x_clean = graph.x[:, :5].clone()
    else:
        x_clean = graph.x.clone()

    corrupted = Data(
        x=x_clean,
        edge_index=graph.edge_index.clone(),
        edge_attr=graph.edge_attr.clone() if hasattr(graph, 'edge_attr') else None,
    )

    if hasattr(graph, 'node_type'):
        corrupted.node_type = graph.node_type.clone()

    num_nodes = corrupted.x.size(0)

    if corruption_rate >= 0.9:
        corrupt_indices = list(range(num_nodes))
    else:
        num_corrupt = max(1, int(num_nodes * corruption_rate))
        import random
        corrupt_indices = random.sample(range(num_nodes), num_corrupt)

    for idx in corrupt_indices:
        corrupted.x[idx, 0] = mask_token_id
        corrupted.x[idx, 1] = mask_token_id

    return corrupted


def test_one_shot(model, dataset, corruption_rate, device, mask_token_id):
    """Test one-shot performance without iterative refinement."""
    model.eval()
    accuracies = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            _, clean_graph, tests = dataset[idx]

            # Corrupt
            corrupted = corrupt_graph(clean_graph, corruption_rate, mask_token_id).to(device)
            clean_graph = clean_graph.to(device)

            # One-shot prediction (no test feedback)
            output = model.forward_full(corrupted, iteration=0, test_feedback=None)
            predictions = output['logits'].argmax(dim=-1)

            # Accuracy
            accuracy = (predictions == clean_graph.x[:, 0]).float().mean().item()
            accuracies.append(accuracy)

    return np.mean(accuracies)


def main():
    # Load config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load vocabulary
    vocab = Vocabulary.load('data/phase1_5/vocabulary.json')
    print(f"Vocabulary size: {vocab.vocab_size}")

    # Create dataset
    val_dataset = IterativeRefinementDataset(
        data_dir='data/phase1_5/pilot',
        corruption_rate=0.5,
        mask_token_id=vocab.mask_token_id,
    )
    print(f"Validation samples: {len(val_dataset)}")

    # Load model
    model = GuidedIterativeGraphUNet(
        vocab_size=vocab.vocab_size,
        hidden_channels=256,
        depth=3,
        max_iterations=5,
        max_tests=100,
        max_nodes=1000,
        pool_ratio=0.5,
        layer_type='GAT',
        num_attention_heads=4,
        use_test_guidance=False,  # BASELINE (no cross-attention)
    ).to(device)

    # Load trained weights
    checkpoint = torch.load('checkpoints/phase1_5_guided_baseline/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✓ Loaded baseline model (use_test_guidance=False)")

    print("\n" + "="*60)
    print("Testing Baseline Model (No Cross-Attention)")
    print("="*60)

    # Test at different corruption levels (one-shot only)
    corruption_levels = [0.2, 0.5, 0.75]

    results = {}

    for corruption_rate in corruption_levels:
        level_name = f"{int(corruption_rate * 100)}%"
        print(f"\n--- Testing at {level_name} corruption ---")

        # Test one-shot only
        one_shot_acc = test_one_shot(
            model=model,
            dataset=val_dataset,
            corruption_rate=corruption_rate,
            device=device,
            mask_token_id=vocab.mask_token_id,
        )

        results[level_name] = {'one_shot': one_shot_acc}
        print(f"One-shot accuracy:  {one_shot_acc:.3f}")

    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    print("\nExperiment 1 (6 features, test_signal, 10x weighting):")
    print("  20%: 73.4% one-shot")
    print("  50%: 57.7% one-shot")

    print("\nExperiment 3 (5 features, cross-attention):")
    print("  20%: 55.7% one-shot")
    print("  50%: 46.9% one-shot")

    print("\nBaseline (5 features, NO cross-attention):")
    if '20%' in results:
        print(f"  20%: {results['20%']['one_shot']:.1%} one-shot")
    if '50%' in results:
        print(f"  50%: {results['50%']['one_shot']:.1%} one-shot")
    if '75%' in results:
        print(f"  75%: {results['75%']['one_shot']:.1%} one-shot")


if __name__ == "__main__":
    main()
