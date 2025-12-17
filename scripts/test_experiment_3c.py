"""Test Experiment 3c: Cross-attention with corrected training (no test feedback at iteration 0)."""

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


def test_iterative_refinement(
    model,
    dataset,
    trajectory_gen,
    corruption_rate,
    max_iterations,
    device,
    mask_token_id,
):
    """Test iterative refinement performance."""
    model.eval()
    one_shot_accuracies = []
    final_accuracies = []
    improvements = []

    for idx in tqdm(range(len(dataset)), desc=f"Testing {int(corruption_rate*100)}%"):
        _, clean_graph, tests = dataset[idx]

        # Corrupt
        corrupted = corrupt_graph(clean_graph, corruption_rate, mask_token_id)
        current_graph = corrupted.clone()

        with torch.no_grad():
            # One-shot (iteration 0, NO test feedback - this is the key fix!)
            current_graph_device = current_graph.to(device)
            output = model.forward_full(current_graph_device, iteration=0, test_feedback=None)
            predictions = output['logits'].argmax(dim=-1)

            one_shot_acc = (predictions == clean_graph.x[:, 0].to(device)).float().mean().item()
            one_shot_accuracies.append(one_shot_acc)

            # Update current graph with one-shot predictions
            current_graph.x[:, 1] = current_graph.x[:, 0].clone()
            current_graph.x[:, 0] = predictions.cpu().float()

            # Iterative refinement (iterations 1+ WITH test feedback)
            for iteration in range(1, max_iterations):
                # Compute test feedback
                test_feedback = trajectory_gen._compute_test_feedback(current_graph, tests)

                # Move to device
                test_feedback_device = {
                    'test_ids': test_feedback['test_ids'].to(device),
                    'test_statuses': test_feedback['test_statuses'].to(device),
                    'test_traces': test_feedback['test_traces'].to(device),
                }

                # Predict with test feedback
                current_graph_device = current_graph.to(device)
                current_graph_device.x[:, 4] = iteration  # Update iteration number

                output = model.forward_full(
                    current_graph_device,
                    iteration=iteration,
                    test_feedback=test_feedback_device
                )
                predictions = output['logits'].argmax(dim=-1)

                # Update current graph
                current_graph.x[:, 1] = current_graph.x[:, 0].clone()
                current_graph.x[:, 0] = predictions.cpu().float()

                # Check if converged
                accuracy = (predictions == clean_graph.x[:, 0].to(device)).float().mean().item()
                if accuracy == 1.0:
                    break

            # Final accuracy after refinement
            final_acc = (current_graph.x[:, 0].cpu() == clean_graph.x[:, 0].cpu()).float().mean().item()
            final_accuracies.append(final_acc)
            improvements.append(final_acc - one_shot_acc)

    return {
        'one_shot': np.mean(one_shot_accuracies),
        'final': np.mean(final_accuracies),
        'improvement': np.mean(improvements),
    }


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
        use_test_guidance=True,
    ).to(device)

    # Load trained weights
    checkpoint = torch.load('checkpoints/phase1_5_guided_fixed/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✓ Loaded Experiment 3c model (corrected training)")

    # Create trajectory generator for test feedback
    builder = ASGBuilder(vocab)
    interpreter = MiniLispInterpreter()
    trajectory_gen = GuidedTrajectoryGenerator(
        builder=builder,
        interpreter=interpreter,
        mask_token_id=vocab.mask_token_id,
        max_tests=100,
        max_nodes=1000,
    )

    print("\n" + "="*60)
    print("Experiment 3c: Cross-Attention with Corrected Training")
    print("(No test feedback at iteration 0)")
    print("="*60)

    # Test at different corruption levels
    corruption_levels = [0.2, 0.5, 0.75]
    max_iterations = 5

    results = {}

    for corruption_rate in corruption_levels:
        level_name = f"{int(corruption_rate * 100)}%"
        print(f"\n--- Testing at {level_name} corruption ---")

        # Test iterative refinement
        result = test_iterative_refinement(
            model=model,
            dataset=val_dataset,
            trajectory_gen=trajectory_gen,
            corruption_rate=corruption_rate,
            max_iterations=max_iterations,
            device=device,
            mask_token_id=vocab.mask_token_id,
        )

        results[level_name] = result

        print(f"One-shot accuracy:  {result['one_shot']:.3f}")
        print(f"Final accuracy:     {result['final']:.3f}")
        print(f"Improvement:        {result['improvement']:+.3f} ({result['improvement']*100:+.1f}%)")

    print("\n" + "="*60)
    print("Comparison Across All Experiments")
    print("="*60)

    print("\nExperiment 1 (6 features + test_signal):")
    print("  20%: 73.4% one-shot, +1.9% improvement")
    print("  50%: 57.7% one-shot, +1.9% improvement")

    print("\nExperiment 3b (5 features, no cross-attn):")
    print("  20%: 63.1% one-shot")
    print("  50%: 52.7% one-shot")

    print("\nExperiment 3 (5 features + cross-attn, BAD training):")
    print("  20%: 55.7% one-shot, +1.3% improvement")
    print("  50%: 46.9% one-shot, +0.1% improvement")
    print("  75%: 41.9% one-shot, +8.2% improvement")

    print("\nExperiment 3c (5 features + cross-attn, CORRECTED training):")
    if '20%' in results:
        print(f"  20%: {results['20%']['one_shot']:.1%} one-shot, {results['20%']['improvement']*100:+.1f}% improvement")
    if '50%' in results:
        print(f"  50%: {results['50%']['one_shot']:.1%} one-shot, {results['50%']['improvement']*100:+.1f}% improvement")
    if '75%' in results:
        print(f"  75%: {results['75%']['one_shot']:.1%} one-shot, {results['75%']['improvement']*100:+.1f}% improvement")

    print("\n" + "="*60)
    print("Analysis")
    print("="*60)

    # Check if one-shot improved vs Experiment 3
    one_shot_20_exp3 = 0.557
    one_shot_20_exp3c = results['20%']['one_shot']
    improvement_vs_exp3 = one_shot_20_exp3c - one_shot_20_exp3

    if improvement_vs_exp3 > 0.05:
        print(f"✅ SUCCESS: One-shot improved by {improvement_vs_exp3*100:.1f}% vs Experiment 3!")
        print(f"   Removing test feedback at iteration 0 fixed the dependency issue.")
    elif improvement_vs_exp3 > 0:
        print(f"⚠️  PARTIAL: One-shot improved by {improvement_vs_exp3*100:.1f}% vs Experiment 3")
        print(f"   Better, but still below baseline (73.4%)")
    else:
        print(f"❌ NO IMPROVEMENT: One-shot changed by {improvement_vs_exp3*100:.1f}% vs Experiment 3")


if __name__ == "__main__":
    main()
