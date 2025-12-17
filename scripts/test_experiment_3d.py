"""Test Experiment 3d: Cross-attention with scaled dataset (1000 samples)."""

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

        corrupted = corrupt_graph(clean_graph, corruption_rate, mask_token_id)
        current_graph = corrupted.clone()

        with torch.no_grad():
            # One-shot (iteration 0, NO test feedback)
            current_graph_device = current_graph.to(device)
            output = model.forward_full(current_graph_device, iteration=0, test_feedback=None)
            predictions = output['logits'].argmax(dim=-1)

            one_shot_acc = (predictions == clean_graph.x[:, 0].to(device)).float().mean().item()
            one_shot_accuracies.append(one_shot_acc)

            # Update current graph
            current_graph.x[:, 1] = current_graph.x[:, 0].clone()
            current_graph.x[:, 0] = predictions.cpu().float()

            # Iterative refinement (iterations 1+)
            for iteration in range(1, max_iterations):
                test_feedback = trajectory_gen._compute_test_feedback(current_graph, tests)

                test_feedback_device = {
                    'test_ids': test_feedback['test_ids'].to(device),
                    'test_statuses': test_feedback['test_statuses'].to(device),
                    'test_traces': test_feedback['test_traces'].to(device),
                }

                current_graph_device = current_graph.to(device)
                current_graph_device.x[:, 4] = iteration

                output = model.forward_full(
                    current_graph_device,
                    iteration=iteration,
                    test_feedback=test_feedback_device
                )
                predictions = output['logits'].argmax(dim=-1)

                current_graph.x[:, 1] = current_graph.x[:, 0].clone()
                current_graph.x[:, 0] = predictions.cpu().float()

                accuracy = (predictions == clean_graph.x[:, 0].to(device)).float().mean().item()
                if accuracy == 1.0:
                    break

            final_acc = (current_graph.x[:, 0].cpu() == clean_graph.x[:, 0].cpu()).float().mean().item()
            final_accuracies.append(final_acc)
            improvements.append(final_acc - one_shot_acc)

    return {
        'one_shot': np.mean(one_shot_accuracies),
        'final': np.mean(final_accuracies),
        'improvement': np.mean(improvements),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    vocab = Vocabulary.load('data/phase1_5/vocabulary.json')
    print(f"Vocabulary size: {vocab.vocab_size}")

    val_dataset = IterativeRefinementDataset(
        data_dir='data/phase1_5/val',
        corruption_rate=0.5,
        mask_token_id=vocab.mask_token_id,
    )
    print(f"Validation samples: {len(val_dataset)}")

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

    checkpoint = torch.load('checkpoints/phase1_5_guided_scaled/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint)
    print(f"✓ Loaded Experiment 3d model (scaled dataset: 1000 samples)")

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
    print("Experiment 3d: Scaled Dataset (1000 samples)")
    print("Cross-attention + corrected training + 12.5x more data")
    print("="*60)

    corruption_levels = [0.2, 0.5, 0.75]
    max_iterations = 5
    results = {}

    for corruption_rate in corruption_levels:
        level_name = f"{int(corruption_rate * 100)}%"
        print(f"\n--- Testing at {level_name} corruption ---")

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
    print("Full Comparison")
    print("="*60)

    print("\n| Experiment | Data Size | One-shot @ 20% | Iterative @ 20% | One-shot @ 50% | Iterative @ 50% |")
    print("|------------|-----------|----------------|-----------------|----------------|-----------------|")
    print("| Exp 1 | 80 | 73.4% | +1.9% | 57.7% | +1.9% |")
    print("| Exp 3c | 80 | 63.3% | -6.2% | 55.3% | -3.7% |")
    print(f"| Exp 3d | 1000 | {results['20%']['one_shot']:.1%} | {results['20%']['improvement']*100:+.1f}% | {results['50%']['one_shot']:.1%} | {results['50%']['improvement']*100:+.1f}% |")

    print("\n" + "="*60)
    print("Analysis")
    print("="*60)

    # Compare with Exp 3c (80 samples)
    one_shot_20_exp3c = 0.633
    one_shot_20_exp3d = results['20%']['one_shot']
    improvement_one_shot = one_shot_20_exp3d - one_shot_20_exp3c

    iter_20_exp3c = -0.062
    iter_20_exp3d = results['20%']['improvement']
    improvement_iterative = iter_20_exp3d - iter_20_exp3c

    print(f"\nOne-shot @ 20%: {one_shot_20_exp3d:.1%} (vs 63.3% with 80 samples)")
    print(f"  Change: {improvement_one_shot*100:+.1f}% {'✅' if improvement_one_shot > 0.01 else '⚠️' if improvement_one_shot > -0.01 else '❌'}")

    print(f"\nIterative improvement @ 20%: {iter_20_exp3d*100:+.1f}% (vs -6.2% with 80 samples)")
    print(f"  Change: {improvement_iterative*100:+.1f}pp {'✅' if iter_20_exp3d > 0 else '⚠️' if iter_20_exp3d > -0.03 else '❌'}")

    if iter_20_exp3d > 0:
        print("\n✅ SUCCESS: Iterative refinement now HELPS instead of hurting!")
        print("   Scaling data fixed the refinement issue.")
    elif iter_20_exp3d > -0.03:
        print("\n⚠️  PARTIAL: Iterative refinement less harmful but still not helpful")
        print("   May need even more data or different approach.")
    else:
        print("\n❌ FAILURE: Iterative refinement still makes things worse")
        print("   Data scaling did not solve the fundamental issue.")

    # Dataset quality note
    print("\n⚠️  NOTE: Dataset has 47.5% duplication rate (525 unique / 1000 total)")
    print("   High overfitting risk detected. Consider:")
    print("   - Increasing template randomization")
    print("   - Using data augmentation")
    print("   - Generating more diverse templates")


if __name__ == "__main__":
    main()
