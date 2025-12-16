"""
Demo script for Phase 1.5 Iterative Refinement System.

Showcases:
1. Graph corruption and masking
2. Iterative refinement loop
3. Evaluation metrics tracking
4. Convergence visualization
"""

import torch
from pathlib import Path

from src.data.dataset import IterativeRefinementDataset
from src.data.vocabulary import Vocabulary
from src.models.graph_unet import IterativeGraphUNet
from src.inference.inference import IterativeRefinementInference, create_masked_graph
from src.runtime.interpreter import MiniLispInterpreter
from src.training.denoising_metrics import IterativeRefinementMetrics
from src.training.trajectory import corrupt_program_curriculum


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_graph_info(graph, vocab: Vocabulary, name: str = "Graph") -> None:
    """Print information about a graph."""
    print(f"{name}:")
    print(f"  Nodes: {graph.x.size(0)}")
    print(f"  Edges: {graph.edge_index.size(1)}")
    print(f"  Tokens: ", end="")

    # Decode first 10 tokens
    tokens = graph.x[:, 0].long()
    decoded = [vocab.decode(t.item()) for t in tokens[:min(10, len(tokens))]]
    print(", ".join(decoded))
    if len(tokens) > 10:
        print(f" ... (+{len(tokens) - 10} more)")
    else:
        print()


def print_iteration_metrics(metrics: dict, iteration: int) -> None:
    """Print metrics for a single iteration."""
    print(f"Iteration {iteration}:")
    print(f"  Accuracy:   {metrics['accuracy']:.3f}")
    print(f"  Precision:  {metrics['precision']:.3f}")
    print(f"  Recall:     {metrics['recall']:.3f}")
    print(f"  F1:         {metrics['f1']:.3f}")
    print(f"  Stability:  {metrics['stability']:.3f}")
    print(f"  Confidence: {metrics['mean_confidence']:.3f}")
    print(f"  Changed:    {metrics['num_changed']} nodes")
    print()


def print_trajectory_summary(metrics: dict) -> None:
    """Print summary of entire refinement trajectory."""
    print("Trajectory Summary:")
    print(f"  Initial Accuracy:  {metrics['initial_accuracy']:.3f}")
    print(f"  Final Accuracy:    {metrics['final_accuracy']:.3f}")
    print(f"  Improvement:       {metrics['improvement']:+.3f}")
    print(f"  Total Iterations:  {metrics['num_iterations']}")
    print(f"  Converged:         {'✓' if metrics['converged'] else '✗'}")
    print(f"  Perfect:           {'✓' if metrics['perfect'] else '✗'}")
    print(f"  Avg Confidence:    {metrics['avg_confidence']:.3f}")
    print(f"  Final F1:          {metrics['final_f1']:.3f}")


def demo_basic_refinement():
    """Demonstrate basic iterative refinement."""
    print_section("Phase 1.5 Iterative Refinement Demo")

    # Load dataset
    print("Loading dataset...")
    dataset_dir = Path("data/phase1_5/pilot")
    vocab_path = Path("data/phase1_5/vocabulary.json")

    if not dataset_dir.exists() or not vocab_path.exists():
        print(f"❌ Dataset not found at {dataset_dir}")
        print("   Please run: python scripts/generate_phase1_5_dataset.py")
        return

    # Load vocabulary
    vocab = Vocabulary.load(str(vocab_path))

    # Load samples
    sample_files = sorted(dataset_dir.glob("sample_*.pt"))
    samples = [torch.load(f, weights_only=False) for f in sample_files]

    print(f"✓ Loaded {len(samples)} samples")
    print(f"✓ Vocabulary size: {vocab.vocab_size}")

    # Select a sample
    sample = samples[0]
    original_graph = sample.graph

    print_section("1. Original Program")
    print_graph_info(original_graph, vocab, "Original")

    # Create corrupted version
    print_section("2. Corrupting Program")
    corrupted = corrupt_program_curriculum(
        original_graph,
        epoch=25,  # 75% corruption
        total_epochs=50,
        mask_token_id=0
    )
    print_graph_info(corrupted, vocab, "Corrupted")

    # Calculate initial accuracy
    original_tokens = original_graph.x[:, 0].long()
    corrupted_tokens = corrupted.x[:, 0].long()
    initial_accuracy = (corrupted_tokens == original_tokens).float().mean()
    print(f"  Initial Accuracy: {initial_accuracy:.1%}")

    # Initialize model
    print_section("3. Initializing Model")
    model = IterativeGraphUNet(
        vocab_size=vocab.vocab_size,
        hidden_channels=256,
        depth=3,
        max_iterations=10,
    )
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("  Note: Model is randomly initialized (not trained)")

    # Initialize inference
    interpreter = MiniLispInterpreter()
    inference = IterativeRefinementInference(
        model=model,
        interpreter=interpreter,
        mask_token_id=0,
        max_iterations=10,
        confidence_threshold=0.95,
    )
    print("✓ Inference engine ready")

    # Run refinement
    print_section("4. Running Iterative Refinement")
    print("Refining program toward target...\n")

    refined, metadata = inference.refine_program(
        corrupted,
        original_graph,
        verbose=True,
    )

    # Show results
    print_section("5. Refinement Results")
    print_graph_info(refined, vocab, "Refined")

    print(f"\nIterations:       {metadata['iterations']}")
    print(f"Final Accuracy:   {metadata['final_accuracy']:.1%}")
    print(f"Final Confidence: {metadata['final_confidence']:.3f}")
    print(f"Converged:        {'✓' if metadata.get('converged') else '✗'}")
    print(f"Perfect:          {'✓' if metadata.get('perfect') else '✗'}")

    # Compute detailed metrics
    print_section("6. Detailed Metrics per Iteration")

    metrics_calc = IterativeRefinementMetrics(vocab_size=vocab.vocab_size)

    # Recompute metrics for each iteration from history
    iteration_metrics = []
    for i, step in enumerate(metadata['history']):
        # Create predictions dict from history
        predictions = {
            'logits': torch.zeros(refined.x.size(0), vocab.vocab_size),  # Dummy
            'confidence': torch.full((refined.x.size(0),), step['confidence']),
        }

        # We don't have the intermediate graphs, so just display history
        print(f"Iteration {step['iteration']}:")
        print(f"  Accuracy:   {step['accuracy']:.3f}")
        print(f"  Confidence: {step['confidence']:.3f}")
        print(f"  Correct:    {step['num_correct']}/{original_graph.x.size(0)}")
        print(f"  Changed:    {step['num_changed']} nodes")
        print()

        # Rename 'confidence' to 'mean_confidence' for trajectory metrics
        step_metrics = step.copy()
        step_metrics['mean_confidence'] = step_metrics.pop('confidence')
        iteration_metrics.append(step_metrics)

    # Trajectory summary
    print_section("7. Trajectory Summary")
    trajectory_metrics = metrics_calc.compute_trajectory_metrics(iteration_metrics)
    print_trajectory_summary(trajectory_metrics)

    # Final notes
    print_section("Summary")
    print("This demo showcases Phase 1.5 iterative refinement:")
    print("  ✓ Token-level graph representation (6 features)")
    print("  ✓ Iterative Graph U-Net architecture")
    print("  ✓ Multi-objective refinement loss")
    print("  ✓ Early stopping conditions (3 types)")
    print("  ✓ Comprehensive evaluation metrics")
    print()
    print("Note: This model is randomly initialized.")
    print("      After training, accuracy should improve significantly!")
    print()


def demo_masked_generation():
    """Demonstrate generation from fully masked graph."""
    print_section("Bonus: Generation from Fully Masked Graph")

    # Load dataset
    dataset_dir = Path("data/phase1_5/pilot")
    vocab_path = Path("data/phase1_5/vocabulary.json")

    if not dataset_dir.exists() or not vocab_path.exists():
        print("❌ Dataset not found")
        return

    vocab = Vocabulary.load(str(vocab_path))
    sample_files = sorted(dataset_dir.glob("sample_*.pt"))
    samples = [torch.load(f, weights_only=False) for f in sample_files]

    sample = samples[1]  # Use second sample
    original_graph = sample.graph

    print_graph_info(original_graph, vocab, "Target")

    # Create fully masked version
    masked = create_masked_graph(original_graph, mask_token_id=0, full_mask=True)
    print_graph_info(masked, vocab, "Fully Masked")

    # Initialize small model
    model = IterativeGraphUNet(
        vocab_size=vocab.vocab_size,
        hidden_channels=128,
        depth=2,
        max_iterations=5,
    )

    interpreter = MiniLispInterpreter()
    inference = IterativeRefinementInference(
        model=model,
        interpreter=interpreter,
        max_iterations=5,
    )

    # Generate from tests
    print("\nGenerating from masked graph with test cases...\n")

    generated, metadata = inference.generate_from_tests(
        masked,
        sample.tests,
        verbose=True,
    )

    print_graph_info(generated, vocab, "\nGenerated")
    print(f"\nIterations:       {metadata['iterations']}")
    print(f"Final Confidence: {metadata['final_confidence']:.3f}")
    print()


def main():
    """Run all demos."""
    try:
        demo_basic_refinement()
        demo_masked_generation()

        print_section("Demo Complete!")
        print("Next steps:")
        print("  1. Train the model: python scripts/train_phase1_5.py")
        print("  2. Evaluate performance: python scripts/evaluate_phase1_5.py")
        print("  3. Generate dataset: python scripts/generate_phase1_5_dataset.py")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
