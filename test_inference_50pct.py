"""Test iterative refinement at 50% corruption (validation level)."""

import torch
from pathlib import Path

from src.data.vocabulary import Vocabulary
from src.data.dataset import IterativeRefinementDataset
from src.models.graph_unet import IterativeGraphUNet
from src.inference.inference import IterativeRefinementInference
from src.runtime.interpreter import MiniLispInterpreter
from src.data.asg_builder import ASGBuilder


def main():
    # Load vocabulary
    vocab_path = Path("data/phase1_5/vocabulary.json")
    vocab = Vocabulary.load(vocab_path)
    print(f"Vocabulary size: {vocab.vocab_size}")

    # Load validation dataset
    val_dataset = IterativeRefinementDataset(
        data_dir=Path("data/phase1_5/pilot"),
        mask_token_id=vocab.mask_token_id,
    )
    print(f"Validation samples: {len(val_dataset)}")

    # Load model (use CPU to avoid CUDA assertion errors from invalid indices)
    device = torch.device("cpu")
    model = IterativeGraphUNet(
        vocab_size=vocab.vocab_size,
        hidden_channels=256,
        depth=3,
        pool_ratio=0.5,
        max_iterations=5,
    ).to(device)

    # Load best checkpoint
    checkpoint_dir = Path("checkpoints/phase1_5")
    best_checkpoint = checkpoint_dir / "best_model.pt"

    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Best val accuracy: {checkpoint['metrics']['val']['accuracy']:.4f}")
    else:
        print("No checkpoint found!")
        return

    # Create interpreter
    interpreter = MiniLispInterpreter()

    # Create inference engine
    inference = IterativeRefinementInference(
        model=model,
        interpreter=interpreter,
        mask_token_id=vocab.mask_token_id,
        max_iterations=10,
        confidence_threshold=0.95,
    )

    # Test on 10 samples at 50% corruption
    print("\n" + "=" * 70)
    print("Testing Iterative Refinement at 50% Corruption (Validation Level)")
    print("=" * 70)

    perfect_count = 0
    improved_count = 0

    for i in range(min(10, len(val_dataset))):
        _, clean_graph, tests = val_dataset[i]
        clean_graph = clean_graph.to(device)

        # Corrupt at 50% (validation level)
        from src.training.trajectory import corrupt_program_curriculum
        corrupted = corrupt_program_curriculum(
            program=clean_graph,
            epoch=10,  # Epoch 10 = 50% corruption
            total_epochs=50,
            mask_token_id=vocab.mask_token_id,
        ).to(device)

        # Run iterative refinement
        refined_graph, metadata = inference.refine_program(
            initial_graph=corrupted,
            target_graph=clean_graph,
            verbose=False,
        )

        # Compute accuracies
        initial_acc = (corrupted.x[:, 0] == clean_graph.x[:, 0]).float().mean().item()
        final_acc = (refined_graph.x[:, 0] == clean_graph.x[:, 0]).float().mean().item()

        is_perfect = final_acc == 1.0
        is_improved = final_acc > initial_acc

        if is_perfect:
            perfect_count += 1
        if is_improved:
            improved_count += 1

        print(f"\nSample {i + 1}:")
        print(f"  Nodes: {clean_graph.x.size(0)}")
        print(f"  Initial accuracy: {initial_acc * 100:.1f}% ({int(initial_acc * clean_graph.x.size(0))}/{clean_graph.x.size(0)})")
        print(f"  Final accuracy:   {final_acc * 100:.1f}% ({int(final_acc * clean_graph.x.size(0))}/{clean_graph.x.size(0)})")
        print(f"  Iterations: {metadata['iterations']}")
        print(f"  Converged: {'✓' if metadata['converged'] else '✗'}")
        print(f"  Perfect: {'✓' if is_perfect else '✗'}")

        # Show trajectory details
        for step in metadata['history']:
            print(f"  Iteration {step['iteration']}: accuracy={step['accuracy']:.3f}, confidence={step['confidence']:.3f}, correct={step['num_correct']}/{clean_graph.x.size(0)}")

        # Show some predictions for non-perfect cases
        if not is_perfect and i < 3:
            target_tokens = [vocab.decode(int(tid)) for tid in clean_graph.x[:, 0]]
            pred_tokens = [vocab.decode(int(tid)) for tid in refined_graph.x[:, 0]]
            print(f"  Target:  {target_tokens}")
            print(f"  Predicted: {pred_tokens}")

    print("\n" + "=" * 70)
    print("Summary (50% corruption):")
    print(f"  Perfect reconstructions: {perfect_count}/10")
    print(f"  Improved from initial: {improved_count}/10")
    print("=" * 70)


if __name__ == "__main__":
    main()
