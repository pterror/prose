"""Test iterative refinement with real test execution."""

import torch
from pathlib import Path

from src.data.vocabulary import Vocabulary
from src.data.dataset import IterativeRefinementDataset
from src.models.graph_unet import IterativeGraphUNet
from src.inference.inference import IterativeRefinementInference
from src.runtime.interpreter import MiniLispInterpreter


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

    # Load model
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
        print(f"Best val accuracy: {checkpoint['metrics']['val']['accuracy']:.4f}\n")
    else:
        print("No checkpoint found!")
        return

    # Create interpreter
    interpreter = MiniLispInterpreter()

    # Create inference engine WITH test execution
    inference = IterativeRefinementInference(
        model=model,
        interpreter=interpreter,
        vocabulary=vocab,
        mask_token_id=vocab.mask_token_id,
        max_iterations=5,  # Match training max_iterations
        confidence_threshold=0.95,
        use_test_execution=True,  # Enable test execution!
    )

    # Test on samples at different corruption levels
    corruption_levels = [0.5, 0.75, 1.0]

    for corruption_rate in corruption_levels:
        print("\n" + "=" * 70)
        print(f"Testing at {int(corruption_rate * 100)}% Corruption with Test Execution")
        print("=" * 70)

        perfect_count = 0
        tests_all_passed_count = 0
        improved_count = 0

        for i in range(min(5, len(val_dataset))):
            _, clean_graph, tests = val_dataset[i]
            clean_graph = clean_graph.to(device)

            # Corrupt at specified rate
            from src.training.trajectory import corrupt_program_curriculum

            epoch_map = {0.5: 10, 0.75: 20, 1.0: 45}
            epoch = epoch_map.get(corruption_rate, 10)
            corrupted = corrupt_program_curriculum(
                program=clean_graph,
                epoch=epoch,
                total_epochs=50,
                mask_token_id=vocab.mask_token_id,
            ).to(device)

            # Run iterative refinement WITH test execution
            refined_graph, metadata = inference.generate_from_tests(
                initial_graph=corrupted,
                tests=tests,
                verbose=(i == 0),  # Verbose for first sample
            )

            # Compute accuracies
            initial_acc = (corrupted.x[:, 0] == clean_graph.x[:, 0]).float().mean().item()
            final_acc = (refined_graph.x[:, 0] == clean_graph.x[:, 0]).float().mean().item()

            is_perfect = final_acc == 1.0
            is_improved = final_acc > initial_acc
            all_tests_passed = metadata['history'][-1]['all_tests_passed'] if metadata['history'] else False

            if is_perfect:
                perfect_count += 1
            if all_tests_passed:
                tests_all_passed_count += 1
            if is_improved:
                improved_count += 1

            print(f"\nSample {i + 1}:")
            print(f"  Nodes: {clean_graph.x.size(0)}")
            print(f"  Initial accuracy: {initial_acc * 100:.1f}%")
            print(f"  Final accuracy:   {final_acc * 100:.1f}%")
            print(f"  Iterations: {metadata['iterations']}")
            print(f"  Tests passed: {metadata['history'][-1]['tests_passed']}/{metadata['history'][-1]['tests_total']}")
            print(f"  All tests passed: {'✓' if all_tests_passed else '✗'}")
            print(f"  Perfect match: {'✓' if is_perfect else '✗'}")
            print(f"  Improved: {'✓' if is_improved else '✗'}")

        print(f"\n{'-' * 70}")
        print(f"Summary at {int(corruption_rate * 100)}% corruption:")
        print(f"  Perfect reconstructions: {perfect_count}/5")
        print(f"  All tests passing: {tests_all_passed_count}/5")
        print(f"  Improved from initial: {improved_count}/5")
        print(f"{'-' * 70}")


if __name__ == "__main__":
    main()
