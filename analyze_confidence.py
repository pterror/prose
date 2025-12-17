"""Analyze confidence distribution of trained model."""

import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.data.vocabulary import Vocabulary
from src.data.dataset import IterativeRefinementDataset
from src.models.graph_unet import IterativeGraphUNet


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
    else:
        print("No checkpoint found!")
        return

    model.eval()

    # Analyze confidence at different corruption levels
    corruption_levels = [0.0, 0.2, 0.5, 0.75, 1.0]

    for corruption_rate in corruption_levels:
        print(f"\n{'='*70}")
        print(f"Corruption rate: {corruption_rate*100:.0f}%")
        print(f"{'='*70}")

        correct_confidences = []
        incorrect_confidences = []

        for i in range(len(val_dataset)):
            _, clean_graph, tests = val_dataset[i]
            clean_graph = clean_graph.to(device)

            # Corrupt at specified rate
            from src.training.trajectory import corrupt_program_curriculum

            if corruption_rate == 0.0:
                corrupted = clean_graph.clone()
            else:
                # Map corruption rate to epoch
                epoch_map = {0.2: 3, 0.5: 10, 0.75: 20, 1.0: 45}
                epoch = epoch_map.get(corruption_rate, 10)
                corrupted = corrupt_program_curriculum(
                    program=clean_graph,
                    epoch=epoch,
                    total_epochs=50,
                    mask_token_id=vocab.mask_token_id,
                ).to(device)

            # Forward pass
            with torch.no_grad():
                output = model.forward_full(data=corrupted, iteration=0)

            predictions = output["logits"].argmax(dim=-1)
            confidences = output["confidence"]
            targets = clean_graph.x[:, 0].long()

            # Separate correct vs incorrect predictions
            is_correct = (predictions == targets)

            for conf, correct in zip(confidences, is_correct):
                if correct.item():
                    correct_confidences.append(conf.item())
                else:
                    incorrect_confidences.append(conf.item())

        # Print statistics
        if correct_confidences:
            print(f"Correct predictions: {len(correct_confidences)}")
            print(f"  Mean confidence: {np.mean(correct_confidences):.3f}")
            print(f"  Median confidence: {np.median(correct_confidences):.3f}")
            print(f"  Std confidence: {np.std(correct_confidences):.3f}")

        if incorrect_confidences:
            print(f"\nIncorrect predictions: {len(incorrect_confidences)}")
            print(f"  Mean confidence: {np.mean(incorrect_confidences):.3f}")
            print(f"  Median confidence: {np.median(incorrect_confidences):.3f}")
            print(f"  Std confidence: {np.std(incorrect_confidences):.3f}")

        total = len(correct_confidences) + len(incorrect_confidences)
        accuracy = len(correct_confidences) / total if total > 0 else 0
        print(f"\nAccuracy: {accuracy:.3f} ({len(correct_confidences)}/{total})")

        # Check calibration: how many incorrect predictions have >0.9 confidence?
        if incorrect_confidences:
            high_conf_incorrect = sum(1 for c in incorrect_confidences if c > 0.9)
            print(f"Overconfident errors (>0.9 conf): {high_conf_incorrect}/{len(incorrect_confidences)} ({100*high_conf_incorrect/len(incorrect_confidences):.1f}%)")


if __name__ == "__main__":
    main()
