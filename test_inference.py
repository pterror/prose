"""Test iterative refinement with trained model."""

import torch
from pathlib import Path

from src.data.dataset import IterativeRefinementDataset
from src.data.vocabulary import Vocabulary
from src.models.graph_unet import IterativeGraphUNet
from src.inference.inference import IterativeRefinementInference
from src.training.denoising_metrics import IterativeRefinementMetrics

# Load vocabulary
vocab = Vocabulary.load(Path("data/phase1_5/vocabulary.json"))
print(f"Vocabulary size: {vocab.vocab_size}")

# Load validation dataset
val_dataset = IterativeRefinementDataset(
    data_dir=Path("data/phase1_5/pilot"),
    mask_token_id=vocab.mask_token_id,
)
print(f"Validation samples: {len(val_dataset)}")

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IterativeGraphUNet(
    vocab_size=vocab.vocab_size,
    hidden_channels=256,
    depth=3,
    pool_ratio=0.5,
    max_iterations=5,
).to(device)

checkpoint = torch.load("checkpoints/phase1_5/best_model.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Best val accuracy: {checkpoint['metrics']['val']['accuracy']:.4f}")

# Create inference engine
from src.runtime.interpreter import MiniLispInterpreter
interpreter = MiniLispInterpreter()

inference = IterativeRefinementInference(
    model=model,
    interpreter=interpreter,
    mask_token_id=vocab.mask_token_id,
    max_iterations=10,  # Allow more iterations
)
model.to(device)

# Test on a few samples
print("\n" + "="*70)
print("Testing Iterative Refinement")
print("="*70)

total_perfect = 0
total_improved = 0

for i in range(min(10, len(val_dataset))):
    _, clean_graph, tests = val_dataset[i]
    clean_graph = clean_graph.to(device)

    # Start with fully masked graph
    from src.inference.inference import create_masked_graph
    masked_graph = create_masked_graph(clean_graph, vocab.mask_token_id).to(device)

    # Run iterative refinement
    final_graph, metadata = inference.refine_program(
        initial_graph=masked_graph,
        target_graph=clean_graph,
        verbose=True,
    )

    # Check accuracy
    final_tokens = final_graph.x[:, 0].cpu()
    target_tokens = clean_graph.x[:, 0].cpu()
    correct = (final_tokens == target_tokens).sum().item()
    total = len(target_tokens)
    accuracy = correct / total

    initial_correct = (masked_graph.x[:, 0].cpu() == target_tokens).sum().item()
    initial_acc = initial_correct / total

    perfect = (accuracy == 1.0)
    improved = (accuracy > initial_acc)

    total_perfect += perfect
    total_improved += improved

    print(f"\nSample {i+1}:")
    print(f"  Nodes: {total}")
    print(f"  Initial accuracy: {initial_acc:.1%} ({initial_correct}/{total})")
    print(f"  Final accuracy:   {accuracy:.1%} ({correct}/{total})")
    print(f"  Iterations: {metadata['iterations']}")
    print(f"  Converged: {'✓' if metadata['converged'] else '✗'}")
    print(f"  Perfect: {'✓' if metadata.get('perfect', False) else '✗'}")

    # Decode a few tokens for inspection
    if total <= 10:
        print(f"  Target:  {[vocab.decode(int(t)) for t in target_tokens]}")
        print(f"  Predicted: {[vocab.decode(int(t)) for t in final_tokens]}")

print("\n" + "="*70)
print(f"Summary:")
print(f"  Perfect reconstructions: {total_perfect}/10")
print(f"  Improved from initial: {total_improved}/10")
print("="*70)
