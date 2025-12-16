#!/usr/bin/env python3
"""Verify position encodings in generated dataset and test model forward pass."""

from pathlib import Path
import torch
from src.models.graph_unet import GraphUNet


def main():
    print("=" * 60)
    print("Verifying Position Encodings")
    print("=" * 60)

    # Load a sample from test dataset
    test_dir = Path("data/test_position_encodings")
    sample_files = sorted(list(test_dir.glob("*.pt")))

    if not sample_files:
        print("❌ No sample files found!")
        return

    print(f"\nLoading sample: {sample_files[0].name}")
    data = torch.load(sample_files[0], weights_only=False)

    print(f"\nNode features shape: {data.x.shape}")
    print(f"Expected shape: [num_nodes, 3] (node_type, depth, sibling_index)")

    assert data.x.ndim == 2, "Node features should be 2D"
    assert data.x.shape[1] == 3, "Node features should have 3 columns"

    print(f"✓ Node features have correct shape: {data.x.shape}")

    # Display first few nodes
    print(f"\nFirst 5 nodes [node_type, depth, sibling_index]:")
    print(data.x[:5])

    # Verify depth and sibling_index are valid
    depths = data.x[:, 1]
    sibling_indices = data.x[:, 2]

    print(f"\nDepth range: [{depths.min().item()}, {depths.max().item()}]")
    print(f"Sibling index range: [{sibling_indices.min().item()}, {sibling_indices.max().item()}]")

    assert depths.min() >= 0, "Depths should be >= 0"
    assert sibling_indices.min() >= 0, "Sibling indices should be >= 0"

    print("✓ Position encodings are valid")

    # Test model forward pass
    print("\n" + "=" * 60)
    print("Testing Model Forward Pass")
    print("=" * 60)

    model = GraphUNet(
        in_channels=128,
        hidden_channels=256,
        out_channels=128,
        depth=3,
        pool_ratio=0.5,
        num_node_types=9,
        layer_type="GAT",
    )

    print(f"\nModel created successfully")
    print(f"Effective input channels: {model.effective_in_channels}")

    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            output = model.forward_full(data)
            print(f"✓ Forward pass successful!")
            print(f"  Input nodes: {data.x.shape[0]}")
            print(f"  Output shape: {output.shape}")
            print(f"  Expected: [{data.x.shape[0]}, 9] (num_nodes, num_node_types)")

            assert output.shape == (data.x.shape[0], 9), "Output shape mismatch"
            print("✓ Output shape is correct")

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            raise

    print("\n" + "=" * 60)
    print("All Checks Passed! ✓")
    print("=" * 60)
    print("\nPosition encodings are correctly implemented.")
    print("The model can process 3D node features.")


if __name__ == "__main__":
    main()
