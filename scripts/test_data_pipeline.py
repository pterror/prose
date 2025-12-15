#!/usr/bin/env python3
"""Test data generation and ASG building."""

from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.asg_builder import ASTNode, ASGBuilder, NodeType
from src.data.synthetic_gen import SyntheticGenerator


def test_asg_builder() -> None:
    """Test ASG builder with a simple program."""
    print("Testing ASG builder...")

    # Create a simple program: (+ 1 2)
    ast = ASTNode(
        NodeType.LIST,
        children=[
            ASTNode(NodeType.OPERATOR, value="+"),
            ASTNode(NodeType.NUMBER, value=1),
            ASTNode(NodeType.NUMBER, value=2),
        ],
    )

    builder = ASGBuilder()
    asg_data = builder.build(ast)

    print(f"  Nodes: {asg_data.x.shape[0]}")
    print(f"  Edges: {asg_data.edge_index.shape[1]}")
    print(f"  Node types: {asg_data.x.tolist()}")
    print("  ✓ ASG builder works!")


def test_synthetic_generator() -> None:
    """Test synthetic program generation."""
    print("\nTesting synthetic generator...")

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SyntheticGenerator(seed=42)
        output_dir = Path(tmpdir)

        # Generate 10 samples
        generator.generate_dataset(10, output_dir, samples_per_template=None)

        # Check files were created
        pt_files = list(output_dir.glob("*.pt"))
        json_files = list(output_dir.glob("*.json"))

        print(f"  Generated {len(pt_files)} .pt files")
        print(f"  Generated {len(json_files)} .json files")

        assert len(pt_files) == 10
        assert len(json_files) == 10

        print("  ✓ Synthetic generator works!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running data pipeline tests...")
    print("=" * 60)

    test_asg_builder()
    test_synthetic_generator()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
