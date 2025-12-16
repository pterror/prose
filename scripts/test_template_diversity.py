#!/usr/bin/env python3
"""Test template diversity by generating samples and measuring uniqueness."""

from pathlib import Path
import torch
import tempfile
import hashlib
from collections import Counter

from src.data.synthetic_gen import SyntheticGenerator


def hash_graph_structure(data):
    """Hash the graph structure (nodes + edges) to identify unique programs."""
    # Create a hashable representation
    node_types = tuple(data.x[:, 0].tolist())  # Just node types (ignoring position encodings)
    edges = tuple(map(tuple, data.edge_index.t().tolist()))
    edge_types = tuple(data.edge_attr.tolist())

    structure_str = f"{node_types}|{edges}|{edge_types}"
    return hashlib.md5(structure_str.encode()).hexdigest()


def test_diversity(num_samples=100):
    """Generate samples and measure uniqueness."""
    print("=" * 60)
    print("Testing Template Diversity")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        generator = SyntheticGenerator(seed=42)
        output_dir = Path(tmpdir)

        print(f"\nGenerating {num_samples} samples...")
        generator.generate_dataset(num_samples, output_dir)

        # Hash all structures
        hashes = []
        pt_files = sorted(list(output_dir.glob("*.pt")))

        for pt_file in pt_files:
            data = torch.load(pt_file, weights_only=False)
            h = hash_graph_structure(data)
            hashes.append(h)

        # Count unique
        unique_hashes = len(set(hashes))
        duplicates = num_samples - unique_hashes
        duplication_rate = duplicates / num_samples * 100

        # Most common duplicates
        hash_counts = Counter(hashes)
        most_common = hash_counts.most_common(5)

        print(f"\n=== Diversity Results ===")
        print(f"Total samples: {num_samples}")
        print(f"Unique structures: {unique_hashes}")
        print(f"Duplicates: {duplicates}")
        print(f"Duplication rate: {duplication_rate:.1f}%")
        print(f"Uniqueness: {(100 - duplication_rate):.1f}%")

        print(f"\nMost duplicated structures:")
        for h, count in most_common:
            if count > 1:
                print(f"  Hash {h[:8]}...: appeared {count} times")

        # Success criteria
        print(f"\n=== Assessment ===")
        if duplication_rate <= 10:
            print(f"✅ EXCELLENT: Duplication rate {duplication_rate:.1f}% ≤ 10% target")
        elif duplication_rate <= 20:
            print(f"✓ GOOD: Duplication rate {duplication_rate:.1f}% ≤ 20%")
        else:
            print(f"⚠️  NEEDS IMPROVEMENT: Duplication rate {duplication_rate:.1f}% > 20%")

        return duplication_rate


if __name__ == "__main__":
    duplication_rate = test_diversity(100)

    print("\n" + "=" * 60)
    if duplication_rate <= 10:
        print("Template expansion successful! ✓")
    else:
        print("May need more template variation")
    print("=" * 60)
