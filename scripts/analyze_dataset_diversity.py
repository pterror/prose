"""Analyze dataset diversity and detect duplicates."""

import torch
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
from tqdm import tqdm


def hash_graph(graph):
    """Create hash of graph structure and tokens."""
    # Hash based on tokens and edges
    tokens = graph.x[:, 0].long().cpu().numpy().tobytes()
    edges = graph.edge_index.cpu().numpy().tobytes()
    combined = tokens + edges
    return hashlib.sha256(combined).hexdigest()


def analyze_dataset(data_dir: Path):
    """Analyze dataset for duplicates and diversity."""
    print(f"Analyzing dataset: {data_dir}")
    print("=" * 70)

    # Load all samples
    sample_files = sorted(data_dir.glob("sample_*.pt"))
    print(f"Found {len(sample_files)} samples\n")

    if len(sample_files) == 0:
        print("No samples found!")
        return

    # Track statistics
    graph_hashes = []
    num_nodes_list = []
    num_edges_list = []
    templates = []
    token_sequences = []

    print("Loading samples...")
    for sample_path in tqdm(sample_files):
        try:
            sample = torch.load(sample_path, weights_only=False)
            graph = sample.graph

            # Hash the graph
            graph_hash = hash_graph(graph)
            graph_hashes.append(graph_hash)

            # Collect statistics
            num_nodes_list.append(graph.x.size(0))
            num_edges_list.append(graph.edge_index.size(1))

            # Template info
            if hasattr(sample, 'metadata') and sample.metadata:
                template = sample.metadata.get('template', 'unknown')
                templates.append(template)

            # Token sequence (for diversity)
            tokens = tuple(graph.x[:, 0].long().cpu().numpy().tolist())
            token_sequences.append(tokens)

        except Exception as e:
            print(f"Warning: Failed to load {sample_path}: {e}")
            continue

    print("\n" + "=" * 70)
    print("DUPLICATION ANALYSIS")
    print("=" * 70)

    # Check for exact duplicates (by graph hash)
    hash_counts = Counter(graph_hashes)
    num_unique = len(hash_counts)
    num_duplicates = len(graph_hashes) - num_unique

    print(f"\nExact graph duplicates:")
    print(f"  Total samples: {len(graph_hashes)}")
    print(f"  Unique graphs: {num_unique}")
    print(f"  Duplicates: {num_duplicates}")

    if len(graph_hashes) > 0:
        print(f"  Duplication rate: {num_duplicates / len(graph_hashes) * 100:.1f}%")
    else:
        print(f"  Duplication rate: N/A (no samples loaded)")

    if num_duplicates > 0:
        print(f"\n  Top duplicated graphs:")
        for hash_val, count in hash_counts.most_common(10):
            if count > 1:
                print(f"    {hash_val[:16]}... appears {count} times")

    # Check token sequence diversity
    token_counts = Counter(token_sequences)
    num_unique_sequences = len(token_counts)

    print(f"\nToken sequence diversity:")
    print(f"  Unique token sequences: {num_unique_sequences}")
    print(f"  Sequence duplication rate: {(len(token_sequences) - num_unique_sequences) / len(token_sequences) * 100:.1f}%")

    print("\n" + "=" * 70)
    print("STRUCTURAL DIVERSITY")
    print("=" * 70)

    # Graph size distribution
    import numpy as np

    print(f"\nGraph sizes:")
    print(f"  Nodes: min={min(num_nodes_list)}, max={max(num_nodes_list)}, mean={np.mean(num_nodes_list):.1f}, std={np.std(num_nodes_list):.1f}")
    print(f"  Edges: min={min(num_edges_list)}, max={max(num_edges_list)}, mean={np.mean(num_edges_list):.1f}, std={np.std(num_edges_list):.1f}")

    # Node count distribution
    node_count_dist = Counter(num_nodes_list)
    print(f"\n  Node count distribution (top 10):")
    for node_count, freq in node_count_dist.most_common(10):
        print(f"    {node_count} nodes: {freq} samples ({freq/len(num_nodes_list)*100:.1f}%)")

    # Template distribution
    if templates:
        template_counts = Counter(templates)
        print(f"\nTemplate distribution:")
        print(f"  Unique templates: {len(template_counts)}")
        for template, count in template_counts.most_common():
            print(f"    {template}: {count} samples ({count/len(templates)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("OVERFITTING RISK ASSESSMENT")
    print("=" * 70)

    if len(graph_hashes) == 0:
        print("\n❌ ERROR: No samples loaded successfully")
        return {}

    duplication_rate = num_duplicates / len(graph_hashes)

    if duplication_rate < 0.05:
        risk = "✅ LOW"
        print(f"\n{risk}: Duplication rate < 5%")
        print("  Dataset has good diversity. Low overfitting risk.")
    elif duplication_rate < 0.20:
        risk = "⚠️  MODERATE"
        print(f"\n{risk}: Duplication rate 5-20%")
        print("  Some duplicates present. Monitor validation performance.")
    else:
        risk = "❌ HIGH"
        print(f"\n{risk}: Duplication rate > 20%")
        print("  Significant duplication detected. High overfitting risk!")
        print("  Recommendation: Increase template diversity or use data augmentation")

    # Structural diversity assessment
    node_std = np.std(num_nodes_list)
    if node_std < 2:
        print(f"\n⚠️  WARNING: Low structural diversity (node std={node_std:.1f})")
        print("  Most graphs have similar sizes. Consider more diverse templates.")
    else:
        print(f"\n✅ Good structural diversity (node std={node_std:.1f})")

    return {
        'total_samples': len(graph_hashes),
        'unique_graphs': num_unique,
        'duplication_rate': duplication_rate,
        'num_templates': len(template_counts) if templates else 0,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze dataset diversity")
    parser.add_argument("data_dir", type=Path, help="Dataset directory")
    args = parser.parse_args()

    analyze_dataset(args.data_dir)


if __name__ == "__main__":
    main()
