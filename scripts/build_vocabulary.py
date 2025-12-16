#!/usr/bin/env python3
"""Build vocabulary from all Mini-Lisp templates and save to file."""

import argparse
import random
from pathlib import Path

from src.data.synthetic_gen import SyntheticGenerator
from src.data.vocabulary import Vocabulary


def main():
    parser = argparse.ArgumentParser(description="Build vocabulary from templates")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/vocabulary.json"),
        help="Output path for vocabulary file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate per template",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Building Mini-Lisp Vocabulary")
    print("=" * 60)

    # Create synthetic generator
    print(f"\n1. Initializing synthetic generator (seed={args.seed})...")
    generator = SyntheticGenerator(seed=args.seed)
    print(f"   Found {len(generator.templates)} template types")

    # Create vocabulary
    print("\n2. Building vocabulary from templates...")
    vocab = Vocabulary()
    vocab.build_from_templates(generator.templates, num_samples=args.num_samples)

    # Print statistics
    print("\n3. Vocabulary Statistics:")
    stats = vocab.get_stats()
    print(f"   Total vocabulary size: {stats['vocab_size']}")
    print(f"   Special tokens: {stats['num_special_tokens']}")
    print(f"   Regular tokens: {stats['num_regular_tokens']}")

    print(f"\n   Most common tokens:")
    for token, count in stats["most_common"][:15]:
        print(f"      '{token}': {count}")

    # Save vocabulary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n4. Saving vocabulary to {args.output}...")
    vocab.save(args.output)

    # Verify save/load
    print("\n5. Verifying save/load...")
    loaded_vocab = Vocabulary.load(args.output)
    assert loaded_vocab.vocab_size == vocab.vocab_size
    print("   ✓ Vocabulary saved and loaded successfully")

    print("\n" + "=" * 60)
    print(f"✓ Vocabulary built successfully!")
    print(f"  Vocabulary size: {vocab.vocab_size} tokens")
    print(f"  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
