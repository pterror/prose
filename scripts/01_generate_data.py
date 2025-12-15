#!/usr/bin/env python3
"""Generate synthetic Mini-Lisp programs for training."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_gen import SyntheticGenerator


def generate_split(
    split_name: str, num_samples: int, base_output: Path, seed: int, balanced: bool
) -> None:
    """Generate a single dataset split."""
    output_dir = base_output / split_name
    print(f"\n{'=' * 70}")
    print(f"Generating {split_name.upper()} split")
    print(f"{'=' * 70}")
    print(f"  Samples: {num_samples}")
    print(f"  Output: {output_dir}")
    print(f"  Balanced: {balanced}")
    print(f"  Seed: {seed}")

    generator = SyntheticGenerator(seed=seed)

    if balanced:
        samples_per_template = num_samples // len(generator.templates)
        print(f"  Samples per template: {samples_per_template}")
    else:
        samples_per_template = None

    generator.generate_dataset(num_samples, output_dir, samples_per_template)
    print(f"✓ {split_name.upper()} split complete!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic program dataset with train/val/test splits"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of programs to generate (default: 1000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Base output directory (default: data/processed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Generate equal samples per template",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        help="Generate specific split (train/val/test). If not specified, generates to output dir directly.",
    )
    parser.add_argument(
        "--generate-all",
        action="store_true",
        help="Generate all splits: 2K train, 500 val, 500 test (ignores --num-samples)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Synthetic Program Dataset Generator")
    print("=" * 70)

    if args.generate_all:
        # Generate all splits with recommended sizes
        print("\nGenerating complete dataset with all splits...")
        generate_split("train", 2000, args.output, args.seed, balanced=True)
        generate_split("val", 500, args.output, args.seed + 1, balanced=True)
        generate_split("test", 500, args.output, args.seed + 2, balanced=True)

        print(f"\n{'=' * 70}")
        print("ALL SPLITS GENERATED SUCCESSFULLY! ✓")
        print(f"{'=' * 70}")
        print(f"\nDataset location: {args.output.resolve()}")
        print(f"  • train/: 2000 samples (4000 files)")
        print(f"  • val/:   500 samples (1000 files)")
        print(f"  • test/:  500 samples (1000 files)")
        print(f"\nTotal: 3000 samples (~150MB)")

    elif args.split:
        # Generate specific split
        generate_split(args.split, args.num_samples, args.output, args.seed, args.balanced)

    else:
        # Legacy mode: generate to output directory directly
        print(f"\nGenerating {args.num_samples} programs...")
        print(f"Output directory: {args.output}")
        print(f"Seed: {args.seed}")
        print(f"Balanced: {args.balanced}")

        generator = SyntheticGenerator(seed=args.seed)

        if args.balanced:
            samples_per_template = args.num_samples // len(generator.templates)
            print(f"Samples per template: {samples_per_template}")
            generator.generate_dataset(
                args.num_samples, args.output, samples_per_template=samples_per_template
            )
        else:
            generator.generate_dataset(args.num_samples, args.output)

        print("\n✓ Done!")


if __name__ == "__main__":
    main()
