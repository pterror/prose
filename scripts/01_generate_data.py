#!/usr/bin/env python3
"""Generate synthetic Mini-Lisp programs for training."""

import argparse
from pathlib import Path

from src.data.synthetic_gen import SyntheticGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic program dataset")
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
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Generate equal samples per template",
    )

    args = parser.parse_args()

    print(f"Generating {args.num_samples} synthetic programs...")
    print(f"Output directory: {args.output}")
    print(f"Random seed: {args.seed}")

    generator = SyntheticGenerator(seed=args.seed)

    if args.balanced:
        samples_per_template = args.num_samples // len(generator.templates)
        print(f"Balanced mode: {samples_per_template} samples per template")
        generator.generate_dataset(
            args.num_samples, args.output, samples_per_template=samples_per_template
        )
    else:
        generator.generate_dataset(args.num_samples, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
