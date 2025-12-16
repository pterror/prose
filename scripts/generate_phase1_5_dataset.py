#!/usr/bin/env python3
"""Generate Phase 1.5 dataset with vocabulary, tokens, and test cases."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.asg_builder import ASGBuilder
from src.data.dataset import ProgramSample, TestCase
from src.data.synthetic_gen import SyntheticGenerator
from src.data.vocabulary import Vocabulary
from src.runtime.interpreter import MiniLispInterpreter


def generate_test_cases(ast, interpreter, num_tests=3):
    """Generate test cases for a program by executing it."""
    import random

    test_cases = []

    # Try to extract function definition
    # For now, generate simple tests based on program structure
    # This is a simplified version - in practice, templates should provide tests

    for i in range(num_tests):
        # For simple programs, use sequential inputs
        inputs = [i + 1]

        try:
            # Execute program
            result = interpreter.eval(ast, interpreter.create_env())

            # If it's a function, try calling it
            if callable(result):
                try:
                    output = result(*inputs)
                    test_cases.append(TestCase(
                        inputs=inputs,
                        expected_output=output
                    ))
                except:
                    pass
            else:
                # For non-functions, just use the result
                test_cases.append(TestCase(
                    inputs=[],
                    expected_output=result
                ))
        except Exception as e:
            # Skip programs that don't execute
            pass

    # If no tests generated, create at least one default test
    if not test_cases:
        test_cases.append(TestCase(
            inputs=[],
            expected_output=None
        ))

    return test_cases


def generate_dataset(
    num_samples: int,
    output_dir: Path,
    vocab_path: Path,
    seed: int = 42,
):
    """Generate Phase 1.5 dataset."""
    import random

    print(f"Generating Phase 1.5 dataset...")
    print(f"  Samples: {num_samples}")
    print(f"  Output: {output_dir}")
    print(f"  Vocabulary: {vocab_path}")
    print(f"  Seed: {seed}")

    # Initialize generator
    generator = SyntheticGenerator(seed=seed)

    # Load or create vocabulary
    if vocab_path.exists():
        print(f"\nLoading vocabulary from {vocab_path}...")
        vocabulary = Vocabulary.load(vocab_path)
        print(f"  Vocabulary size: {vocabulary.vocab_size}")
    else:
        print(f"\nBuilding vocabulary...")
        vocabulary = Vocabulary()
        vocabulary.build_from_templates(generator.templates, num_samples=1000)

        # Save vocabulary
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocabulary.save(vocab_path)
        print(f"  Vocabulary size: {vocabulary.vocab_size}")
        print(f"  Saved to: {vocab_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    builder = ASGBuilder(vocabulary=vocabulary)
    interpreter = MiniLispInterpreter()
    rng = random.Random(seed)

    # Generate samples
    print(f"\nGenerating samples...")
    samples_per_template = max(1, num_samples // len(generator.templates))

    sample_idx = 0
    for template_idx, template in enumerate(generator.templates):
        template_name = template.__class__.__name__

        for i in range(samples_per_template):
            if sample_idx >= num_samples:
                break

            try:
                # Generate program
                ast, metadata = template.generate(rng)

                # Build graph with vocabulary
                graph = builder.build(ast)

                # Generate test cases
                tests = generate_test_cases(ast, interpreter)

                # Create sample
                # Convert metadata dataclass to dict if needed
                metadata_dict = {
                    'template': template_name,
                    'source_code': metadata.source_code if hasattr(metadata, 'source_code') else None,
                    'category': metadata.category if hasattr(metadata, 'category') else None,
                }

                sample = ProgramSample(
                    graph=graph,
                    tests=tests,
                    metadata=metadata_dict
                )

                # Save sample
                output_path = output_dir / f"sample_{sample_idx:05d}.pt"
                torch.save(sample, output_path)

                sample_idx += 1

                if sample_idx % 10 == 0:
                    print(f"  Generated {sample_idx}/{num_samples} samples...")

            except Exception as e:
                print(f"  Warning: Failed to generate sample from {template_name}: {e}")
                continue

    print(f"\n✓ Generated {sample_idx} samples successfully!")
    print(f"  Location: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 1.5 dataset with vocabulary and test cases"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phase1_5/pilot"),
        help="Output directory (default: data/phase1_5/pilot)",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("data/phase1_5/vocabulary.json"),
        help="Vocabulary file path (default: data/phase1_5/vocabulary.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Phase 1.5 Dataset Generator")
    print("=" * 70)

    generate_dataset(
        num_samples=args.num_samples,
        output_dir=args.output,
        vocab_path=args.vocab,
        seed=args.seed,
    )

    print(f"\n{'=' * 70}")
    print("DATASET GENERATED SUCCESSFULLY! ✓")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
