"""Synthetic Mini-Lisp program generator with property-based constraints."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data.asg_builder import ASTNode, ASGBuilder, NodeType


@dataclass
class ProgramMetadata:
    """Metadata about a generated program."""

    template_id: str
    depth: int
    num_nodes: int
    num_operators: int
    has_recursion: bool
    has_variables: bool
    operator_types: list[str]


class ProgramTemplate:
    """Base class for Mini-Lisp program templates."""

    def __init__(self, template_id: str, description: str) -> None:
        self.template_id = template_id
        self.description = description

    def generate(self, rng: random.Random) -> tuple[ASTNode, ProgramMetadata]:
        """Generate a program instance from this template."""
        raise NotImplementedError


class ArithmeticTemplate(ProgramTemplate):
    """Template for arithmetic expressions: (+ (* a b) (- c d))"""

    def generate(self, rng: random.Random) -> tuple[ASTNode, ProgramMetadata]:
        ops = ["+", "-", "*", "/"]
        op1 = rng.choice(ops)
        op2 = rng.choice(ops)
        op3 = rng.choice(ops)

        # (op1 (op2 a b) (op3 c d))
        root = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value=op1),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=op2),
                        ASTNode(NodeType.SYMBOL, value="a"),
                        ASTNode(NodeType.SYMBOL, value="b"),
                    ],
                ),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=op3),
                        ASTNode(NodeType.SYMBOL, value="c"),
                        ASTNode(NodeType.SYMBOL, value="d"),
                    ],
                ),
            ],
        )

        return root, ProgramMetadata(
            template_id=self.template_id,
            depth=2,
            num_nodes=11,
            num_operators=3,
            has_recursion=False,
            has_variables=True,
            operator_types=[op1, op2, op3],
        )


class RecursionTemplate(ProgramTemplate):
    """Template for recursive functions: (define (fact n) (if (= n 0) 1 ...))"""

    def generate(self, rng: random.Random) -> tuple[ASTNode, ProgramMetadata]:
        func_name = rng.choice(["fact", "fib", "sum"])
        param = "n"
        base_val = rng.randint(0, 2)

        # (define (func_name param) (if (= param base_val) 1 (* param (func_name (- param 1)))))
        root = ASTNode(
            NodeType.DEFINE,
            value=func_name,
            children=[
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.SYMBOL, value=func_name),
                        ASTNode(NodeType.SYMBOL, value=param),
                    ],
                ),
                ASTNode(
                    NodeType.IF,
                    children=[
                        # Condition: (= param base_val)
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.OPERATOR, value="="),
                                ASTNode(NodeType.SYMBOL, value=param),
                                ASTNode(NodeType.NUMBER, value=base_val),
                            ],
                        ),
                        # Then: 1
                        ASTNode(NodeType.NUMBER, value=1),
                        # Else: (* param (func_name (- param 1)))
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.OPERATOR, value="*"),
                                ASTNode(NodeType.SYMBOL, value=param),
                                ASTNode(
                                    NodeType.LIST,
                                    children=[
                                        ASTNode(NodeType.SYMBOL, value=func_name),
                                        ASTNode(
                                            NodeType.LIST,
                                            children=[
                                                ASTNode(NodeType.OPERATOR, value="-"),
                                                ASTNode(NodeType.SYMBOL, value=param),
                                                ASTNode(NodeType.NUMBER, value=1),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )

        return root, ProgramMetadata(
            template_id=self.template_id,
            depth=5,
            num_nodes=19,
            num_operators=3,
            has_recursion=True,
            has_variables=True,
            operator_types=["=", "*", "-"],
        )


class LambdaTemplate(ProgramTemplate):
    """Template for higher-order functions: (map (lambda (x) (* x 2)) list)"""

    def generate(self, rng: random.Random) -> tuple[ASTNode, ProgramMetadata]:
        hof = rng.choice(["map", "filter", "reduce"])
        param = "x"
        op = rng.choice(["+", "*", "-"])
        const = rng.randint(1, 10)

        # (hof (lambda (param) (op param const)) list)
        root = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.SYMBOL, value=hof),
                ASTNode(
                    NodeType.LAMBDA,
                    children=[
                        ASTNode(NodeType.LIST, children=[ASTNode(NodeType.SYMBOL, value=param)]),
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.OPERATOR, value=op),
                                ASTNode(NodeType.SYMBOL, value=param),
                                ASTNode(NodeType.NUMBER, value=const),
                            ],
                        ),
                    ],
                ),
                ASTNode(NodeType.SYMBOL, value="list"),
            ],
        )

        return root, ProgramMetadata(
            template_id=self.template_id,
            depth=3,
            num_nodes=11,
            num_operators=1,
            has_recursion=False,
            has_variables=True,
            operator_types=[op],
        )


class LetBindingTemplate(ProgramTemplate):
    """Template for variable scoping: (let ((x 1) (y 2)) (+ x y))"""

    def generate(self, rng: random.Random) -> tuple[ASTNode, ProgramMetadata]:
        var1, var2 = "x", "y"
        val1 = rng.randint(1, 10)
        val2 = rng.randint(1, 10)
        op = rng.choice(["+", "-", "*"])

        # (let ((var1 val1) (var2 val2)) (op var1 var2))
        root = ASTNode(
            NodeType.LET,
            children=[
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.SYMBOL, value=var1),
                                ASTNode(NodeType.NUMBER, value=val1),
                            ],
                        ),
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.SYMBOL, value=var2),
                                ASTNode(NodeType.NUMBER, value=val2),
                            ],
                        ),
                    ],
                ),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=op),
                        ASTNode(NodeType.SYMBOL, value=var1),
                        ASTNode(NodeType.SYMBOL, value=var2),
                    ],
                ),
            ],
        )

        return root, ProgramMetadata(
            template_id=self.template_id,
            depth=3,
            num_nodes=13,
            num_operators=1,
            has_recursion=False,
            has_variables=True,
            operator_types=[op],
        )


class SyntheticGenerator:
    """Generates synthetic Mini-Lisp programs with property-based diversity."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.templates = [
            ArithmeticTemplate("arithmetic", "Basic arithmetic expressions"),
            RecursionTemplate("recursion", "Recursive function definitions"),
            LambdaTemplate("lambda", "Higher-order functions with lambdas"),
            LetBindingTemplate("let", "Variable scoping with let bindings"),
        ]
        self.asg_builder = ASGBuilder()

        # Diversity tracking
        self.operator_counts: dict[str, int] = {}
        self.template_counts: dict[str, int] = {}

    def generate_dataset(
        self, num_samples: int, output_dir: Path, samples_per_template: int | None = None
    ) -> None:
        """
        Generate a dataset of synthetic programs.

        Args:
            num_samples: Total number of programs to generate
            output_dir: Directory to save the generated programs
            samples_per_template: If set, generate exactly this many per template
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if samples_per_template:
            num_samples = len(self.templates) * samples_per_template

        generated = 0
        while generated < num_samples:
            # Select template (round-robin or weighted)
            if samples_per_template:
                template_idx = (generated // samples_per_template) % len(self.templates)
            else:
                template_idx = self.rng.randint(0, len(self.templates) - 1)

            template = self.templates[template_idx]

            # Generate program
            ast_root, metadata = template.generate(self.rng)

            # Verify properties
            if not self._verify_properties(metadata):
                continue  # Reject and regenerate

            # Convert to ASG
            asg_data = self.asg_builder.build(ast_root)

            # Save as PyTorch binary
            pt_path = output_dir / f"sample_{generated:06d}.pt"
            import torch

            torch.save(asg_data, pt_path)

            # Save metadata as JSON
            json_path = output_dir / f"sample_{generated:06d}.json"
            with open(json_path, "w") as f:
                json.dump(asdict(metadata), f, indent=2)

            # Update tracking
            self.template_counts[template.template_id] = (
                self.template_counts.get(template.template_id, 0) + 1
            )
            for op in metadata.operator_types:
                self.operator_counts[op] = self.operator_counts.get(op, 0) + 1

            generated += 1

            if generated % 10000 == 0:
                print(f"Generated {generated}/{num_samples} programs")

        print("\n=== Dataset Statistics ===")
        print(f"Total programs: {generated}")
        print(f"\nTemplate distribution:")
        for tid, count in sorted(self.template_counts.items()):
            print(f"  {tid}: {count}")
        print(f"\nOperator distribution:")
        for op, count in sorted(self.operator_counts.items()):
            print(f"  {op}: {count}")

    def _verify_properties(self, metadata: ProgramMetadata) -> bool:
        """Verify that generated program satisfies property constraints."""
        # Must have non-trivial depth
        if metadata.depth < 2:
            return False

        # Must use at least 1 operator
        if metadata.num_operators < 1:
            return False

        # Must have at least 1 variable binding
        if not metadata.has_variables:
            return False

        return True
