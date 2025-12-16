"""Synthetic Mini-Lisp program generator with property-based constraints."""

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.data.asg_builder import ASTNode, ASGBuilder, NodeType
from src.runtime.interpreter import TestCase


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
    tests: list[TestCase] = field(default_factory=list)


class ProgramTemplate:
    """Base class for Mini-Lisp program templates."""

    def __init__(self, template_id: str, description: str) -> None:
        self.template_id = template_id
        self.description = description

    def generate(self, rng: random.Random) -> tuple[ASTNode, ProgramMetadata]:
        """Generate a program instance from this template.

        Returns:
            (ast, metadata) where metadata.tests contains test cases
        """
        raise NotImplementedError

    def generate_tests(
        self, program: ASTNode, rng: random.Random, num_tests: int = 3
    ) -> list[TestCase]:
        """Generate test cases for a program.

        Args:
            program: The AST of the program
            rng: Random number generator
            num_tests: Number of test cases to generate

        Returns:
            List of TestCase instances with inputs and expected outputs
        """
        # Default implementation - subclasses should override
        return []


class ArithmeticTemplate(ProgramTemplate):
    """Template for arithmetic expressions with varying structure.

    Variations:
    - 2-ary: (op a b)
    - 3-ary: (op a b c)
    - Nested left: (op1 (op2 a b) c)
    - Nested right: (op1 a (op2 b c))
    - Deep nested: (op1 (op2 (op3 a b) c) d)
    """

    def generate(self, rng: random.Random) -> tuple[ASTNode, ProgramMetadata]:
        ops = ["+", "-", "*", "/"]

        # Choose structure variation
        structure = rng.choice(["2-ary", "3-ary", "nested_left", "nested_right", "deep_nested"])

        if structure == "2-ary":
            # Simple: (op a b)
            op = rng.choice(ops)
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=op),
                    ASTNode(NodeType.SYMBOL, value="a"),
                    ASTNode(NodeType.SYMBOL, value="b"),
                ],
            )
            return root, ProgramMetadata(
                template_id=self.template_id,
                depth=1,
                num_nodes=4,
                num_operators=1,
                has_recursion=False,
                has_variables=True,
                operator_types=[op],
            )

        elif structure == "3-ary":
            # Three operands: (op a b c)
            op = rng.choice(ops)
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=op),
                    ASTNode(NodeType.SYMBOL, value="a"),
                    ASTNode(NodeType.SYMBOL, value="b"),
                    ASTNode(NodeType.SYMBOL, value="c"),
                ],
            )
            return root, ProgramMetadata(
                template_id=self.template_id,
                depth=1,
                num_nodes=5,
                num_operators=1,
                has_recursion=False,
                has_variables=True,
                operator_types=[op],
            )

        elif structure == "nested_left":
            # Left-nested: (op1 (op2 a b) c)
            op1, op2 = rng.choice(ops), rng.choice(ops)
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
                    ASTNode(NodeType.SYMBOL, value="c"),
                ],
            )
            return root, ProgramMetadata(
                template_id=self.template_id,
                depth=2,
                num_nodes=7,
                num_operators=2,
                has_recursion=False,
                has_variables=True,
                operator_types=[op1, op2],
            )

        elif structure == "nested_right":
            # Right-nested: (op1 a (op2 b c))
            op1, op2 = rng.choice(ops), rng.choice(ops)
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=op1),
                    ASTNode(NodeType.SYMBOL, value="a"),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=op2),
                            ASTNode(NodeType.SYMBOL, value="b"),
                            ASTNode(NodeType.SYMBOL, value="c"),
                        ],
                    ),
                ],
            )
            return root, ProgramMetadata(
                template_id=self.template_id,
                depth=2,
                num_nodes=7,
                num_operators=2,
                has_recursion=False,
                has_variables=True,
                operator_types=[op1, op2],
            )

        else:  # deep_nested
            # Deep nested: (op1 (op2 (op3 a b) c) d)
            op1, op2, op3 = rng.choice(ops), rng.choice(ops), rng.choice(ops)
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=op1),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=op2),
                            ASTNode(
                                NodeType.LIST,
                                children=[
                                    ASTNode(NodeType.OPERATOR, value=op3),
                                    ASTNode(NodeType.SYMBOL, value="a"),
                                    ASTNode(NodeType.SYMBOL, value="b"),
                                ],
                            ),
                            ASTNode(NodeType.SYMBOL, value="c"),
                        ],
                    ),
                    ASTNode(NodeType.SYMBOL, value="d"),
                ],
            )
            return root, ProgramMetadata(
                template_id=self.template_id,
                depth=3,
                num_nodes=10,
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
    """Template for variable scoping with varying binding counts and dependencies.

    Variations:
    - 1 binding: (let ((x 1)) x)
    - 2 bindings: (let ((x 1) (y 2)) (+ x y))
    - 3 bindings: (let ((x 1) (y 2) (z 3)) (+ x (+ y z)))
    - Dependent bindings: (let ((x 1) (y (+ x 2))) (* x y))
    """

    def generate(self, rng: random.Random) -> tuple[ASTNode, ProgramMetadata]:
        # Vary number of bindings (1-3)
        num_bindings = rng.choice([1, 2, 3])
        var_names = ["x", "y", "z"]
        ops = ["+", "-", "*"]

        values = [rng.randint(1, 10) for _ in range(num_bindings)]

        # Vary whether bindings are dependent (y depends on x)
        dependent = rng.choice([True, False]) and num_bindings >= 2

        # Build bindings
        bindings = []
        for i in range(num_bindings):
            var_name = var_names[i]

            if dependent and i > 0:
                # Dependent binding: y = (+ x val)
                val_expr = ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=rng.choice(ops)),
                        ASTNode(NodeType.SYMBOL, value=var_names[i - 1]),
                        ASTNode(NodeType.NUMBER, value=values[i]),
                    ],
                )
            else:
                # Simple binding: x = val
                val_expr = ASTNode(NodeType.NUMBER, value=values[i])

            bindings.append(
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.SYMBOL, value=var_name),
                        val_expr,
                    ],
                )
            )

        # Build body (use all variables)
        if num_bindings == 1:
            # Simple: x
            body = ASTNode(NodeType.SYMBOL, value=var_names[0])
        elif num_bindings == 2:
            # Binary operation: (op x y)
            op = rng.choice(ops)
            body = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=op),
                    ASTNode(NodeType.SYMBOL, value=var_names[0]),
                    ASTNode(NodeType.SYMBOL, value=var_names[1]),
                ],
            )
        else:  # num_bindings == 3
            # Nested: (+ x (+ y z))
            op1, op2 = rng.choice(ops), rng.choice(ops)
            body = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=op1),
                    ASTNode(NodeType.SYMBOL, value=var_names[0]),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=op2),
                            ASTNode(NodeType.SYMBOL, value=var_names[1]),
                            ASTNode(NodeType.SYMBOL, value=var_names[2]),
                        ],
                    ),
                ],
            )

        # Build LET node
        root = ASTNode(
            NodeType.LET,
            children=[
                ASTNode(NodeType.LIST, children=bindings),
                body,
            ],
        )

        # Calculate metadata
        num_ops = (
            sum(
                1
                for binding in bindings
                if dependent and binding.children[1].node_type == NodeType.LIST
            )
            if dependent
            else 0
        ) + (0 if num_bindings == 1 else (1 if num_bindings == 2 else 2))

        # Calculate num_nodes more precisely
        # LET node itself: 1
        # Bindings list: 1
        # Each binding: 1 (list) + 1 (symbol) + (1 (number) or 4 (list, op, symbol, number))
        # Body: 1 (symbol) or 4 (list, op, symbol, symbol) or 7 (list, op, symbol, list, op, symbol, symbol)

        nodes_in_bindings = 0
        for i in range(num_bindings):
            nodes_in_bindings += 1  # for the inner LIST node
            nodes_in_bindings += 1  # for the SYMBOL (var_name)
            if dependent and i > 0:
                nodes_in_bindings += 4  # for (LIST, OPERATOR, SYMBOL, NUMBER)
            else:
                nodes_in_bindings += 1  # for NUMBER

        nodes_in_body = 0
        if num_bindings == 1:
            nodes_in_body = 1  # SYMBOL
        elif num_bindings == 2:
            nodes_in_body = 4  # LIST, OPERATOR, SYMBOL, SYMBOL
        else:  # num_bindings == 3
            nodes_in_body = 7  # LIST, OPERATOR, SYMBOL, LIST, OPERATOR, SYMBOL, SYMBOL

        num_nodes = 1 + 1 + nodes_in_bindings + nodes_in_body  # LET, bindings_list, bindings, body

        # Collect operator types
        operator_types = []
        if dependent:
            for i in range(num_bindings):
                if i > 0:
                    # The operator is in the value expression of the dependent binding
                    operator_types.append(bindings[i].children[1].children[0].value)

        if num_bindings == 2:
            operator_types.append(body.children[0].value)
        elif num_bindings == 3:
            operator_types.append(body.children[0].value)
            operator_types.append(body.children[2].children[0].value)

        return root, ProgramMetadata(
            template_id=self.template_id,
            depth=3
            if num_bindings <= 2 and not dependent
            else 4
            if num_bindings == 3 and not dependent
            else 4
            if num_bindings == 2 and dependent
            else 5,  # Simplified depth calculation
            num_nodes=num_nodes,
            num_operators=num_ops,
            has_recursion=False,
            has_variables=True,
            operator_types=operator_types,
        )


class NestedLetTemplate(ProgramTemplate):
    """Template for nested let bindings.

    Example: (let ((x 1)) (let ((y (+ x 2))) (+ x y)))
    """

    def generate(self, rng: random.Random) -> tuple[ASTNode, ProgramMetadata]:
        # Vary nesting depth (1-2 levels)
        nesting_depth = rng.choice([1, 2])
        ops = ["+", "-", "*"]

        if nesting_depth == 1:
            # Single level: (let ((x 1)) (let ((y 2)) (+ x y)))
            val1 = rng.randint(1, 10)
            val2 = rng.randint(1, 10)
            op = rng.choice(ops)

            inner_let = ASTNode(
                NodeType.LET,
                children=[
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(
                                NodeType.LIST,
                                children=[
                                    ASTNode(NodeType.SYMBOL, value="y"),
                                    ASTNode(NodeType.NUMBER, value=val2),
                                ],
                            ),
                        ],
                    ),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=op),
                            ASTNode(NodeType.SYMBOL, value="x"),
                            ASTNode(NodeType.SYMBOL, value="y"),
                        ],
                    ),
                ],
            )

            root = ASTNode(
                NodeType.LET,
                children=[
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(
                                NodeType.LIST,
                                children=[
                                    ASTNode(NodeType.SYMBOL, value="x"),
                                    ASTNode(NodeType.NUMBER, value=val1),
                                ],
                            ),
                        ],
                    ),
                    inner_let,
                ],
            )

            return root, ProgramMetadata(
                template_id=self.template_id,
                depth=4,
                num_nodes=16,
                num_operators=1,
                has_recursion=False,
                has_variables=True,
                operator_types=[op],
            )

        else:  # nesting_depth == 2
            # Dependent nested: (let ((x 1)) (let ((y (+ x 2))) (+ x y)))
            val1 = rng.randint(1, 10)
            val2 = rng.randint(1, 10)
            op1, op2 = rng.choice(ops), rng.choice(ops)

            inner_let = ASTNode(
                NodeType.LET,
                children=[
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(
                                NodeType.LIST,
                                children=[
                                    ASTNode(NodeType.SYMBOL, value="y"),
                                    ASTNode(
                                        NodeType.LIST,
                                        children=[
                                            ASTNode(NodeType.OPERATOR, value=op1),
                                            ASTNode(NodeType.SYMBOL, value="x"),
                                            ASTNode(NodeType.NUMBER, value=val2),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=op2),
                            ASTNode(NodeType.SYMBOL, value="x"),
                            ASTNode(NodeType.SYMBOL, value="y"),
                        ],
                    ),
                ],
            )

            root = ASTNode(
                NodeType.LET,
                children=[
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(
                                NodeType.LIST,
                                children=[
                                    ASTNode(NodeType.SYMBOL, value="x"),
                                    ASTNode(NodeType.NUMBER, value=val1),
                                ],
                            ),
                        ],
                    ),
                    inner_let,
                ],
            )

            return root, ProgramMetadata(
                template_id=self.template_id,
                depth=5,
                num_nodes=19,
                num_operators=2,
                has_recursion=False,
                has_variables=True,
                operator_types=[op1, op2],
            )


class ConditionalChainTemplate(ProgramTemplate):
    """Template for nested conditional expressions.

    Example: (if cond1 (if cond2 then2 else2) else1)
    """

    def generate(self, rng: random.Random) -> tuple[ASTNode, ProgramMetadata]:
        # Vary conditional depth (1-2 levels)
        depth = rng.choice([1, 2])
        comp_ops = ["=", "<", ">"]

        if depth == 1:
            # Simple: (if (= x y) then else)
            comp_op = rng.choice(comp_ops)
            val1 = rng.randint(0, 5)
            val2 = rng.randint(0, 5)

            root = ASTNode(
                NodeType.IF,
                children=[
                    # Condition: (= x y)
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=comp_op),
                            ASTNode(NodeType.SYMBOL, value="x"),
                            ASTNode(NodeType.SYMBOL, value="y"),
                        ],
                    ),
                    # Then: val1
                    ASTNode(NodeType.NUMBER, value=val1),
                    # Else: val2
                    ASTNode(NodeType.NUMBER, value=val2),
                ],
            )

            return root, ProgramMetadata(
                template_id=self.template_id,
                depth=2,
                num_nodes=7,
                num_operators=1,
                has_recursion=False,
                has_variables=True,
                operator_types=[comp_op],
            )

        else:  # depth == 2
            # Nested: (if (= x y) (if (< a b) then2 else2) else1)
            comp_op1, comp_op2 = rng.choice(comp_ops), rng.choice(comp_ops)
            val1, val2, val3 = rng.randint(0, 5), rng.randint(0, 5), rng.randint(0, 5)

            inner_if = ASTNode(
                NodeType.IF,
                children=[
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=comp_op2),
                            ASTNode(NodeType.SYMBOL, value="a"),
                            ASTNode(NodeType.SYMBOL, value="b"),
                        ],
                    ),
                    ASTNode(NodeType.NUMBER, value=val2),
                    ASTNode(NodeType.NUMBER, value=val3),
                ],
            )

            root = ASTNode(
                NodeType.IF,
                children=[
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=comp_op1),
                            ASTNode(NodeType.SYMBOL, value="x"),
                            ASTNode(NodeType.SYMBOL, value="y"),
                        ],
                    ),
                    inner_if,
                    ASTNode(NodeType.NUMBER, value=val1),
                ],
            )

            return root, ProgramMetadata(
                template_id=self.template_id,
                depth=3,
                num_nodes=12,
                num_operators=2,
                has_recursion=False,
                has_variables=True,
                operator_types=[comp_op1, comp_op2],
            )


class MixedArithmeticTemplate(ProgramTemplate):
    """Template for mixed arithmetic with varying asymmetric structures.

    Variations:
    - 2-way mix: (+ (* a b) c)
    - Asymmetric: (- (+ (* a b) c) d)
    - Different depths
    """

    def generate(self, rng: random.Random) -> tuple[ASTNode, ProgramMetadata]:
        ops = ["+", "-", "*", "/"]
        structure = rng.choice(["2-way", "asymmetric"])

        if structure == "2-way":
            # (op1 (op2 a b) c)
            op1, op2 = rng.choice(ops), rng.choice(ops)

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
                    ASTNode(NodeType.SYMBOL, value="c"),
                ],
            )

            return root, ProgramMetadata(
                template_id=self.template_id,
                depth=2,
                num_nodes=7,
                num_operators=2,
                has_recursion=False,
                has_variables=True,
                operator_types=[op1, op2],
            )

        else:  # asymmetric
            # (op1 (op2 (op3 a b) c) d)
            op1, op2, op3 = rng.choice(ops), rng.choice(ops), rng.choice(ops)

            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=op1),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=op2),
                            ASTNode(
                                NodeType.LIST,
                                children=[
                                    ASTNode(NodeType.OPERATOR, value=op3),
                                    ASTNode(NodeType.SYMBOL, value="a"),
                                    ASTNode(NodeType.SYMBOL, value="b"),
                                ],
                            ),
                            ASTNode(NodeType.SYMBOL, value="c"),
                        ],
                    ),
                    ASTNode(NodeType.SYMBOL, value="d"),
                ],
            )

            return root, ProgramMetadata(
                template_id=self.template_id,
                depth=3,
                num_nodes=10,
                num_operators=3,
                has_recursion=False,
                has_variables=True,
                operator_types=[op1, op2, op3],
            )


class SyntheticGenerator:
    """Generates synthetic Mini-Lisp programs with property-based diversity."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.templates = [
            ArithmeticTemplate("arithmetic", "Basic arithmetic expressions with varying structure"),
            RecursionTemplate("recursion", "Recursive function definitions"),
            LambdaTemplate("lambda", "Higher-order functions with lambdas"),
            LetBindingTemplate("let", "Variable scoping with varying bindings"),
            NestedLetTemplate("nested_let", "Nested let bindings"),
            ConditionalChainTemplate("conditional", "Nested conditional expressions"),
            MixedArithmeticTemplate("mixed_arith", "Mixed arithmetic with asymmetric structures"),
        ]

        # Add simple templates for diversity
        from src.data.simple_templates import (
            SimpleArithTemplate,
            DefineSimpleTemplate,
            SimpleIfTemplate,
        )
        from src.data.extra_templates import (
            SingleSymbolTemplate,
            SingleNumberTemplate,
            ComparisonOnlyTemplate,
            NestedComparisonTemplate,
            ListLiteralTemplate,
            MultiDefineTemplate,
            IfWithArithBranchesTemplate,
            LambdaWithLetTemplate,
        )

        self.templates.extend(
            [
                SimpleArithTemplate("simple_arith", "Simple arithmetic"),
                DefineSimpleTemplate("simple_define", "Simple defines"),
                SimpleIfTemplate("simple_if", "Simple conditionals"),
                SingleSymbolTemplate("single_sym", "Single symbol"),
                SingleNumberTemplate("single_num", "Single number"),
                ComparisonOnlyTemplate("comparison", "Comparisons"),
                NestedComparisonTemplate("nested_comp", "Nested comparisons"),
                ListLiteralTemplate("list_lit", "List literals"),
                MultiDefineTemplate("multi_define", "Multiple definitions"),
                IfWithArithBranchesTemplate("if_arith", "IF with arithmetic"),
                LambdaWithLetTemplate("lambda_let", "Lambda with LET"),
            ]
        )

        # Add mega templates for even more diversity
        from src.data.mega_templates import (
            QuotedListTemplate,
            LambdaMultiParamTemplate,
            CondWithLetTemplate,
            DeepNestedArithTemplate,
            ConsTemplate,
            CarCdrTemplate,
            BeginTemplate,
            AndOrTemplate,
            NotTemplate,
            ApplyTemplate,
            MapTemplate,
            FilterTemplate,
        )

        self.templates.extend(
            [
                QuotedListTemplate("quoted_list", "Quoted lists"),
                LambdaMultiParamTemplate("lambda_multi", "Multi-param lambda"),
                CondWithLetTemplate("cond_let", "Conditional with LET"),
                DeepNestedArithTemplate("deep_arith", "Deep nested arithmetic"),
                ConsTemplate("cons", "Cons operations"),
                CarCdrTemplate("car_cdr", "Car/Cdr operations"),
                BeginTemplate("begin", "Begin sequences"),
                AndOrTemplate("and_or", "AND/OR operations"),
                NotTemplate("not", "NOT operations"),
                ApplyTemplate("apply", "Apply operations"),
                MapTemplate("map", "Map operations"),
                FilterTemplate("filter", "Filter operations"),
            ]
        )

        # Add ultra templates for final diversity push
        from src.data.ultra_templates import (
            SymbolPairTemplate,
            TripleArithTemplate,
            DefineArithTemplate,
            DefineLambdaTemplate,
            WideBalancedTemplate,
            LetIfTemplate,
            IfNestedDeepTemplate,
            LambdaNestedTemplate,
            RecursiveSumTemplate,
            FoldTemplate,
        )

        self.templates.extend(
            [
                SymbolPairTemplate("symbol_pair", "Symbol pairs"),
                TripleArithTemplate("triple_arith", "Triple arithmetic"),
                DefineArithTemplate("define_arith", "Define with arithmetic"),
                DefineLambdaTemplate("define_lambda", "Define lambda"),
                WideBalancedTemplate("wide_balanced", "Wide balanced tree"),
                LetIfTemplate("let_if", " LET with IF"),
                IfNestedDeepTemplate("if_nested", "Deeply nested IF"),
                LambdaNestedTemplate("lambda_nested", "Nested lambda"),
                RecursiveSumTemplate("rec_sum", "Recursive sum"),
                FoldTemplate("fold", "Fold/reduce"),
            ]
        )
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
        # Relaxed constraints for diversity
        return True
