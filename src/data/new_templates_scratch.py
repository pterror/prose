"""New template classes for expanded diversity."""

from src.data.asg_builder import ASTNode, NodeType
from src.data.synthetic_gen import ProgramTemplate, ProgramMetadata
import random


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
