"""Additional simple templates for maximum diversity - each with unique root structure."""

from src.data.asg_builder import ASTNode, NodeType
import random


class SingleSymbolTemplate:
    """Just a symbol: x"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        sym = rng.choice(["x", "y", "z", "result", "value", "temp"])
        root = ASTNode(NodeType.SYMBOL, value=sym)

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 0, 1, 0, False, True, [])


class SingleNumberTemplate:
    """Just a number: 42"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        num = rng.randint(0, 100)
        root = ASTNode(NodeType.NUMBER, value=num)

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 0, 1, 0, False, False, [])


class ComparisonOnlyTemplate:
    """Just a comparison: (= x y), (< a b), (> m n)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        comp_ops = ["=", "<", ">"]
        op = rng.choice(comp_ops)

        # Vary operands
        operand_type = rng.choice(["symbols", "numbers", "mixed"])

        if operand_type == "symbols":
            left = ASTNode(NodeType.SYMBOL, value=rng.choice(["x", "y", "a", "b"]))
            right = ASTNode(NodeType.SYMBOL, value=rng.choice(["x", "y", "a", "b"]))
        elif operand_type == "numbers":
            left = ASTNode(NodeType.NUMBER, value=rng.randint(0, 10))
            right = ASTNode(NodeType.NUMBER, value=rng.randint(0, 10))
        else:  # mixed
            left = ASTNode(NodeType.SYMBOL, value=rng.choice(["x", "y"]))
            right = ASTNode(NodeType.NUMBER, value=rng.randint(0, 10))

        root = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value=op),
                left,
                right,
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 1, 4, 1, False, True, [op])


class NestedComparisonTemplate:
    """Nested comparisons: (= (+ x 1) y)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        comp_op = rng.choice(["=", "<", ">"])
        arith_op = rng.choice(["+", "-", "*"])

        # Which side is nested
        side = rng.choice(["left", "right", "both"])

        if side == "left":
            # (= (+ x 1) y)
            left = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=arith_op),
                    ASTNode(NodeType.SYMBOL, value="x"),
                    ASTNode(NodeType.NUMBER, value=rng.randint(1, 5)),
                ],
            )
            right = ASTNode(NodeType.SYMBOL, value="y")
            depth, nodes, ops = 2, 7, 2
        elif side == "right":
            # (= x (+ y 1))
            left = ASTNode(NodeType.SYMBOL, value="x")
            right = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=arith_op),
                    ASTNode(NodeType.SYMBOL, value="y"),
                    ASTNode(NodeType.NUMBER, value=rng.randint(1, 5)),
                ],
            )
            depth, nodes, ops = 2, 7, 2
        else:  # both
            # (= (+ x 1) (- y 2))
            left = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=arith_op),
                    ASTNode(NodeType.SYMBOL, value="x"),
                    ASTNode(NodeType.NUMBER, value=rng.randint(1, 5)),
                ],
            )
            arith_op2 = rng.choice(["+", "-", "*"])
            right = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=arith_op2),
                    ASTNode(NodeType.SYMBOL, value="y"),
                    ASTNode(NodeType.NUMBER, value=rng.randint(1, 5)),
                ],
            )
            depth, nodes, ops = 2, 10, 3

        root = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value=comp_op),
                left,
                right,
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(
            self.template_id,
            depth,
            nodes,
            ops,
            False,
            True,
            [comp_op, arith_op] if side != "both" else [comp_op, arith_op, arith_op2],
        )


class ListLiteralTemplate:
    """List literals: (list a b c)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        num_elements = rng.choice([2, 3, 4, 5])

        children = [ASTNode(NodeType.SYMBOL, value="list")]
        for i in range(num_elements):
            # Mix symbols and numbers
            if rng.choice([True, False]):
                children.append(ASTNode(NodeType.SYMBOL, value=chr(ord("a") + i)))
            else:
                children.append(ASTNode(NodeType.NUMBER, value=rng.randint(1, 10)))

        root = ASTNode(NodeType.LIST, children=children)

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 1, num_elements + 1, 0, False, True, [])


class MultiDefineTemplate:
    """Multiple defines in sequence (simulated with nested LETs)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        # Create nested DE FINE-like structure using LET
        var1 = "x"
        val1 = rng.randint(1, 10)
        var2 = "y"
        val2 = rng.randint(1, 10)
        op = rng.choice(["+", "-", "*"])

        # (let ((x val1)) (let ((y val2)) (op x y)))
        inner_let = ASTNode(
            NodeType.LET,
            children=[
                ASTNode(
                    NodeType.LIST,
                    children=[
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
                    ],
                ),
                inner_let,
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 4, 16, 1, False, True, [op])


class IfWithArithBranchesTemplate:
    """IF with arithmetic in branches: (if cond (+ a b) (- c d))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        comp_op = rng.choice(["=", "<", ">"])
        op1, op2 = rng.choice(["+", "-", "*"]), rng.choice(["+", "-", "*"])

        # Condition
        cond = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value=comp_op),
                ASTNode(NodeType.SYMBOL, value="x"),
                ASTNode(NodeType.NUMBER, value=rng.randint(0, 5)),
            ],
        )

        # Then branch: (op1 a b)
        then_branch = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value=op1),
                ASTNode(NodeType.SYMBOL, value="a"),
                ASTNode(NodeType.SYMBOL, value="b"),
            ],
        )

        # Else branch: (op2 c d)
        else_branch = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value=op2),
                ASTNode(NodeType.SYMBOL, value="c"),
                ASTNode(NodeType.SYMBOL, value="d"),
            ],
        )

        root = ASTNode(NodeType.IF, children=[cond, then_branch, else_branch])

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 2, 13, 3, False, True, [comp_op, op1, op2])


class LambdaWithLetTemplate:
    """Lambda with let in body: (lambda (x) (let ((y 1)) (+ x y)))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        param = "x"
        var = "y"
        val = rng.randint(1, 10)
        op = rng.choice(["+", "-", "*"])

        # Lambda body: (let ((y val)) (op x y))
        let_body = ASTNode(
            NodeType.LET,
            children=[
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.SYMBOL, value=var),
                                ASTNode(NodeType.NUMBER, value=val),
                            ],
                        ),
                    ],
                ),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=op),
                        ASTNode(NodeType.SYMBOL, value=param),
                        ASTNode(NodeType.SYMBOL, value=var),
                    ],
                ),
            ],
        )

        root = ASTNode(
            NodeType.LAMBDA,
            children=[
                ASTNode(NodeType.LIST, children=[ASTNode(NodeType.SYMBOL, value=param)]),
                let_body,
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 3, 11, 1, False, True, [op])
