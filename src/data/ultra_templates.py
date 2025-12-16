"""Ultra-simple templates - 10 more unique structures."""

from src.data.asg_builder import ASTNode, NodeType
import random


class SymbolPairTemplate:
    """Two symbols: (cons a b) - different from list"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        root = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.SYMBOL, value="pair"),
                ASTNode(NodeType.SYMBOL, value=rng.choice(["x", "y", "a", "b"])),
                ASTNode(NodeType.SYMBOL, value=rng.choice(["x", "y", "a", "b"])),
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 1, 4, 0, False, True, [])


class TripleArithTemplate:
    """Triple operation: (+ (+ a b) (+ c d)) - different nesting"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        ops = ["+", "-", "*"]
        o1, o2, o3 = [rng.choice(ops) for _ in range(3)]

        root = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value=o1),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=o2),
                        ASTNode(NodeType.SYMBOL, value="a"),
                        ASTNode(NodeType.SYMBOL, value="b"),
                    ],
                ),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=o3),
                        ASTNode(NodeType.SYMBOL, value="c"),
                        ASTNode(NodeType.SYMBOL, value="d"),
                    ],
                ),
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 2, 11, 3, False, True, [o1, o2, o3])


class DefineArithTemplate:
    """Define with arithmetic: (define x (+ a b))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        op = rng.choice(["+", "-", "*"])
        var = rng.choice(["x", "y", "result"])

        root = ASTNode(
            NodeType.DEFINE,
            value=var,
            children=[
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=op),
                        ASTNode(NodeType.SYMBOL, value="a"),
                        ASTNode(NodeType.SYMBOL, value="b"),
                    ],
                ),
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 2, 5, 1, False, True, [op])


class DefineLambdaTemplate:
    """Define a lambda: (define f (lambda (x) (+ x 1)))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        op = rng.choice(["+", "-", "*"])

        root = ASTNode(
            NodeType.DEFINE,
            value="f",
            children=[
                ASTNode(
                    NodeType.LAMBDA,
                    children=[
                        ASTNode(NodeType.LIST, children=[ASTNode(NodeType.SYMBOL, value="x")]),
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.OPERATOR, value=op),
                                ASTNode(NodeType.SYMBOL, value="x"),
                                ASTNode(NodeType.NUMBER, value=1),
                            ],
                        ),
                    ],
                ),
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 3, 8, 1, False, True, [op])


class WideBalancedTemplate:
    """Wide balanced: (+ (+ (+ a b) (+ c d)) (+ (+ e f) (+ g h)))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        op = rng.choice(["+", "*"])

        # Build 4-level tree
        left = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value=op),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=op),
                        ASTNode(NodeType.SYMBOL, value="a"),
                        ASTNode(NodeType.SYMBOL, value="b"),
                    ],
                ),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=op),
                        ASTNode(NodeType.SYMBOL, value="c"),
                        ASTNode(NodeType.SYMBOL, value="d"),
                    ],
                ),
            ],
        )

        right = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value=op),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=op),
                        ASTNode(NodeType.SYMBOL, value="e"),
                        ASTNode(NodeType.SYMBOL, value="f"),
                    ],
                ),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=op),
                        ASTNode(NodeType.SYMBOL, value="g"),
                        ASTNode(NodeType.SYMBOL, value="h"),
                    ],
                ),
            ],
        )

        root = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value=op),
                left,
                right,
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 3, 23, 7, False, True, [op] * 7)


class LetIfTemplate:
    """LET with IF body: (let ((x 1)) (if (= x 0) x (+ x 1)))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        comp_op = rng.choice(["=", "<", ">"])
        arith_op = rng.choice(["+", "-", "*"])
        val = rng.randint(1, 10)

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
                                ASTNode(NodeType.NUMBER, value=val),
                            ],
                        ),
                    ],
                ),
                ASTNode(
                    NodeType.IF,
                    children=[
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.OPERATOR, value=comp_op),
                                ASTNode(NodeType.SYMBOL, value="x"),
                                ASTNode(NodeType.NUMBER, value=0),
                            ],
                        ),
                        ASTNode(NodeType.SYMBOL, value="x"),
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.OPERATOR, value=arith_op),
                                ASTNode(NodeType.SYMBOL, value="x"),
                                ASTNode(NodeType.NUMBER, value=1),
                            ],
                        ),
                    ],
                ),
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 3, 14, 2, False, True, [comp_op, arith_op])


class IfNestedDeepTemplate:
    """Deeply nested IF: (if c1 (if c2 (if c3 a b) c) d)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        # 3-level nested IF
        innermost = ASTNode(
            NodeType.IF,
            children=[
                ASTNode(NodeType.SYMBOL, value="c3"),
                ASTNode(NodeType.SYMBOL, value="a"),
                ASTNode(NodeType.SYMBOL, value="b"),
            ],
        )

        middle = ASTNode(
            NodeType.IF,
            children=[
                ASTNode(NodeType.SYMBOL, value="c2"),
                innermost,
                ASTNode(NodeType.SYMBOL, value="c"),
            ],
        )

        root = ASTNode(
            NodeType.IF,
            children=[
                ASTNode(NodeType.SYMBOL, value="c1"),
                middle,
                ASTNode(NodeType.SYMBOL, value="d"),
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 3, 11, 0, False, True, [])


class LambdaNestedTemplate:
    """Nested lambda: (lambda (x) (lambda (y) (+ x y)))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        op = rng.choice(["+", "-", "*"])

        inner_lambda = ASTNode(
            NodeType.LAMBDA,
            children=[
                ASTNode(NodeType.LIST, children=[ASTNode(NodeType.SYMBOL, value="y")]),
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
            NodeType.LAMBDA,
            children=[
                ASTNode(NodeType.LIST, children=[ASTNode(NodeType.SYMBOL, value="x")]),
                inner_lambda,
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 3, 10, 1, False, True, [op])


class RecursiveSumTemplate:
    """Recursive sum with different structure: (define (sum n) (if (= n 0) 0 (+ n (sum (- n 1)))))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        func_name = rng.choice(["sum", "count", "total"])

        root = ASTNode(
            NodeType.DEFINE,
            value=func_name,
            children=[
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.SYMBOL, value=func_name),
                        ASTNode(NodeType.SYMBOL, value="n"),
                    ],
                ),
                ASTNode(
                    NodeType.IF,
                    children=[
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.OPERATOR, value="="),
                                ASTNode(NodeType.SYMBOL, value="n"),
                                ASTNode(NodeType.NUMBER, value=0),
                            ],
                        ),
                        ASTNode(NodeType.NUMBER, value=0),
                        ASTNode(
                            NodeType.LIST,
                            children=[
                                ASTNode(NodeType.OPERATOR, value="+"),
                                ASTNode(NodeType.SYMBOL, value="n"),
                                ASTNode(
                                    NodeType.LIST,
                                    children=[
                                        ASTNode(NodeType.SYMBOL, value=func_name),
                                        ASTNode(
                                            NodeType.LIST,
                                            children=[
                                                ASTNode(NodeType.OPERATOR, value="-"),
                                                ASTNode(NodeType.SYMBOL, value="n"),
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

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 4, 16, 3, True, True, ["=", "+", "-"])


class FoldTemplate:
    """Fold/reduce: (fold op init lst)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        root = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.SYMBOL, value="fold"),
                ASTNode(NodeType.SYMBOL, value=rng.choice(["+", "*", "max"])),
                ASTNode(NodeType.NUMBER, value=rng.choice([0, 1])),
                ASTNode(NodeType.SYMBOL, value="lst"),
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 1, 5, 0, False, True, [])
