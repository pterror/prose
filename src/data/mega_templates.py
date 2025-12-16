"""Additional 12 template types for maximum diversity - each with unique structure."""

from src.data.asg_builder import ASTNode, NodeType
import random


class QuotedListTemplate:
    """Quoted list: '(a b c)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        num_items = rng.choice([2, 3, 4])
        # Use (list a b c) instead of '(a b c) since we don't have QUOTE
        items = [ASTNode(NodeType.SYMBOL, value=chr(ord("a") + i)) for i in range(num_items)]
        root = ASTNode(NodeType.LIST, children=[ASTNode(NodeType.SYMBOL, value="list"), *items])

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 1, num_items + 2, 0, False, True, [])


class LambdaMultiParamTemplate:
    """Lambda with multiple parameters: (lambda (x y) (+ x y))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        num_params = rng.choice([2, 3])
        params = [chr(ord("x") + i) for i in range(num_params)]
        op = rng.choice(["+", "-", "*"])

        if num_params == 2:
            body = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=op),
                    ASTNode(NodeType.SYMBOL, value=params[0]),
                    ASTNode(NodeType.SYMBOL, value=params[1]),
                ],
            )
        else:  # 3 params
            body = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=op),
                    ASTNode(NodeType.SYMBOL, value=params[0]),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=op),
                            ASTNode(NodeType.SYMBOL, value=params[1]),
                            ASTNode(NodeType.SYMBOL, value=params[2]),
                        ],
                    ),
                ],
            )

        root = ASTNode(
            NodeType.LAMBDA,
            children=[
                ASTNode(
                    NodeType.LIST, children=[ASTNode(NodeType.SYMBOL, value=p) for p in params]
                ),
                body,
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(
            self.template_id,
            2 if num_params == 2 else 3,
            7 if num_params == 2 else 10,
            1 if num_params == 2 else 2,
            False,
            True,
            [op] if num_params == 2 else [op, op],
        )


class CondWithLetTemplate:
    """Conditional with LET in branches: (if cond (let ((x 1)) x) y)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        comp_op = rng.choice(["=", "<", ">"])
        val = rng.randint(1, 10)

        cond = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value=comp_op),
                ASTNode(NodeType.SYMBOL, value="test"),
                ASTNode(NodeType.NUMBER, value=0),
            ],
        )

        then_branch = ASTNode(
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
                ASTNode(NodeType.SYMBOL, value="x"),
            ],
        )

        else_branch = ASTNode(NodeType.SYMBOL, value="y")

        root = ASTNode(NodeType.IF, children=[cond, then_branch, else_branch])

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 3, 12, 1, False, True, [comp_op])


class DeepNestedArithTemplate:
    """Very deep nested: (+ (- (* (/ a b) c) d) e)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        ops = ["+", "-", "*", "/"]
        depth = rng.choice([4, 5])

        if depth == 4:
            # (op1 (op2 (op3 (op4 a b) c) d) e)
            o1, o2, o3, o4 = [rng.choice(ops) for _ in range(4)]
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=o1),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=o2),
                            ASTNode(
                                NodeType.LIST,
                                children=[
                                    ASTNode(NodeType.OPERATOR, value=o3),
                                    ASTNode(
                                        NodeType.LIST,
                                        children=[
                                            ASTNode(NodeType.OPERATOR, value=o4),
                                            ASTNode(NodeType.SYMBOL, value="a"),
                                            ASTNode(NodeType.SYMBOL, value="b"),
                                        ],
                                    ),
                                    ASTNode(NodeType.SYMBOL, value="c"),
                                ],
                            ),
                            ASTNode(NodeType.SYMBOL, value="d"),
                        ],
                    ),
                    ASTNode(NodeType.SYMBOL, value="e"),
                ],
            )

            from src.data.synthetic_gen import ProgramMetadata

            return root, ProgramMetadata(self.template_id, 4, 13, 4, False, True, [o1, o2, o3, o4])
        else:  # depth 5
            o1, o2, o3, o4, o5 = [rng.choice(ops) for _ in range(5)]
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=o1),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=o2),
                            ASTNode(
                                NodeType.LIST,
                                children=[
                                    ASTNode(NodeType.OPERATOR, value=o3),
                                    ASTNode(
                                        NodeType.LIST,
                                        children=[
                                            ASTNode(NodeType.OPERATOR, value=o4),
                                            ASTNode(
                                                NodeType.LIST,
                                                children=[
                                                    ASTNode(NodeType.OPERATOR, value=o5),
                                                    ASTNode(NodeType.SYMBOL, value="a"),
                                                    ASTNode(NodeType.SYMBOL, value="b"),
                                                ],
                                            ),
                                            ASTNode(NodeType.SYMBOL, value="c"),
                                        ],
                                    ),
                                    ASTNode(NodeType.SYMBOL, value="d"),
                                ],
                            ),
                            ASTNode(NodeType.SYMBOL, value="e"),
                        ],
                    ),
                    ASTNode(NodeType.SYMBOL, value="f"),
                ],
            )

            from src.data.synthetic_gen import ProgramMetadata

            return root, ProgramMetadata(
                self.template_id, 5, 16, 5, False, True, [o1, o2, o3, o4, o5]
            )


class ConsTemplate:
    """Cons operation: (cons a (cons b nil))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        depth = rng.choice([1, 2, 3])

        if depth == 1:
            # (cons a nil)
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.SYMBOL, value="cons"),
                    ASTNode(NodeType.SYMBOL, value="a"),
                    ASTNode(NodeType.SYMBOL, value="nil"),
                ],
            )

            from src.data.synthetic_gen import ProgramMetadata

            return root, ProgramMetadata(self.template_id, 1, 4, 0, False, True, [])
        elif depth == 2:
            # (cons a (cons b nil))
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.SYMBOL, value="cons"),
                    ASTNode(NodeType.SYMBOL, value="a"),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.SYMBOL, value="cons"),
                            ASTNode(NodeType.SYMBOL, value="b"),
                            ASTNode(NodeType.SYMBOL, value="nil"),
                        ],
                    ),
                ],
            )

            from src.data.synthetic_gen import ProgramMetadata

            return root, ProgramMetadata(self.template_id, 2, 7, 0, False, True, [])
        else:  # depth 3
            # (cons a (cons b (cons c nil)))
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.SYMBOL, value="cons"),
                    ASTNode(NodeType.SYMBOL, value="a"),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.SYMBOL, value="cons"),
                            ASTNode(NodeType.SYMBOL, value="b"),
                            ASTNode(
                                NodeType.LIST,
                                children=[
                                    ASTNode(NodeType.SYMBOL, value="cons"),
                                    ASTNode(NodeType.SYMBOL, value="c"),
                                    ASTNode(NodeType.SYMBOL, value="nil"),
                                ],
                            ),
                        ],
                    ),
                ],
            )

            from src.data.synthetic_gen import ProgramMetadata

            return root, ProgramMetadata(self.template_id, 3, 10, 0, False, True, [])


class CarCdrTemplate:
    """Car/Cdr operations: (car (cdr lst))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        ops = ["car", "cdr"]
        depth = rng.choice([1, 2])

        if depth == 1:
            # (car lst) or (cdr lst)
            op = rng.choice(ops)
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.SYMBOL, value=op),
                    ASTNode(NodeType.SYMBOL, value="lst"),
                ],
            )

            from src.data.synthetic_gen import ProgramMetadata

            return root, ProgramMetadata(self.template_id, 1, 3, 0, False, True, [])
        else:  # depth 2
            # (car (cdr lst))
            op1, op2 = rng.choice(ops), rng.choice(ops)
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.SYMBOL, value=op1),
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.SYMBOL, value=op2),
                            ASTNode(NodeType.SYMBOL, value="lst"),
                        ],
                    ),
                ],
            )

            from src.data.synthetic_gen import ProgramMetadata

            return root, ProgramMetadata(self.template_id, 2, 5, 0, False, True, [])


class BeginTemplate:
    """Begin/sequence: (begin expr1 expr2 expr3)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        num_exprs = rng.choice([2, 3, 4])
        ops = ["+", "-", "*"]

        children = [ASTNode(NodeType.SYMBOL, value="begin")]

        for i in range(num_exprs):
            if i < num_exprs - 1:
                # Simple arithmetic for side effects
                op = rng.choice(ops)
                children.append(
                    ASTNode(
                        NodeType.LIST,
                        children=[
                            ASTNode(NodeType.OPERATOR, value=op),
                            ASTNode(NodeType.SYMBOL, value=chr(ord("a") + i)),
                            ASTNode(NodeType.NUMBER, value=rng.randint(1, 5)),
                        ],
                    )
                )
            else:
                # Return value
                children.append(ASTNode(NodeType.SYMBOL, value="result"))

        root = ASTNode(NodeType.LIST, children=children)

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(
            self.template_id,
            2,
            2 + (num_exprs - 1) * 4 + 1,
            num_exprs - 1,
            False,
            True,
            ops[: num_exprs - 1],
        )


class AndOrTemplate:
    """Logical operations: (and (> x 0) (< x 10))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        logical_op = rng.choice(["and", "or"])
        comp_ops = ["=", "<", ">"]

        num_conditions = rng.choice([2, 3])

        children = [ASTNode(NodeType.SYMBOL, value=logical_op)]
        ops_list = []

        for i in range(num_conditions):
            comp_op = rng.choice(comp_ops)
            ops_list.append(comp_op)
            children.append(
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=comp_op),
                        ASTNode(NodeType.SYMBOL, value=chr(ord("x") + i)),
                        ASTNode(NodeType.NUMBER, value=rng.randint(0, 10)),
                    ],
                )
            )

        root = ASTNode(NodeType.LIST, children=children)

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(
            self.template_id, 2, 2 + num_conditions * 4, num_conditions, False, True, ops_list
        )


class NotTemplate:
    """Negation: (not (= x 0))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        comp_op = rng.choice(["=", "<", ">"])

        root = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.SYMBOL, value="not"),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value=comp_op),
                        ASTNode(NodeType.SYMBOL, value="x"),
                        ASTNode(NodeType.NUMBER, value=rng.randint(0, 10)),
                    ],
                ),
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 2, 6, 1, False, True, [comp_op])


class ApplyTemplate:
    """Function application: (apply f (list a b c))"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        num_args = rng.choice([2, 3, 4])

        args = [ASTNode(NodeType.SYMBOL, value=chr(ord("a") + i)) for i in range(num_args)]

        root = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.SYMBOL, value="apply"),
                ASTNode(NodeType.SYMBOL, value="f"),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.SYMBOL, value="list"),
                        *args,
                    ],
                ),
            ],
        )

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(self.template_id, 2, 4 + num_args, 0, False, True, [])


class MapTemplate:
    """Map operation: (map f lst)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        # Simple or with lambda
        use_lambda = rng.choice([True, False])

        if use_lambda:
            # (map (lambda (x) (+ x 1)) lst)
            op = rng.choice(["+", "-", "*"])
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.SYMBOL, value="map"),
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
                    ASTNode(NodeType.SYMBOL, value="lst"),
                ],
            )

            from src.data.synthetic_gen import ProgramMetadata

            return root, ProgramMetadata(self.template_id, 3, 10, 1, False, True, [op])
        else:
            # (map f lst)
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.SYMBOL, value="map"),
                    ASTNode(NodeType.SYMBOL, value="f"),
                    ASTNode(NodeType.SYMBOL, value="lst"),
                ],
            )

            from src.data.synthetic_gen import ProgramMetadata

            return root, ProgramMetadata(self.template_id, 1, 4, 0, False, True, [])


class FilterTemplate:
    """Filter operation: (filter pred lst)"""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        # With predicate lambda
        use_lambda = rng.choice([True, False])

        if use_lambda:
            # (filter (lambda (x) (> x 0)) lst)
            comp_op = rng.choice(["=", "<", ">"])
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.SYMBOL, value="filter"),
                    ASTNode(
                        NodeType.LAMBDA,
                        children=[
                            ASTNode(NodeType.LIST, children=[ASTNode(NodeType.SYMBOL, value="x")]),
                            ASTNode(
                                NodeType.LIST,
                                children=[
                                    ASTNode(NodeType.OPERATOR, value=comp_op),
                                    ASTNode(NodeType.SYMBOL, value="x"),
                                    ASTNode(NodeType.NUMBER, value=0),
                                ],
                            ),
                        ],
                    ),
                    ASTNode(NodeType.SYMBOL, value="lst"),
                ],
            )

            from src.data.synthetic_gen import ProgramMetadata

            return root, ProgramMetadata(self.template_id, 3, 10, 1, False, True, [comp_op])
        else:
            # (filter pred lst)
            root = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.SYMBOL, value="filter"),
                    ASTNode(NodeType.SYMBOL, value="pred"),
                    ASTNode(NodeType.SYMBOL, value="lst"),
                ],
            )

            from src.data.synthetic_gen import ProgramMetadata

            return root, ProgramMetadata(self.template_id, 1, 4, 0, False, True, [])
