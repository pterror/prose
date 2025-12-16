"""Simple templates with diverse structures."""

from src.data.asg_builder import ASTNode, NodeType
import random


class SimpleArithTemplate:
    """Very simple arithmetic - just (op a b)."""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        ops = ["+", "-", "*", "/"]
        op = rng.choice(ops)

        # Vary structure
        num_operands = rng.choice([2, 3, 4])

        children = [ASTNode(NodeType.OPERATOR, value=op)]
        for i in range(num_operands):
            children.append(ASTNode(NodeType.SYMBOL, value=chr(ord("a") + i)))

        root = ASTNode(NodeType.LIST, children=children)

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(
            template_id=self.template_id,
            depth=1,
            num_nodes=num_operands + 1,
            num_operators=1,
            has_recursion=False,
            has_variables=True,
            operator_types=[op],
        )


class DefineSimpleTemplate:
    """Simple define: (define x value)."""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        var_name = rng.choice(["x", "y", "z", "result"])

        # Vary the value expression
        value_type = rng.choice(["number", "symbol", "simple_expr"])

        if value_type == "number":
            value_node = ASTNode(NodeType.NUMBER, value=rng.randint(1, 100))
        elif value_type == "symbol":
            value_node = ASTNode(NodeType.SYMBOL, value=rng.choice(["a", "b", "c"]))
        else:  # simple_expr
            op = rng.choice(["+", "-", "*"])
            value_node = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=op),
                    ASTNode(NodeType.SYMBOL, value="a"),
                    ASTNode(NodeType.SYMBOL, value="b"),
                ],
            )

        root = ASTNode(NodeType.DEFINE, value=var_name, children=[value_node])

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(
            template_id=self.template_id,
            depth=1 if value_type != "simple_expr" else 2,
            num_nodes=2 if value_type != "simple_expr" else 5,
            num_operators=0 if value_type != "simple_expr" else 1,
            has_recursion=False,
            has_variables=True,
            operator_types=[] if value_type != "simple_expr" else [op],
        )


class SimpleIfTemplate:
    """Simple if without nesting: (if cond then else)."""

    def __init__(self, template_id, description):
        self.template_id = template_id
        self.description = description

    def generate(self, rng):
        # Vary condition complexity
        cond_type = rng.choice(["symbol", "comparison"])

        if cond_type == "symbol":
            cond = ASTNode(NodeType.SYMBOL, value=rng.choice(["flag", "test", "check"]))
        else:  # comparison
            op = rng.choice(["=", "<", ">"])
            cond = ASTNode(
                NodeType.LIST,
                children=[
                    ASTNode(NodeType.OPERATOR, value=op),
                    ASTNode(NodeType.SYMBOL, value="x"),
                    ASTNode(NodeType.NUMBER, value=rng.randint(0, 10)),
                ],
            )

        # Vary branches
        then_type = rng.choice(["number", "symbol"])
        else_type = rng.choice(["number", "symbol"])

        then_node = (
            ASTNode(NodeType.NUMBER, value=rng.randint(1, 10))
            if then_type == "number"
            else ASTNode(NodeType.SYMBOL, value="a")
        )
        else_node = (
            ASTNode(NodeType.NUMBER, value=rng.randint(1, 10))
            if else_type == "number"
            else ASTNode(NodeType.SYMBOL, value="b")
        )

        root = ASTNode(NodeType.IF, children=[cond, then_node, else_node])

        from src.data.synthetic_gen import ProgramMetadata

        return root, ProgramMetadata(
            template_id=self.template_id,
            depth=2 if cond_type == "comparison" else 1,
            num_nodes=4 if cond_type == "symbol" else 7,
            num_operators=1 if cond_type == "comparison" else 0,
            has_recursion=False,
            has_variables=True,
            operator_types=[op] if cond_type == "comparison" else [],
        )
