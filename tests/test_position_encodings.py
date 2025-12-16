"""Unit tests for position encodings in ASG builder."""

import pytest
import torch
from src.data.asg_builder import ASTNode, ASGBuilder, NodeType


class TestPositionEncodings:
    """Test position encoding calculation in ASG builder."""

    def test_root_node_position(self):
        """Root node should have depth=0, sibling_index=0."""
        # Create simple root node: SYMBOL
        ast = ASTNode(NodeType.SYMBOL, value="x")

        builder = ASGBuilder()
        data = builder.build(ast)

        # Node features: [node_type, depth, sibling_index]
        assert data.x.shape == (1, 3), "Node features should be 3D"
        assert data.x[0, 0].item() == NodeType.SYMBOL.value
        assert data.x[0, 1].item() == 0, "Root depth should be 0"
        assert data.x[0, 2].item() == 0, "Root sibling_index should be 0"

    def test_first_child_position(self):
        """First child should have depth=1, sibling_index=0."""
        # Create: (+ 1 2)
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="+"),
                ASTNode(NodeType.NUMBER, value=1),
                ASTNode(NodeType.NUMBER, value=2),
            ],
        )

        builder = ASGBuilder()
        data = builder.build(ast)

        # 4 nodes: LIST (root), OPERATOR, NUMBER, NUMBER
        assert data.x.shape == (4, 3)

        # Root: LIST
        assert data.x[0, 1].item() == 0, "Root depth = 0"
        assert data.x[0, 2].item() == 0, "Root sibling_index = 0"

        # First child: OPERATOR
        assert data.x[1, 0].item() == NodeType.OPERATOR.value
        assert data.x[1, 1].item() == 1, "First child depth = 1"
        assert data.x[1, 2].item() == 0, "First child sibling_index = 0"

    def test_second_child_position(self):
        """Second child should have depth=1, sibling_index=1."""
        # Create: (+ 1 2)
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="+"),
                ASTNode(NodeType.NUMBER, value=1),
                ASTNode(NodeType.NUMBER, value=2),
            ],
        )

        builder = ASGBuilder()
        data = builder.build(ast)

        # Second child: NUMBER (value=1)
        assert data.x[2, 0].item() == NodeType.NUMBER.value
        assert data.x[2, 1].item() == 1, "Second child depth = 1"
        assert data.x[2, 2].item() == 1, "Second child sibling_index = 1"

        # Third child: NUMBER (value=2)
        assert data.x[3, 0].item() == NodeType.NUMBER.value
        assert data.x[3, 1].item() == 1, "Third child depth = 1"
        assert data.x[3, 2].item() == 2, "Third child sibling_index = 2"

    def test_nested_structure(self):
        """Test nested structure with depth=2."""
        # Create: (+ (- 3 1) 2)
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="+"),
                ASTNode(
                    NodeType.LIST,
                    children=[
                        ASTNode(NodeType.OPERATOR, value="-"),
                        ASTNode(NodeType.NUMBER, value=3),
                        ASTNode(NodeType.NUMBER, value=1),
                    ],
                ),
                ASTNode(NodeType.NUMBER, value=2),
            ],
        )

        builder = ASGBuilder()
        data = builder.build(ast)

        # 7 nodes: LIST, OPERATOR(+), LIST, OPERATOR(-), NUMBER(3), NUMBER(1), NUMBER(2)
        assert data.x.shape[0] == 7

        # Root LIST: depth=0
        assert data.x[0, 1].item() == 0

        # OPERATOR(+): depth=1, sibling_index=0
        assert data.x[1, 1].item() == 1
        assert data.x[1, 2].item() == 0

        # Nested LIST: depth=1, sibling_index=1
        assert data.x[2, 0].item() == NodeType.LIST.value
        assert data.x[2, 1].item() == 1
        assert data.x[2, 2].item() == 1

        # OPERATOR(-): depth=2, sibling_index=0 (first child of nested LIST)
        assert data.x[3, 0].item() == NodeType.OPERATOR.value
        assert data.x[3, 1].item() == 2
        assert data.x[3, 2].item() == 0

        # NUMBER(3): depth=2, sibling_index=1
        assert data.x[4, 1].item() == 2
        assert data.x[4, 2].item() == 1

    def test_if_expression_positions(self):
        """Test IF expression with 3 children."""
        # Create: (if condition then-branch else-branch)
        ast = ASTNode(
            NodeType.IF,
            children=[
                ASTNode(NodeType.SYMBOL, value="condition"),
                ASTNode(NodeType.NUMBER, value=1),
                ASTNode(NodeType.NUMBER, value=2),
            ],
        )

        builder = ASGBuilder()
        data = builder.build(ast)

        # 4 nodes: IF, SYMBOL, NUMBER, NUMBER
        assert data.x.shape == (4, 3)

        # Condition: depth=1, sibling_index=0
        assert data.x[1, 1].item() == 1
        assert data.x[1, 2].item() == 0

        # Then branch: depth=1, sibling_index=1
        assert data.x[2, 1].item() == 1
        assert data.x[2, 2].item() == 1

        # Else branch: depth=1, sibling_index=2
        assert data.x[3, 1].item() == 1
        assert data.x[3, 2].item() == 2

    def test_define_expression_positions(self):
        """Test DEFINE expression with variable binding."""
        # Create: (define x 42)
        ast = ASTNode(
            NodeType.DEFINE,
            value="x",
            children=[
                ASTNode(NodeType.NUMBER, value=42),
            ],
        )

        builder = ASGBuilder()
        data = builder.build(ast)

        # 2 nodes: DEFINE, NUMBER
        assert data.x.shape == (2, 3)

        # DEFINE root: depth=0, sibling_index=0
        assert data.x[0, 0].item() == NodeType.DEFINE.value
        assert data.x[0, 1].item() == 0
        assert data.x[0, 2].item() == 0

        # NUMBER child: depth=1, sibling_index=0
        assert data.x[1, 0].item() == NodeType.NUMBER.value
        assert data.x[1, 1].item() == 1
        assert data.x[1, 2].item() == 0


class TestPositionEncodingEdgeCases:
    """Test edge cases for position encodings."""

    def test_single_node(self):
        """Single node graph should work correctly."""
        ast = ASTNode(NodeType.NUMBER, value=42)

        builder = ASGBuilder()
        data = builder.build(ast)

        assert data.x.shape == (1, 3)
        assert data.x[0, 1].item() == 0  # depth
        assert data.x[0, 2].item() == 0  # sibling_index

    def test_many_siblings(self):
        """Test with many siblings (e.g., 10 children)."""
        children = [ASTNode(NodeType.NUMBER, value=i) for i in range(10)]
        ast = ASTNode(NodeType.LIST, children=children)

        builder = ASGBuilder()
        data = builder.build(ast)

        # 11 nodes: LIST + 10 NUMBERs
        assert data.x.shape == (11, 3)

        # Check all sibling indices
        for i in range(10):
            node_idx = i + 1  # Skip root LIST
            assert data.x[node_idx, 1].item() == 1  # All at depth 1
            assert data.x[node_idx, 2].item() == i  # Sibling index = i

    def test_deep_nesting(self):
        """Test deeply nested structure (depth=5)."""
        # Create: (((((leaf)))))
        ast = ASTNode(NodeType.SYMBOL, value="leaf")
        for _ in range(5):
            ast = ASTNode(NodeType.LIST, children=[ast])

        builder = ASGBuilder()
        data = builder.build(ast)

        # 6 nodes: 5 LISTs + 1 SYMBOL
        assert data.x.shape == (6, 3)

        # Check depths: 0, 1, 2, 3, 4, 5
        expected_depths = [0, 1, 2, 3, 4, 5]
        actual_depths = [data.x[i, 1].item() for i in range(6)]
        assert actual_depths == expected_depths
