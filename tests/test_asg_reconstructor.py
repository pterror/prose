"""Tests for ASG Reconstructor."""

import pytest
from pathlib import Path

from src.data.asg_builder import ASGBuilder, ASTNode, NodeType
from src.data.asg_reconstructor import ASGReconstructor, ReconstructionError
from src.data.vocabulary import Vocabulary


@pytest.fixture
def vocabulary():
    """Load test vocabulary."""
    vocab_path = Path("data/phase1_5/vocabulary.json")
    if not vocab_path.exists():
        pytest.skip("Vocabulary not found")
    return Vocabulary.load(vocab_path)


@pytest.fixture
def builder(vocabulary):
    """Create ASG builder with vocabulary."""
    return ASGBuilder(vocabulary=vocabulary)


@pytest.fixture
def reconstructor(vocabulary):
    """Create ASG reconstructor."""
    return ASGReconstructor(vocabulary=vocabulary)


def test_simple_addition_roundtrip(builder, reconstructor):
    """Test roundtrip: AST → ASG → AST for simple addition."""
    # Build AST: (+ 2 3)
    ast_root = ASTNode(
        node_type=NodeType.LIST,
        children=[
            ASTNode(node_type=NodeType.OPERATOR, value="+"),
            ASTNode(node_type=NodeType.NUMBER, value=2),
            ASTNode(node_type=NodeType.NUMBER, value=3),
        ]
    )

    # Convert to ASG
    graph = builder.build(ast_root)

    # Reconstruct AST
    reconstructed_ast, node_map = reconstructor.reconstruct(graph)

    # Verify structure
    assert reconstructed_ast.node_type == NodeType.LIST
    assert len(reconstructed_ast.children) == 3
    assert reconstructed_ast.children[0].node_type == NodeType.OPERATOR
    assert reconstructed_ast.children[0].value == "+"
    assert reconstructed_ast.children[1].node_type == NodeType.NUMBER
    assert reconstructed_ast.children[1].value == 2
    assert reconstructed_ast.children[2].node_type == NodeType.NUMBER
    assert reconstructed_ast.children[2].value == 3


def test_define_function_roundtrip(builder, reconstructor):
    """Test roundtrip: AST → ASG → AST for function definition."""
    # Build AST: (define (add a b) (+ a b))
    ast_root = ASTNode(
        node_type=NodeType.DEFINE,
        value="add",
        children=[
            # Function name and params
            ASTNode(node_type=NodeType.SYMBOL, value="add"),
            ASTNode(node_type=NodeType.SYMBOL, value="a"),
            ASTNode(node_type=NodeType.SYMBOL, value="b"),
            # Body: (+ a b)
            ASTNode(
                node_type=NodeType.LIST,
                children=[
                    ASTNode(node_type=NodeType.OPERATOR, value="+"),
                    ASTNode(node_type=NodeType.SYMBOL, value="a"),
                    ASTNode(node_type=NodeType.SYMBOL, value="b"),
                ]
            ),
        ]
    )

    # Convert to ASG
    graph = builder.build(ast_root)

    # Reconstruct AST
    reconstructed_ast, node_map = reconstructor.reconstruct(graph)

    # Verify structure (NOTE: function name "add" not in vocab, will be "define" or generic)
    assert reconstructed_ast.node_type == NodeType.DEFINE
    # Value will be "define" since that's what gets encoded
    assert reconstructed_ast.value == "define"
    assert len(reconstructed_ast.children) == 4

    # Check children have correct types
    assert reconstructed_ast.children[0].node_type == NodeType.SYMBOL  # func name (will be <UNK>)
    assert reconstructed_ast.children[1].node_type == NodeType.SYMBOL  # param a
    assert reconstructed_ast.children[1].value == "a"
    assert reconstructed_ast.children[2].node_type == NodeType.SYMBOL  # param b
    assert reconstructed_ast.children[2].value == "b"

    # Check body
    body = reconstructed_ast.children[3]
    assert body.node_type == NodeType.LIST
    assert len(body.children) == 3
    assert body.children[0].value == "+"
    assert body.children[1].value == "a"
    assert body.children[2].value == "b"


def test_node_id_mapping(builder, reconstructor):
    """Test that node_id_map is correctly built."""
    # Simple AST: (+ 1 2)
    ast_root = ASTNode(
        node_type=NodeType.LIST,
        children=[
            ASTNode(node_type=NodeType.OPERATOR, value="+"),
            ASTNode(node_type=NodeType.NUMBER, value=1),
            ASTNode(node_type=NodeType.NUMBER, value=2),
        ]
    )

    # Convert to ASG
    graph = builder.build(ast_root)
    num_nodes = graph.x.size(0)

    # Reconstruct
    reconstructed_ast, node_map = reconstructor.reconstruct(graph)

    # Verify mapping has all nodes
    assert len(node_map) == num_nodes

    # Verify each index maps to a valid python object ID
    for idx in range(num_nodes):
        assert idx in node_map
        assert isinstance(node_map[idx], int)  # Python object IDs are ints
        assert node_map[idx] > 0


def test_empty_graph_raises_error(reconstructor):
    """Test that empty graph raises ReconstructionError."""
    import torch
    from torch_geometric.data import Data

    # Create empty graph
    empty_graph = Data(
        x=torch.empty((0, 6)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0,), dtype=torch.long),
    )

    with pytest.raises(ReconstructionError, match="Empty graph"):
        reconstructor.reconstruct(empty_graph)


def test_if_expression_roundtrip(builder, reconstructor):
    """Test roundtrip for if expression."""
    # Build AST: (if (> x 0) 1 -1)
    ast_root = ASTNode(
        node_type=NodeType.IF,
        children=[
            # Condition: (> x 0)
            ASTNode(
                node_type=NodeType.LIST,
                children=[
                    ASTNode(node_type=NodeType.OPERATOR, value=">"),
                    ASTNode(node_type=NodeType.SYMBOL, value="x"),
                    ASTNode(node_type=NodeType.NUMBER, value=0),
                ]
            ),
            # Then: 1
            ASTNode(node_type=NodeType.NUMBER, value=1),
            # Else: -1
            ASTNode(node_type=NodeType.NUMBER, value=-1),
        ]
    )

    # Convert to ASG
    graph = builder.build(ast_root)

    # Reconstruct
    reconstructed_ast, node_map = reconstructor.reconstruct(graph)

    # Verify structure
    assert reconstructed_ast.node_type == NodeType.IF
    assert len(reconstructed_ast.children) == 3

    # Check condition
    condition = reconstructed_ast.children[0]
    assert condition.node_type == NodeType.LIST
    assert condition.children[0].value == ">"

    # Check branches
    assert reconstructed_ast.children[1].value == 1
    assert reconstructed_ast.children[2].value == -1
