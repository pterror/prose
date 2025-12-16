"""ASG Reconstructor - converts PyG Data back to executable AST."""

from typing import Any

import torch
from torch_geometric.data import Data

from src.data.asg_builder import ASTNode, NodeType
from src.data.vocabulary import Vocabulary


class ReconstructionError(Exception):
    """Raised when ASG cannot be reconstructed into valid AST."""
    pass


class ASGReconstructor:
    """Converts PyG Data (ASG) back to executable ASTNode tree."""

    def __init__(self, vocabulary: Vocabulary):
        """
        Initialize reconstructor.

        Args:
            vocabulary: Vocabulary for decoding token_ids to tokens
        """
        self.vocabulary = vocabulary

    def reconstruct(self, graph: Data) -> tuple[ASTNode, dict[int, int]]:
        """
        Rebuild AST from graph.

        Args:
            graph: PyG Data object with ASG structure

        Returns:
            (ast_root, node_id_map) where:
            - ast_root: Reconstructed ASTNode tree
            - node_id_map: {graph_node_idx → python_object_id(ASTNode)}

        Raises:
            ReconstructionError: If graph structure is invalid
        """
        num_nodes = graph.x.size(0)

        if num_nodes == 0:
            raise ReconstructionError("Empty graph")

        # 1. Decode token_ids to tokens
        token_ids = graph.x[:, 0].long()
        tokens = [self.vocabulary.decode(tid.item()) for tid in token_ids]

        # 2. Get node types (preserved by ASGBuilder in graph.node_type)
        if hasattr(graph, 'node_type'):
            node_types = graph.node_type
        else:
            # Fallback: infer from tokens
            node_types = None

        # 3. Build adjacency dict from CHILD edges
        child_edges = self._extract_child_edges(graph)

        # 4. Find root (node with no incoming CHILD edges)
        root_idx = self._find_root(child_edges, num_nodes)

        # 5. Recursively build AST tree with DFS pre-order (same as ASGBuilder)
        node_id_map = {}
        ast_root = self._build_ast_recursive(
            node_idx=root_idx,
            tokens=tokens,
            node_types=node_types,
            child_edges=child_edges,
            node_id_map=node_id_map,
        )

        return ast_root, node_id_map

    def _extract_child_edges(self, graph: Data) -> dict[int, list[int]]:
        """
        Extract CHILD edges from graph.

        Returns:
            Dict mapping parent_idx → [child_idx, ...]
        """
        child_edges = {}

        if graph.edge_index.size(1) == 0:
            return child_edges

        edge_index = graph.edge_index
        edge_attr = graph.edge_attr

        # Filter for CHILD edges (edge_type == 0)
        for i in range(edge_index.size(1)):
            if edge_attr[i].item() == 0:  # EdgeType.CHILD
                parent_idx = edge_index[0, i].item()
                child_idx = edge_index[1, i].item()

                if parent_idx not in child_edges:
                    child_edges[parent_idx] = []

                child_edges[parent_idx].append(child_idx)

        # Sort children by sibling_index to maintain order
        for parent_idx in child_edges:
            children = child_edges[parent_idx]
            # Get sibling_index for each child (feature [3])
            children_with_indices = [
                (child_idx, graph.x[child_idx, 3].item())
                for child_idx in children
            ]
            # Sort by sibling_index
            children_with_indices.sort(key=lambda x: x[1])
            child_edges[parent_idx] = [child_idx for child_idx, _ in children_with_indices]

        return child_edges

    def _find_root(self, child_edges: dict[int, list[int]], num_nodes: int) -> int:
        """
        Find root node (has no incoming CHILD edges).

        Returns:
            Root node index

        Raises:
            ReconstructionError: If no root found or multiple roots
        """
        # Collect all nodes that are children
        children_set = set()
        for children in child_edges.values():
            children_set.update(children)

        # Root is the node that's not a child of any other node
        roots = []
        for i in range(num_nodes):
            if i not in children_set:
                roots.append(i)

        if len(roots) == 0:
            raise ReconstructionError("No root node found (cyclic graph?)")
        elif len(roots) > 1:
            # Multiple roots - take the one with lowest index (should be first in DFS)
            return min(roots)

        return roots[0]

    def _build_ast_recursive(
        self,
        node_idx: int,
        tokens: list[str],
        node_types: torch.Tensor | None,
        child_edges: dict[int, list[int]],
        node_id_map: dict[int, int],
    ) -> ASTNode:
        """
        Recursively build AST from node index.

        Args:
            node_idx: Current node index in graph
            tokens: List of decoded tokens
            node_types: Tensor of node types (if available)
            child_edges: Parent → children mapping
            node_id_map: Accumulates graph_idx → python_id mapping

        Returns:
            ASTNode for this subtree
        """
        token = tokens[node_idx]

        # Use preserved node_type if available, otherwise infer
        if node_types is not None:
            node_type = NodeType(node_types[node_idx].item())
            value = self._token_to_value(token, node_type)
        else:
            node_type, value = self._token_to_node_type_and_value(token)

        # Create ASTNode
        ast_node = ASTNode(node_type=node_type, value=value, children=[])

        # Track mapping BEFORE recursing (pre-order)
        node_id_map[node_idx] = id(ast_node)

        # Recursively build children
        if node_idx in child_edges:
            for child_idx in child_edges[node_idx]:
                child_ast = self._build_ast_recursive(
                    node_idx=child_idx,
                    tokens=tokens,
                    node_types=node_types,
                    child_edges=child_edges,
                    node_id_map=node_id_map,
                )
                ast_node.children.append(child_ast)

        return ast_node

    def _token_to_value(self, token: str, node_type: NodeType) -> Any:
        """
        Extract value from token given known node_type.

        Args:
            token: Decoded token string
            node_type: Known NodeType from graph

        Returns:
            Value for the ASTNode
        """
        # Structural keywords - use the keyword as value
        if node_type in (NodeType.DEFINE, NodeType.LAMBDA, NodeType.IF, NodeType.LET):
            return token if token != "<UNK>" else node_type.name.lower()

        # LIST nodes have no value
        elif node_type == NodeType.LIST:
            return None

        # Operators - use the operator symbol
        elif node_type == NodeType.OPERATOR:
            return token if token != "<UNK>" else "+"

        # Numbers - parse as int or float
        elif node_type == NodeType.NUMBER:
            if token == "<UNK>":
                return 0
            try:
                return int(token)
            except ValueError:
                try:
                    return float(token)
                except ValueError:
                    return 0

        # Symbols - use token as symbol name
        elif node_type == NodeType.SYMBOL:
            return token if token != "<UNK>" else "x"

        # Fallback
        else:
            return token

    def _token_to_node_type_and_value(self, token: str) -> tuple[NodeType, Any]:
        """
        Infer NodeType and value from token string.

        Args:
            token: Token string (e.g., "define", "+", "42", "x")

        Returns:
            (NodeType, value) tuple
        """
        # Structural keywords
        if token == "define":
            return NodeType.DEFINE, "define"
        elif token == "lambda":
            return NodeType.LAMBDA, "lambda"
        elif token == "if":
            return NodeType.IF, "if"
        elif token == "let":
            return NodeType.LET, "let"
        elif token == "(":
            return NodeType.LIST, None

        # Operators
        elif token in ["+", "-", "*", "/", "<", ">", "="]:
            return NodeType.OPERATOR, token

        # Numbers
        elif self._is_number(token):
            try:
                value = int(token)
            except ValueError:
                value = float(token)
            return NodeType.NUMBER, value

        # Symbols (variables, function names, parameters)
        else:
            return NodeType.SYMBOL, token

    def _is_number(self, token: str) -> bool:
        """Check if token is a number."""
        try:
            float(token)
            return True
        except ValueError:
            return False
