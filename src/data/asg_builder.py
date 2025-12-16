"""ASG (Abstract Syntax Graph) builder for Mini-Lisp programs."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import networkx as nx
import torch
from torch_geometric.data import Data


class NodeType(Enum):
    """AST node types for Mini-Lisp."""

    SYMBOL = 0
    NUMBER = 1
    LIST = 2
    DEFINE = 3
    LAMBDA = 4
    IF = 5
    LET = 6
    OPERATOR = 7


class EdgeType(Enum):
    """Edge types in the ASG."""

    CHILD = 0  # Parent-child syntactic relationship
    SIBLING = 1  # Sequential evaluation order
    DATAFLOW = 2  # Variable definition -> usage


@dataclass
class ASTNode:
    """Represents a node in the abstract syntax tree."""

    node_type: NodeType
    value: Any = None
    children: list["ASTNode"] | None = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []


class ASGBuilder:
    """Converts Mini-Lisp AST to Abstract Syntax Graph with DataFlow edges."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.symbol_table: dict[str, list[int]] = {}  # symbol -> node IDs where defined

    def build(self, ast_root: ASTNode) -> Data:
        """
        Convert AST to PyG Data object representing the ASG.

        Args:
            ast_root: Root node of the AST

        Returns:
            PyG Data object with node features and edges
        """
        self.graph.clear()
        self.node_counter = 0
        self.symbol_table.clear()

        # Build tree structure with Child and Sibling edges
        root_id = self._add_tree_edges(ast_root, parent_id=None)

        # Add DataFlow edges by analyzing variable scopes
        self._add_dataflow_edges(root_id)

        # Convert to PyG format
        return self._to_pyg_data()

    def _add_tree_edges(
        self, node: ASTNode, parent_id: int | None, depth: int = 0, sibling_index: int = 0
    ) -> int:
        """
        Recursively add nodes and Child/Sibling edges.

        Args:
            node: AST node to process
            parent_id: Parent node ID (None for root)
            depth: Depth in the tree (0 for root)
            sibling_index: Position among siblings (0-indexed)

        Returns:
            Node ID of the added node
        """
        node_id = self.node_counter
        self.node_counter += 1

        # Add node with features (including position encodings)
        self.graph.add_node(
            node_id,
            node_type=node.node_type.value,
            value=node.value,
            depth=depth,
            sibling_index=sibling_index,
        )

        # Add Child edge from parent
        if parent_id is not None:
            self.graph.add_edge(parent_id, node_id, edge_type=EdgeType.CHILD.value)

        # Track variable definitions
        if node.node_type in (NodeType.DEFINE, NodeType.LET, NodeType.LAMBDA):
            if node.value and isinstance(node.value, str):
                if node.value not in self.symbol_table:
                    self.symbol_table[node.value] = []
                self.symbol_table[node.value].append(node_id)

        # Process children (increment depth, track sibling index)
        prev_child_id = None
        for child_idx, child in enumerate(node.children or []):
            child_id = self._add_tree_edges(
                child, parent_id=node_id, depth=depth + 1, sibling_index=child_idx
            )

            # Add Sibling edge between consecutive children
            if prev_child_id is not None:
                self.graph.add_edge(prev_child_id, child_id, edge_type=EdgeType.SIBLING.value)

            prev_child_id = child_id

        return node_id

    def _add_dataflow_edges(self, root_id: int) -> None:
        """Add DataFlow edges from variable definitions to uses."""
        # Traverse graph to find variable usages
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]

            # Check if this is a variable reference (SYMBOL node)
            if node_data["node_type"] == NodeType.SYMBOL.value:
                symbol_name = node_data.get("value")

                if symbol_name and symbol_name in self.symbol_table:
                    # Find the closest definition in scope
                    # (Simplified: just link to most recent definition)
                    def_nodes = self.symbol_table[symbol_name]
                    if def_nodes:
                        # Add DataFlow edge from definition to usage
                        for def_id in def_nodes:
                            if def_id != node_id:  # Don't self-reference
                                self.graph.add_edge(
                                    def_id, node_id, edge_type=EdgeType.DATAFLOW.value
                                )

    def _to_pyg_data(self) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data object."""
        num_nodes = len(self.graph.nodes())

        # Create node feature matrix with position encodings
        # Shape: [num_nodes, 3] where columns are [node_type, depth, sibling_index]
        node_features = []
        for i in range(num_nodes):
            node_data = self.graph.nodes[i]
            node_features.append(
                [
                    node_data["node_type"],
                    node_data["depth"],
                    node_data["sibling_index"],
                ]
            )

        x = torch.tensor(node_features, dtype=torch.long)

        # Create edge index and edge attributes
        edge_list = list(self.graph.edges(data=True))
        if edge_list:
            edge_index = torch.tensor([[e[0], e[1]] for e in edge_list], dtype=torch.long).t()
            edge_attr = torch.tensor([e[2]["edge_type"] for e in edge_list], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def to_json(self, data: Data) -> dict[str, Any]:
        """Serialize PyG Data to JSON-serializable dict."""
        return {
            "num_nodes": data.x.size(0),
            "node_features": data.x.tolist(),
            "edge_index": data.edge_index.tolist(),
            "edge_attr": data.edge_attr.tolist(),
        }

    @staticmethod
    def from_json(json_data: dict[str, Any]) -> Data:
        """Deserialize JSON dict to PyG Data."""
        return Data(
            x=torch.tensor(json_data["node_features"], dtype=torch.long),
            edge_index=torch.tensor(json_data["edge_index"], dtype=torch.long),
            edge_attr=torch.tensor(json_data["edge_attr"], dtype=torch.long),
        )
