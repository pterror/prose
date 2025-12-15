"""Visualization utilities for ASG reconstruction and evaluation."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.figure import Figure
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.data.asg_builder import EdgeType, NodeType


class ASGVisualizer:
    """Visualize Abstract Syntax Graphs and reconstructions."""

    # Color schemes for visualization
    NODE_COLORS = {
        NodeType.SYMBOL.value: "#87CEEB",  # Sky blue
        NodeType.NUMBER.value: "#98FB98",  # Pale green
        NodeType.LIST.value: "#FFE4B5",  # Moccasin
        NodeType.DEFINE.value: "#FFB6C1",  # Light pink
        NodeType.LAMBDA.value: "#DDA0DD",  # Plum
        NodeType.IF.value: "#F0E68C",  # Khaki
        NodeType.LET.value: "#E6E6FA",  # Lavender
        NodeType.OPERATOR.value: "#FFA07A",  # Light salmon
        8: "#D3D3D3",  # Light gray (MASK token)
    }

    EDGE_COLORS = {
        EdgeType.CHILD.value: "black",
        EdgeType.SIBLING.value: "blue",
        EdgeType.DATAFLOW.value: "red",
    }

    EDGE_STYLES = {
        EdgeType.CHILD.value: "solid",
        EdgeType.SIBLING.value: "dashed",
        EdgeType.DATAFLOW.value: "dotted",
    }

    def __init__(self) -> None:
        """Initialize visualizer."""
        self.node_type_labels = {
            NodeType.SYMBOL.value: "SYM",
            NodeType.NUMBER.value: "NUM",
            NodeType.LIST.value: "LIST",
            NodeType.DEFINE.value: "DEF",
            NodeType.LAMBDA.value: "λ",
            NodeType.IF.value: "IF",
            NodeType.LET.value: "LET",
            NodeType.OPERATOR.value: "OP",
            8: "MASK",
        }

    def visualize_reconstruction(
        self,
        corrupted: Data,
        prediction: Data,
        ground_truth: Data,
        output_path: Path,
        title: str = "ASG Reconstruction",
    ) -> None:
        """
        Create side-by-side visualization of corrupted, prediction, and ground truth.

        Args:
            corrupted: Corrupted input graph
            prediction: Model prediction
            ground_truth: Ground truth graph
            output_path: Path to save the figure
            title: Title for the figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Visualize each graph
        self._plot_graph(corrupted, axes[0], "Corrupted Input")
        self._plot_graph(prediction, axes[1], "Prediction")
        self._plot_graph(ground_truth, axes[2], "Ground Truth")

        # Add overall title
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Save figure
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def visualize_with_diff(
        self,
        prediction: Data,
        ground_truth: Data,
        output_path: Path,
        title: str = "Prediction vs Ground Truth",
    ) -> None:
        """
        Visualize prediction and ground truth with differences highlighted.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth graph
            output_path: Path to save the figure
            title: Title for the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Compute differences for highlighting
        node_errors = self._compute_node_errors(prediction, ground_truth)
        edge_errors = self._compute_edge_errors(prediction, ground_truth)

        # Plot with error highlighting
        self._plot_graph(
            prediction,
            axes[0],
            "Prediction (errors in red)",
            highlight_nodes=node_errors["wrong_nodes"],
            highlight_edges=edge_errors["wrong_edges"],
        )

        self._plot_graph(ground_truth, axes[1], "Ground Truth")

        fig.suptitle(title, fontsize=16, fontweight="bold")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_graph(
        self,
        data: Data,
        ax: plt.Axes,
        title: str,
        highlight_nodes: set[int] | None = None,
        highlight_edges: set[tuple[int, int]] | None = None,
    ) -> None:
        """Plot a single ASG on the given axes."""
        if highlight_nodes is None:
            highlight_nodes = set()
        if highlight_edges is None:
            highlight_edges = set()

        # Convert to NetworkX
        G = to_networkx(data, to_undirected=False)

        # Use hierarchical layout (dag-style)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            # Fallback to spring layout if graphviz not available
            pos = nx.spring_layout(G, k=1, iterations=50)

        # Prepare node colors
        node_colors = []
        for node_id in G.nodes():
            if node_id in highlight_nodes:
                node_colors.append("#FF6B6B")  # Red for errors
            else:
                node_type = data.x[node_id].item()
                node_colors.append(self.NODE_COLORS.get(node_type, "#CCCCCC"))

        # Prepare node labels
        labels = {}
        for node_id in G.nodes():
            node_type = data.x[node_id].item()
            label = self.node_type_labels.get(node_type, "?")
            labels[node_id] = f"{node_id}:{label}"

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax, alpha=0.9)

        # Draw node labels
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        # Draw edges by type
        if data.edge_index.size(1) > 0:
            for edge_type_val in [
                EdgeType.CHILD.value,
                EdgeType.SIBLING.value,
                EdgeType.DATAFLOW.value,
            ]:
                # Get edges of this type
                edge_mask = data.edge_attr == edge_type_val
                edges_of_type = []

                for i in range(data.edge_index.size(1)):
                    if edge_mask[i]:
                        src = data.edge_index[0, i].item()
                        dst = data.edge_index[1, i].item()
                        edges_of_type.append((src, dst))

                if edges_of_type:
                    # Determine edge color
                    edge_color = self.EDGE_COLORS[edge_type_val]

                    # Check if any edges should be highlighted
                    edge_colors = []
                    for edge in edges_of_type:
                        if edge in highlight_edges:
                            edge_colors.append("#FF6B6B")  # Red for errors
                        else:
                            edge_colors.append(edge_color)

                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=edges_of_type,
                        edge_color=edge_colors if highlight_edges else edge_color,
                        style=self.EDGE_STYLES[edge_type_val],
                        arrows=True,
                        arrowsize=15,
                        width=2,
                        ax=ax,
                        alpha=0.7,
                    )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    def _compute_node_errors(self, prediction: Data, ground_truth: Data) -> dict[str, Any]:
        """Compute which nodes are incorrectly predicted."""
        min_nodes = min(prediction.x.size(0), ground_truth.x.size(0))

        wrong_nodes = set()
        for i in range(min_nodes):
            if prediction.x[i].item() != ground_truth.x[i].item():
                wrong_nodes.add(i)

        # Nodes that exist in prediction but not in GT (or vice versa)
        if prediction.x.size(0) > ground_truth.x.size(0):
            for i in range(ground_truth.x.size(0), prediction.x.size(0)):
                wrong_nodes.add(i)

        return {"wrong_nodes": wrong_nodes}

    def _compute_edge_errors(self, prediction: Data, ground_truth: Data) -> dict[str, Any]:
        """Compute which edges are incorrectly predicted."""
        # Convert to sets for comparison
        pred_edges = set()
        for i in range(prediction.edge_index.size(1)):
            src = prediction.edge_index[0, i].item()
            dst = prediction.edge_index[1, i].item()
            edge_type = prediction.edge_attr[i].item()
            pred_edges.add((src, dst, edge_type))

        gt_edges = set()
        for i in range(ground_truth.edge_index.size(1)):
            src = ground_truth.edge_index[0, i].item()
            dst = ground_truth.edge_index[1, i].item()
            edge_type = ground_truth.edge_attr[i].item()
            gt_edges.add((src, dst, edge_type))

        # Find wrong edges (in prediction but not in GT)
        wrong_edges = {(src, dst) for src, dst, _ in (pred_edges - gt_edges)}

        return {"wrong_edges": wrong_edges}

    def render_mini_lisp(self, data: Data) -> str:
        """
        Render ASG as Mini-Lisp code string.

        Args:
            data: PyG graph data

        Returns:
            Mini-Lisp code as string
        """
        try:
            # Build adjacency list for Child edges
            child_edges: dict[int, list[int]] = {}
            for i in range(data.edge_index.size(1)):
                src = data.edge_index[0, i].item()
                dst = data.edge_index[1, i].item()
                edge_type = data.edge_attr[i].item()

                if edge_type == EdgeType.CHILD.value:
                    if src not in child_edges:
                        child_edges[src] = []
                    child_edges[src].append(dst)

            # Recursively render from root
            return self._render_node(0, data, child_edges)

        except Exception as e:
            return f"<Error: {str(e)}>"

    def _render_node(self, node_id: int, data: Data, child_edges: dict[int, list[int]]) -> str:
        """Recursively render a node as Mini-Lisp."""
        if node_id >= data.x.size(0):
            return "<invalid>"

        node_type = data.x[node_id].item()

        # Handle different node types
        if node_type == NodeType.SYMBOL.value:
            return "sym"  # Generic symbol (we don't store actual names)

        if node_type == NodeType.NUMBER.value:
            return "42"  # Generic number

        if node_type == NodeType.OPERATOR.value:
            return "+"  # Generic operator

        if node_type == 8:  # MASK token
            return "<MASK>"

        # For structural nodes, render as S-expressions
        children = []
        if node_id in child_edges:
            for child_id in sorted(child_edges[node_id]):
                children.append(self._render_node(child_id, data, child_edges))

        if node_type == NodeType.LIST.value:
            return f"({' '.join(children)})"

        if node_type == NodeType.DEFINE.value:
            return f"(define {' '.join(children)})"

        if node_type == NodeType.IF.value:
            if len(children) == 3:
                return f"(if {children[0]} {children[1]} {children[2]})"
            return f"(if {' '.join(children)})"

        if node_type == NodeType.LAMBDA.value:
            if len(children) >= 2:
                return f"(lambda {children[0]} {children[1]})"
            return f"(lambda {' '.join(children)})"

        if node_type == NodeType.LET.value:
            if len(children) >= 2:
                return f"(let {children[0]} {children[1]})"
            return f"(let {' '.join(children)})"

        return f"<unknown:{node_type}>"


def visualize_reconstruction(
    corrupted: Data,
    prediction: Data,
    ground_truth: Data,
    output_path: Path,
    title: str = "ASG Reconstruction",
) -> None:
    """
    Convenience function to visualize reconstruction.

    Args:
        corrupted: Corrupted input graph
        prediction: Model prediction
        ground_truth: Ground truth graph
        output_path: Path to save the figure
        title: Title for the figure
    """
    visualizer = ASGVisualizer()
    visualizer.visualize_reconstruction(corrupted, prediction, ground_truth, output_path, title)


def render_code(data: Data) -> str:
    """
    Convenience function to render ASG as Mini-Lisp code.

    Args:
        data: PyG graph data

    Returns:
        Mini-Lisp code string
    """
    visualizer = ASGVisualizer()
    return visualizer.render_mini_lisp(data)
