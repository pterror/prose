"""Comprehensive evaluation metrics for graph denoising task."""

from typing import Any

import torch
from torch_geometric.data import Batch, Data

from src.data.asg_builder import ASGBuilder, ASTNode, EdgeType, NodeType


class DenoisingMetrics:
    """Compute evaluation metrics for graph denoising."""

    def __init__(self, num_node_types: int = 9) -> None:
        """
        Initialize metrics calculator.

        Args:
            num_node_types: Total number of node types (including MASK token)
        """
        self.num_node_types = num_node_types

    def compute_all(
        self, predictions: Data | Batch, ground_truth: Data | Batch
    ) -> dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            predictions: Predicted graphs (PyG Data or Batch)
            ground_truth: Ground truth graphs (PyG Data or Batch)

        Returns:
            Dictionary with metrics: exact_match, node_accuracy, edge_f1, syntax_valid
        """
        # Convert to lists of individual graphs if batched
        if isinstance(predictions, Batch):
            pred_graphs = predictions.to_data_list()
            gt_graphs = ground_truth.to_data_list()
        else:
            pred_graphs = [predictions]
            gt_graphs = [ground_truth]

        assert len(pred_graphs) == len(gt_graphs), "Mismatch in number of graphs"

        # Compute per-graph metrics
        exact_matches = 0
        total_nodes = 0
        correct_nodes = 0
        edge_tp = 0  # True positives
        edge_fp = 0  # False positives
        edge_fn = 0  # False negatives
        syntax_valid = 0

        for pred, gt in zip(pred_graphs, gt_graphs):
            # Exact match
            if self._is_exact_match(pred, gt):
                exact_matches += 1

            # Node accuracy
            node_stats = self._compute_node_accuracy(pred, gt)
            total_nodes += node_stats["total"]
            correct_nodes += node_stats["correct"]

            # Edge F1
            edge_stats = self._compute_edge_stats(pred, gt)
            edge_tp += edge_stats["tp"]
            edge_fp += edge_stats["fp"]
            edge_fn += edge_stats["fn"]

            # Syntax validity
            if self._is_syntax_valid(pred):
                syntax_valid += 1

        num_graphs = len(pred_graphs)

        # Aggregate metrics
        exact_match_rate = exact_matches / num_graphs if num_graphs > 0 else 0.0
        node_accuracy = correct_nodes / total_nodes if total_nodes > 0 else 0.0

        # Edge F1 = 2 * precision * recall / (precision + recall)
        precision = edge_tp / (edge_tp + edge_fp) if (edge_tp + edge_fp) > 0 else 0.0
        recall = edge_tp / (edge_tp + edge_fn) if (edge_tp + edge_fn) > 0 else 0.0
        edge_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        syntax_validity = syntax_valid / num_graphs if num_graphs > 0 else 0.0

        return {
            "exact_match": exact_match_rate,
            "node_accuracy": node_accuracy,
            "edge_f1": edge_f1,
            "edge_precision": precision,
            "edge_recall": recall,
            "syntax_valid": syntax_validity,
        }

    def _is_exact_match(self, pred: Data, gt: Data) -> bool:
        """Check if prediction exactly matches ground truth."""
        # Compare number of nodes
        if pred.x.size(0) != gt.x.size(0):
            return False

        # Compare node features
        if not torch.equal(pred.x, gt.x):
            return False

        # Compare edge structure
        if pred.edge_index.size(1) != gt.edge_index.size(1):
            return False

        # Sort edges for comparison (edges may be in different order)
        pred_edges = self._sort_edges(pred.edge_index, pred.edge_attr)
        gt_edges = self._sort_edges(gt.edge_index, gt.edge_attr)

        if not torch.equal(pred_edges[0], gt_edges[0]):
            return False
        if not torch.equal(pred_edges[1], gt_edges[1]):
            return False

        return True

    def _sort_edges(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sort edges by (src, dst, edge_type) for consistent comparison."""
        if edge_index.size(1) == 0:
            return edge_index, edge_attr

        # Create sortable keys: src * 1M + dst * 10 + edge_type
        keys = edge_index[0] * 1_000_000 + edge_index[1] * 10 + edge_attr
        sorted_indices = torch.argsort(keys)

        return edge_index[:, sorted_indices], edge_attr[sorted_indices]

    def _compute_node_accuracy(self, pred: Data, gt: Data) -> dict[str, int]:
        """Compute per-node classification accuracy."""
        # If sizes don't match, we can only compare up to the smaller size
        min_nodes = min(pred.x.size(0), gt.x.size(0))

        if min_nodes == 0:
            return {"total": 0, "correct": 0}

        # Compare node types (assuming x is node type indices)
        correct = torch.sum(pred.x[:min_nodes] == gt.x[:min_nodes]).item()

        # Penalize for size mismatch
        total = max(pred.x.size(0), gt.x.size(0))

        return {"total": total, "correct": correct}

    def _compute_edge_stats(self, pred: Data, gt: Data) -> dict[str, int]:
        """Compute edge true positives, false positives, false negatives."""
        # Convert edges to set of tuples for comparison
        pred_edges = self._edges_to_set(pred.edge_index, pred.edge_attr)
        gt_edges = self._edges_to_set(gt.edge_index, gt.edge_attr)

        tp = len(pred_edges & gt_edges)  # True positives (in both)
        fp = len(pred_edges - gt_edges)  # False positives (in pred, not in gt)
        fn = len(gt_edges - pred_edges)  # False negatives (in gt, not in pred)

        return {"tp": tp, "fp": fp, "fn": fn}

    def _edges_to_set(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> set[tuple[int, int, int]]:
        """Convert edge representation to set of (src, dst, edge_type) tuples."""
        if edge_index.size(1) == 0:
            return set()

        edges = set()
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            edge_type = edge_attr[i].item()
            edges.add((src, dst, edge_type))

        return edges

    def _is_syntax_valid(self, pred: Data) -> bool:
        """
        Check if predicted graph is syntactically valid Mini-Lisp.

        This attempts to reconstruct an AST from the predicted graph
        and validates it follows Mini-Lisp grammar rules.
        """
        try:
            # Convert PyG graph back to AST
            ast = self._pyg_to_ast(pred)

            # Validate AST structure
            return self._validate_ast(ast)

        except Exception:
            # Any error during reconstruction means invalid syntax
            return False

    def _pyg_to_ast(self, data: Data) -> ASTNode | None:
        """
        Reconstruct AST from PyG graph (reverse of ASGBuilder).

        This is a simplified reconstruction that assumes:
        - Node 0 is the root
        - Child edges define parent-child relationships
        - Sibling edges define ordering
        """
        if data.x.size(0) == 0:
            return None

        # Build adjacency lists for each edge type
        child_edges: dict[int, list[int]] = {}
        for i in range(data.edge_index.size(1)):
            src = data.edge_index[0, i].item()
            dst = data.edge_index[1, i].item()
            edge_type = data.edge_attr[i].item()

            # We only care about Child edges for AST reconstruction
            if edge_type == EdgeType.CHILD.value:
                if src not in child_edges:
                    child_edges[src] = []
                child_edges[src].append(dst)

        # Recursively build AST starting from root (node 0)
        return self._build_ast_node(0, data, child_edges)

    def _build_ast_node(
        self, node_id: int, data: Data, child_edges: dict[int, list[int]]
    ) -> ASTNode:
        """Recursively build AST node from graph."""
        # Get node type
        node_type_id = data.x[node_id].item()

        # Handle MASK token (invalid for syntax)
        if node_type_id >= len(NodeType):
            node_type = NodeType.SYMBOL  # Fallback
        else:
            try:
                node_type = NodeType(node_type_id)
            except ValueError:
                node_type = NodeType.SYMBOL

        # Build children recursively
        children = []
        if node_id in child_edges:
            for child_id in sorted(child_edges[node_id]):
                children.append(self._build_ast_node(child_id, data, child_edges))

        return ASTNode(node_type=node_type, value=None, children=children)

    def _validate_ast(self, ast: ASTNode | None) -> bool:
        """Validate AST follows Mini-Lisp grammar rules."""
        if ast is None:
            return False

        # Basic structural validation
        if ast.node_type == NodeType.DEFINE:
            # Define must have at least 2 children (name + body)
            return len(ast.children) >= 2

        if ast.node_type == NodeType.IF:
            # If must have 3 children (condition + then + else)
            return len(ast.children) == 3

        if ast.node_type == NodeType.LAMBDA:
            # Lambda must have 2 children (params + body)
            return len(ast.children) == 2

        if ast.node_type == NodeType.LET:
            # Let must have 2 children (bindings + body)
            return len(ast.children) == 2

        if ast.node_type == NodeType.LIST:
            # List can have any number of children
            return len(ast.children) >= 0

        if ast.node_type in (NodeType.SYMBOL, NodeType.NUMBER, NodeType.OPERATOR):
            # Leaf nodes should have no children
            return len(ast.children) == 0

        # Unknown node type
        return False


def compute_metrics(
    predictions: Data | Batch, ground_truth: Data | Batch, num_node_types: int = 9
) -> dict[str, float]:
    """
    Convenience function to compute all metrics.

    Args:
        predictions: Predicted graphs
        ground_truth: Ground truth graphs
        num_node_types: Total number of node types

    Returns:
        Dictionary with all metrics
    """
    metrics_calculator = DenoisingMetrics(num_node_types=num_node_types)
    return metrics_calculator.compute_all(predictions, ground_truth)
