"""Unit tests for evaluation metrics."""

import pytest
import torch
from torch_geometric.data import Data, Batch

from src.data.asg_builder import EdgeType, NodeType
from src.training.denoising_metrics import DenoisingMetrics, compute_metrics


@pytest.fixture
def simple_graph():
    """Create a simple test graph."""
    # 3 nodes: SYMBOL, OPERATOR, NUMBER
    # 2 edges: CHILD from 0->1, CHILD from 0->2
    x = torch.tensor([NodeType.SYMBOL.value, NodeType.OPERATOR.value, NodeType.NUMBER.value])
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    edge_attr = torch.tensor([EdgeType.CHILD.value, EdgeType.CHILD.value])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


@pytest.fixture
def metrics_calculator():
    """Create metrics calculator instance."""
    return DenoisingMetrics(num_node_types=9)


class TestExactMatch:
    """Test exact match metric."""

    def test_identical_graphs_match(self, simple_graph, metrics_calculator):
        """Identical graphs should have exact match = 1.0."""
        result = metrics_calculator.compute_all(simple_graph, simple_graph)
        assert result["exact_match"] == 1.0

    def test_different_nodes_no_match(self, simple_graph, metrics_calculator):
        """Graphs with different nodes should not match."""
        # Create graph with different node types
        pred = simple_graph.clone()
        pred.x[0] = NodeType.NUMBER.value  # Change first node type

        result = metrics_calculator.compute_all(pred, simple_graph)
        assert result["exact_match"] == 0.0

    def test_different_edges_no_match(self, simple_graph, metrics_calculator):
        """Graphs with different edges should not match."""
        pred = simple_graph.clone()
        # Change edge type
        pred.edge_attr[0] = EdgeType.SIBLING.value

        result = metrics_calculator.compute_all(pred, simple_graph)
        assert result["exact_match"] == 0.0

    def test_batched_graphs(self, simple_graph, metrics_calculator):
        """Test exact match with batched graphs."""
        # Create batch with 2 identical graphs
        batch_pred = Batch.from_data_list([simple_graph, simple_graph])
        batch_gt = Batch.from_data_list([simple_graph, simple_graph])

        result = metrics_calculator.compute_all(batch_pred, batch_gt)
        assert result["exact_match"] == 1.0  # 100% match


class TestNodeAccuracy:
    """Test node accuracy metric."""

    def test_perfect_accuracy(self, simple_graph, metrics_calculator):
        """Perfect predictions should have 100% accuracy."""
        result = metrics_calculator.compute_all(simple_graph, simple_graph)
        assert result["node_accuracy"] == 1.0

    def test_partial_accuracy(self, simple_graph, metrics_calculator):
        """Partial correct predictions should have fractional accuracy."""
        pred = simple_graph.clone()
        pred.x[0] = NodeType.NUMBER.value  # 1 wrong out of 3

        result = metrics_calculator.compute_all(pred, simple_graph)
        # 2 correct out of 3 = 2/3 ≈ 0.667
        assert abs(result["node_accuracy"] - 2 / 3) < 0.01

    def test_size_mismatch_penalty(self, simple_graph, metrics_calculator):
        """Graphs with different sizes should be penalized."""
        # Create graph with extra node
        pred = Data(
            x=torch.tensor(
                [
                    NodeType.SYMBOL.value,
                    NodeType.OPERATOR.value,
                    NodeType.NUMBER.value,
                    NodeType.LIST.value,
                ]
            ),
            edge_index=simple_graph.edge_index,
            edge_attr=simple_graph.edge_attr,
        )

        result = metrics_calculator.compute_all(pred, simple_graph)
        # First 3 nodes match, but total is 4, so 3/4 = 0.75
        assert abs(result["node_accuracy"] - 0.75) < 0.01


class TestEdgeF1:
    """Test edge F1 score metric."""

    def test_perfect_edges(self, simple_graph, metrics_calculator):
        """Perfect edge predictions should have F1 = 1.0."""
        result = metrics_calculator.compute_all(simple_graph, simple_graph)
        assert result["edge_f1"] == 1.0
        assert result["edge_precision"] == 1.0
        assert result["edge_recall"] == 1.0

    def test_missing_edge(self, simple_graph, metrics_calculator):
        """Missing edge should reduce recall."""
        pred = Data(
            x=simple_graph.x,
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),  # Only first edge
            edge_attr=torch.tensor([EdgeType.CHILD.value]),
        )

        result = metrics_calculator.compute_all(pred, simple_graph)

        # TP=1, FP=0, FN=1
        # Precision = 1/(1+0) = 1.0
        # Recall = 1/(1+1) = 0.5
        # F1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 2/3 ≈ 0.667

        assert result["edge_precision"] == 1.0
        assert abs(result["edge_recall"] - 0.5) < 0.01
        assert abs(result["edge_f1"] - 2 / 3) < 0.01

    def test_extra_edge(self, simple_graph, metrics_calculator):
        """Extra edge should reduce precision."""
        pred = Data(
            x=simple_graph.x,
            edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
            edge_attr=torch.tensor(
                [EdgeType.CHILD.value, EdgeType.CHILD.value, EdgeType.SIBLING.value]
            ),
        )

        result = metrics_calculator.compute_all(pred, simple_graph)

        # TP=2, FP=1, FN=0
        # Precision = 2/(2+1) = 2/3
        # Recall = 2/(2+0) = 1.0
        # F1 = 2 * (2/3) * 1.0 / (2/3 + 1.0) = 4/5 = 0.8

        assert abs(result["edge_precision"] - 2 / 3) < 0.01
        assert result["edge_recall"] == 1.0
        assert abs(result["edge_f1"] - 0.8) < 0.01


class TestSyntaxValidity:
    """Test syntax validity metric."""

    def test_valid_simple_graph(self, simple_graph, metrics_calculator):
        """Simple valid graph should pass syntax check."""
        # Note: SYMBOL nodes should have no children
        valid_graph = Data(
            x=torch.tensor([NodeType.SYMBOL.value]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0,), dtype=torch.long),
        )

        result = metrics_calculator.compute_all(valid_graph, valid_graph)
        assert result["syntax_valid"] == 1.0

    def test_invalid_if_structure(self, metrics_calculator):
        """IF node without 3 children should be invalid."""
        # IF with only 2 children (invalid)
        invalid_graph = Data(
            x=torch.tensor([NodeType.IF.value, NodeType.SYMBOL.value, NodeType.NUMBER.value]),
            edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
            edge_attr=torch.tensor([EdgeType.CHILD.value, EdgeType.CHILD.value]),
        )

        # Use any valid graph as ground truth
        valid_graph = Data(
            x=torch.tensor([NodeType.SYMBOL.value]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0,), dtype=torch.long),
        )

        result = metrics_calculator.compute_all(invalid_graph, valid_graph)
        # Prediction fails syntax check
        assert result["syntax_valid"] == 0.0


class TestConvenienceFunction:
    """Test convenience function."""

    def test_compute_metrics(self, simple_graph):
        """Test the convenience function."""
        result = compute_metrics(simple_graph, simple_graph, num_node_types=9)

        assert "exact_match" in result
        assert "node_accuracy" in result
        assert "edge_f1" in result
        assert "syntax_valid" in result

        # All metrics should be perfect for identical graphs
        assert result["exact_match"] == 1.0
        assert result["node_accuracy"] == 1.0
        assert result["edge_f1"] == 1.0


class TestBatchedEvaluation:
    """Test evaluation on batched data."""

    def test_mixed_batch(self, simple_graph, metrics_calculator):
        """Test batch with both perfect and imperfect predictions."""
        # Create second graph (different from simple_graph)
        graph2 = Data(
            x=torch.tensor([NodeType.LIST.value, NodeType.DEFINE.value]),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.tensor([EdgeType.CHILD.value]),
        )

        # Batch: [simple_graph, graph2]
        batch_pred = Batch.from_data_list([simple_graph, simple_graph])
        batch_gt = Batch.from_data_list([simple_graph, graph2])

        result = metrics_calculator.compute_all(batch_pred, batch_gt)

        # First is exact match, second is not
        assert result["exact_match"] == 0.5  # 1/2 = 50%
