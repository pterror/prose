"""Unit tests for IterativeGraphUNet model."""

import pytest
import torch
from src.models.graph_unet import IterativeGraphUNet
from torch_geometric.data import Data


class TestIterativeGraphUNet:
    """Test IterativeGraphUNet architecture."""

    @pytest.fixture
    def model(self):
        """Create a small test model."""
        return IterativeGraphUNet(
            vocab_size=100,
            hidden_channels=64,
            depth=2,
            max_iterations=5,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample 6-feature graph data."""
        # 4 nodes with 6 features each
        # [token_id, prev_token_id, depth, sibling_index, iteration, test_signal]
        x = torch.tensor([
            [10, 0, 0, 0, 0, 0.0],  # Root
            [20, 0, 1, 0, 0, 0.0],  # Child 1
            [30, 0, 1, 1, 0, 0.5],  # Child 2 (with test signal)
            [40, 0, 1, 2, 0, 1.0],  # Child 3 (high test signal)
        ], dtype=torch.float)

        # Simple tree structure
        edge_index = torch.tensor([
            [0, 0, 0],  # From root
            [1, 2, 3],  # To children
        ], dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def test_model_initialization(self, model):
        """Test model initializes with correct parameters."""
        assert model.vocab_size == 100
        assert model.hidden_channels == 64
        assert model.depth == 2
        assert model.max_iterations == 5

        # Check embeddings exist
        assert model.token_embedding.num_embeddings == 100
        assert model.prev_token_embedding.num_embeddings == 100
        assert model.iteration_embedding.num_embeddings == 5

    def test_forward_full_output_shape(self, model, sample_data):
        """Test forward_full returns correct output shapes."""
        with torch.no_grad():
            output = model.forward_full(sample_data, iteration=0)

        assert 'logits' in output
        assert 'confidence' in output

        # Check shapes
        num_nodes = sample_data.x.size(0)
        assert output['logits'].shape == (num_nodes, 100)  # vocab_size = 100
        assert output['confidence'].shape == (num_nodes, 1)

    def test_forward_full_confidence_range(self, model, sample_data):
        """Test confidence values are in valid range [0, 1]."""
        with torch.no_grad():
            output = model.forward_full(sample_data, iteration=0)

        confidence = output['confidence']
        assert (confidence >= 0).all(), "Confidence should be >= 0"
        assert (confidence <= 1).all(), "Confidence should be <= 1"

    def test_forward_pooling(self, model, sample_data):
        """Test forward with pooling."""
        with torch.no_grad():
            output = model.forward(sample_data, iteration=0)

        # Output will be smaller due to pooling
        assert 'logits' in output
        assert 'confidence' in output

        # Should have fewer nodes after pooling
        assert output['logits'].size(0) <= sample_data.x.size(0)

    def test_different_iterations(self, model, sample_data):
        """Test model handles different iteration numbers."""
        with torch.no_grad():
            output_iter0 = model.forward_full(sample_data, iteration=0)
            output_iter3 = model.forward_full(sample_data, iteration=3)

        # Outputs should be different for different iterations
        # (Due to different iteration embeddings)
        assert not torch.allclose(output_iter0['logits'], output_iter3['logits'])

    def test_test_signal_influence(self, model):
        """Test that test signals affect model output."""
        # Create two identical graphs except for test_signal
        x_no_signal = torch.tensor([
            [10, 0, 0, 0, 0, 0.0],
            [20, 0, 1, 0, 0, 0.0],
        ], dtype=torch.float)

        x_with_signal = torch.tensor([
            [10, 0, 0, 0, 0, 0.0],
            [20, 0, 1, 0, 0, 1.0],  # High test signal
        ], dtype=torch.float)

        edge_index = torch.tensor([[0], [1]], dtype=torch.long)

        data_no_signal = Data(x=x_no_signal, edge_index=edge_index)
        data_with_signal = Data(x=x_with_signal, edge_index=edge_index)

        with torch.no_grad():
            output_no_signal = model.forward_full(data_no_signal)
            output_with_signal = model.forward_full(data_with_signal)

        # Outputs should differ due to test signal
        assert not torch.allclose(
            output_no_signal['logits'],
            output_with_signal['logits']
        )

    def test_batch_processing(self, model):
        """Test model can handle batched graphs."""
        # Create two small graphs
        x1 = torch.tensor([
            [10, 0, 0, 0, 0, 0.0],
            [20, 0, 1, 0, 0, 0.0],
        ], dtype=torch.float)

        x2 = torch.tensor([
            [30, 0, 0, 0, 0, 0.0],
            [40, 0, 1, 0, 0, 0.0],
        ], dtype=torch.float)

        edge_index1 = torch.tensor([[0], [1]], dtype=torch.long)
        edge_index2 = torch.tensor([[0], [1]], dtype=torch.long)

        # Batch them
        x = torch.cat([x1, x2], dim=0)
        edge_index = torch.cat([
            edge_index1,
            edge_index2 + 2  # Offset for second graph
        ], dim=1)
        batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, batch=batch)

        with torch.no_grad():
            output = model.forward_full(data)

        # Should handle batched data
        assert output['logits'].shape == (4, 100)

    def test_gradient_flow(self, model, sample_data):
        """Test gradients flow through the model."""
        output = model.forward_full(sample_data)

        # Create dummy target
        target = torch.randint(0, 100, (sample_data.x.size(0),))

        # Compute loss
        loss = torch.nn.functional.cross_entropy(output['logits'], target)

        # Backward pass
        loss.backward()

        # Check some parameters have gradients
        assert model.token_embedding.weight.grad is not None
        assert model.token_predictor.weight.grad is not None


class TestIterativeGraphUNetEdgeCases:
    """Test edge cases for IterativeGraphUNet."""

    def test_single_node_graph(self):
        """Test model handles single-node graph."""
        model = IterativeGraphUNet(vocab_size=50, hidden_channels=32, depth=2)

        x = torch.tensor([[5, 0, 0, 0, 0, 0.0]], dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges

        data = Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            output = model.forward_full(data)

        assert output['logits'].shape == (1, 50)
        assert output['confidence'].shape == (1, 1)

    def test_large_vocab_size(self):
        """Test model with large vocabulary."""
        model = IterativeGraphUNet(vocab_size=1000, hidden_channels=128, depth=2)

        x = torch.tensor([
            [500, 0, 0, 0, 0, 0.0],
            [999, 0, 1, 0, 0, 0.0],
        ], dtype=torch.float)

        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            output = model.forward_full(data)

        assert output['logits'].shape == (2, 1000)
