"""Tests for iterative refinement inference."""

import pytest
import torch
from torch_geometric.data import Data

from src.inference.inference import IterativeRefinementInference, create_masked_graph
from src.models.graph_unet import IterativeGraphUNet
from src.runtime.interpreter import MiniLispInterpreter
from src.data.dataset import TestCase


class TestIterativeRefinementInference:
    """Test inference engine."""

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
    def inference(self, model):
        """Create inference engine."""
        interpreter = MiniLispInterpreter()
        return IterativeRefinementInference(
            model=model,
            interpreter=interpreter,
            mask_token_id=0,
            max_iterations=5,
            confidence_threshold=0.95,
        )

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph."""
        x = torch.tensor([
            [10, 0, 0, 0, 0, 0.0],
            [20, 0, 1, 0, 0, 0.0],
            [30, 0, 1, 1, 0, 0.0],
        ], dtype=torch.float)

        edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def test_inference_initialization(self, inference, model):
        """Test inference engine initialization."""
        assert inference.model == model
        assert inference.max_iterations == 5
        assert inference.confidence_threshold == 0.95

    def test_refine_program(self, inference, sample_graph):
        """Test program refinement."""
        # Create corrupted version
        corrupted = sample_graph.clone()
        corrupted.x[:, 0] = 0  # Mask all tokens

        # Refine toward target
        refined, metadata = inference.refine_program(
            corrupted,
            sample_graph,
            max_iterations=3,
            verbose=False,
        )

        # Check metadata structure
        assert 'iterations' in metadata
        assert 'final_accuracy' in metadata
        assert 'final_confidence' in metadata
        assert 'history' in metadata
        assert metadata['iterations'] <= 3

        # Check history
        assert len(metadata['history']) == metadata['iterations']
        for step in metadata['history']:
            assert 'iteration' in step
            assert 'accuracy' in step
            assert 'confidence' in step

    def test_early_stopping_high_confidence(self, model, sample_graph):
        """Test early stopping on high confidence."""
        # Create inference with very low threshold
        interpreter = MiniLispInterpreter()
        inference = IterativeRefinementInference(
            model=model,
            interpreter=interpreter,
            confidence_threshold=0.01,  # Very low - will stop early
            max_iterations=10,
        )

        corrupted = sample_graph.clone()
        corrupted.x[:, 0] = 0

        refined, metadata = inference.refine_program(
            corrupted,
            sample_graph,
            verbose=False,
        )

        # Should stop early due to confidence threshold
        assert metadata['iterations'] < 10

    def test_early_stopping_convergence(self, inference, sample_graph):
        """Test early stopping when no changes occur."""
        # Start with already correct graph
        refined, metadata = inference.refine_program(
            sample_graph,
            sample_graph,
            max_iterations=5,
            verbose=False,
        )

        # Should stop early (likely iteration 1 or 2)
        assert metadata['iterations'] <= 3
        assert metadata.get('converged', False) or metadata.get('perfect', False)

    def test_iteration_history_tracking(self, inference, sample_graph):
        """Test that iteration history is properly tracked."""
        corrupted = sample_graph.clone()
        corrupted.x[:, 0] = 0

        refined, metadata = inference.refine_program(
            corrupted,
            sample_graph,
            max_iterations=3,
        )

        history = metadata['history']
        assert len(history) > 0

        # Check each iteration has required fields
        for i, step in enumerate(history):
            assert step['iteration'] == i
            assert 0.0 <= step['accuracy'] <= 1.0
            assert 0.0 <= step['confidence'] <= 1.0
            assert step['num_correct'] >= 0
            assert step['num_changed'] >= 0

    def test_generate_from_tests(self, inference, sample_graph):
        """Test generation from tests (simplified)."""
        # Create masked graph
        masked = create_masked_graph(sample_graph, mask_token_id=0)

        # Mock test cases
        tests = [TestCase(inputs=[], expected_output=None)]

        # Generate
        generated, metadata = inference.generate_from_tests(
            masked,
            tests,
            verbose=False,
        )

        # Check output
        assert generated.x.shape == sample_graph.x.shape
        assert 'iterations' in metadata
        assert 'final_confidence' in metadata
        assert 'history' in metadata


class TestCreateMaskedGraph:
    """Test masked graph creation."""

    def test_full_mask(self):
        """Test creating fully masked graph."""
        original = Data(
            x=torch.tensor([
                [10, 20, 0, 0, 0, 0.0],
                [30, 40, 1, 0, 0, 0.0],
            ], dtype=torch.float),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        )

        masked = create_masked_graph(original, mask_token_id=99, full_mask=True)

        # All tokens should be masked
        assert (masked.x[:, 0] == 99).all()
        assert (masked.x[:, 1] == 99).all()

        # Other features should be initialized
        assert (masked.x[:, 4] == 0).all()  # iteration
        assert (masked.x[:, 5] == 0).all()  # test_signal

        # Structure should be preserved
        assert torch.equal(masked.edge_index, original.edge_index)

    def test_partial_mask_preserves_structure(self):
        """Test partial masking that preserves structural keywords."""
        original = Data(
            x=torch.tensor([
                [10, 20, 0, 0, 0, 0.0],
                [30, 40, 1, 0, 0, 0.0],
                [50, 60, 1, 1, 0, 0.0],
            ], dtype=torch.float),
            edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        )
        # Mark first node as structural (DEFINE=3)
        original.node_type = torch.tensor([3, 0, 0], dtype=torch.long)

        masked = create_masked_graph(original, mask_token_id=99, full_mask=False)

        # Structural node should be preserved
        assert masked.x[0, 0] != 99  # Not masked

        # Non-structural nodes should be masked
        assert masked.x[1, 0] == 99
        assert masked.x[2, 0] == 99

    def test_masked_graph_structure(self):
        """Test that masked graph preserves graph structure."""
        original = Data(
            x=torch.zeros(5, 6),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        )

        masked = create_masked_graph(original, mask_token_id=99)

        # Shape should match
        assert masked.x.shape == original.x.shape
        assert torch.equal(masked.edge_index, original.edge_index)


class TestInferenceIntegration:
    """Integration tests for inference."""

    def test_end_to_end_refinement(self):
        """Test complete refinement pipeline."""
        # Create model
        model = IterativeGraphUNet(
            vocab_size=50,
            hidden_channels=32,
            depth=2,
        )

        # Create inference
        interpreter = MiniLispInterpreter()
        inference = IterativeRefinementInference(
            model=model,
            interpreter=interpreter,
            max_iterations=3,
        )

        # Create target graph
        target = Data(
            x=torch.tensor([
                [10, 0, 0, 0, 0, 0.0],
                [20, 0, 1, 0, 0, 0.0],
            ], dtype=torch.float),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        )

        # Corrupt it
        corrupted = target.clone()
        corrupted.x[:, 0] = 0  # Mask all

        # Refine
        refined, metadata = inference.refine_program(
            corrupted,
            target,
            verbose=False,
        )

        # Should complete without errors
        assert refined.x.shape == target.x.shape
        assert metadata['iterations'] > 0
        assert len(metadata['history']) == metadata['iterations']

    def test_multiple_refinement_runs(self):
        """Test running refinement multiple times."""
        model = IterativeGraphUNet(vocab_size=50, hidden_channels=32, depth=2)
        interpreter = MiniLispInterpreter()
        inference = IterativeRefinementInference(model, interpreter, max_iterations=2)

        target = Data(
            x=torch.tensor([[10, 0, 0, 0, 0, 0.0]], dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

        # Run multiple times
        for _ in range(3):
            corrupted = target.clone()
            corrupted.x[:, 0] = 0

            refined, metadata = inference.refine_program(corrupted, target)

            # Each run should complete successfully
            assert metadata['iterations'] > 0
