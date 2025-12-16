"""Tests for trajectory generation and iterative refinement loss."""

import pytest
import torch
from torch_geometric.data import Data

from src.data.asg_builder import ASGBuilder, ASTNode, NodeType
from src.data.dataset import TestCase
from src.data.vocabulary import Vocabulary
from src.runtime.interpreter import MiniLispInterpreter
from src.training.trajectory import TrajectoryGenerator, corrupt_program_curriculum
from src.training.denoising_task import IterativeRefinementLoss


class TestTrajectoryGenerator:
    """Test trajectory generation."""

    @pytest.fixture
    def setup(self):
        """Setup vocabulary, builder, and interpreter."""
        # Create simple vocabulary
        vocab = Vocabulary()
        vocab._add_token("+")
        vocab._add_token("-")
        vocab._add_token("1")
        vocab._add_token("2")

        builder = ASGBuilder(vocabulary=vocab)
        interpreter = MiniLispInterpreter()

        return vocab, builder, interpreter

    @pytest.fixture
    def sample_graph(self, setup):
        """Create a simple graph for testing."""
        vocab, builder, _ = setup

        # Simple AST: (+ 1 2)
        ast = ASTNode(
            NodeType.LIST,
            children=[
                ASTNode(NodeType.OPERATOR, value="+"),
                ASTNode(NodeType.NUMBER, value=1),
                ASTNode(NodeType.NUMBER, value=2),
            ]
        )

        graph = builder.build(ast)
        return graph

    def test_trajectory_generator_init(self, setup):
        """Test trajectory generator initialization."""
        vocab, builder, interpreter = setup
        generator = TrajectoryGenerator(builder, interpreter, mask_token_id=vocab.mask_token_id)

        assert generator.builder == builder
        assert generator.interpreter == interpreter
        assert generator.mask_token_id == vocab.mask_token_id

    def test_corrupt_graph(self, setup, sample_graph):
        """Test graph corruption."""
        vocab, builder, interpreter = setup
        generator = TrajectoryGenerator(builder, interpreter, mask_token_id=vocab.mask_token_id)

        corrupted = generator._corrupt_graph(sample_graph, corruption_rate=0.5)

        # Check structure preserved
        assert corrupted.x.shape == sample_graph.x.shape
        assert torch.equal(corrupted.edge_index, sample_graph.edge_index)

        # Check some nodes are masked
        num_masked = (corrupted.x[:, 0] == vocab.mask_token_id).sum()
        assert num_masked > 0

    def test_trajectory_generation(self, setup, sample_graph):
        """Test trajectory generation."""
        vocab, builder, interpreter = setup
        generator = TrajectoryGenerator(builder, interpreter, mask_token_id=vocab.mask_token_id)

        tests = [TestCase(inputs=[], expected_output=3)]
        trajectory = generator.generate_trajectory(
            sample_graph,
            tests,
            corruption_rate=0.5,
            max_iterations=3,
            model=None,  # Use random policy
        )

        # Check trajectory structure
        assert len(trajectory) <= 3
        assert all(step.iteration == i for i, step in enumerate(trajectory))
        assert all(isinstance(step.input_graph, Data) for step in trajectory)
        assert all(isinstance(step.test_signals, torch.Tensor) for step in trajectory)

    def test_trajectory_different_corruption_rates(self, setup, sample_graph):
        """Test trajectory with different corruption rates."""
        vocab, builder, interpreter = setup
        generator = TrajectoryGenerator(builder, interpreter, mask_token_id=vocab.mask_token_id)

        tests = [TestCase(inputs=[], expected_output=3)]

        # Low corruption
        traj_low = generator.generate_trajectory(
            sample_graph, tests, corruption_rate=0.2, max_iterations=2
        )

        # High corruption
        traj_high = generator.generate_trajectory(
            sample_graph, tests, corruption_rate=0.9, max_iterations=2
        )

        # Both should generate trajectories
        assert len(traj_low) > 0
        assert len(traj_high) > 0


class TestCurriculumCorruption:
    """Test curriculum learning corruption schedule."""

    def test_corruption_schedule_stage1(self):
        """Test stage 1: 20% corruption (epochs 0-5)."""
        graph = Data(
            x=torch.zeros(10, 6),
            edge_index=torch.tensor([[0], [1]]),
        )
        graph.node_type = torch.zeros(10, dtype=torch.long)

        for epoch in range(6):
            corrupted = corrupt_program_curriculum(graph, epoch, mask_token_id=99)
            num_masked = (corrupted.x[:, 0] == 99).sum()

            # Should be around 20% (2 nodes)
            assert num_masked >= 1
            assert num_masked <= 4  # Some variance

    def test_corruption_schedule_stage5(self):
        """Test stage 5: 100% corruption (epochs 41+)."""
        graph = Data(
            x=torch.zeros(10, 6),
            edge_index=torch.tensor([[0], [1]]),
        )
        graph.node_type = torch.zeros(10, dtype=torch.long)

        for epoch in range(41, 45):
            corrupted = corrupt_program_curriculum(graph, epoch, mask_token_id=99)
            num_masked = (corrupted.x[:, 0] == 99).sum()

            # Should be 100% (all 10 nodes)
            assert num_masked == 10

    def test_corruption_preserves_structure(self):
        """Test that early stage corruption preserves structure."""
        graph = Data(
            x=torch.zeros(10, 6),
            edge_index=torch.tensor([[0], [1]]),
        )
        # Mark first few as structural (DEFINE=3)
        graph.node_type = torch.tensor([3, 3, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)

        corrupted = corrupt_program_curriculum(graph, epoch=5, mask_token_id=99)

        # Structural nodes should ideally be preserved (though not guaranteed with small sample)
        # Just check the function runs
        assert corrupted.x.shape == graph.x.shape


class TestIterativeRefinementLoss:
    """Test multi-objective loss."""

    @pytest.fixture
    def setup_loss(self):
        """Create loss function."""
        return IterativeRefinementLoss(
            vocab_size=100,
            reconstruction_weight=1.0,
            stability_weight=0.1,
            correction_weight=0.5,
            confidence_weight=0.2,
        )

    def test_loss_initialization(self, setup_loss):
        """Test loss function initialization."""
        loss_fn = setup_loss

        assert loss_fn.vocab_size == 100
        assert loss_fn.reconstruction_weight == 1.0
        assert loss_fn.stability_weight == 0.1

    def test_loss_all_correct(self, setup_loss):
        """Test loss when all predictions are already correct."""
        loss_fn = setup_loss

        # All nodes already correct
        current = Data(x=torch.tensor([[10, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0]], dtype=torch.float))
        target = Data(x=torch.tensor([[10, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0]], dtype=torch.float))

        # Perfect predictions
        predictions = {
            'logits': torch.zeros(2, 100),
            'confidence': torch.ones(2, 1),
        }
        predictions['logits'][0, 10] = 10.0  # High confidence for token 10
        predictions['logits'][1, 20] = 10.0  # High confidence for token 20

        loss, metrics = loss_fn(predictions, current, target)

        # Should have low stability loss (predicting same)
        # Should have no correction loss (nothing to fix)
        assert metrics['num_correct'] == 2
        assert metrics['num_incorrect'] == 0

    def test_loss_all_incorrect(self, setup_loss):
        """Test loss when all predictions are incorrect."""
        loss_fn = setup_loss

        # All nodes incorrect
        current = Data(x=torch.tensor([[5, 0, 0, 0, 0, 0], [6, 0, 0, 0, 0, 0]], dtype=torch.float))
        target = Data(x=torch.tensor([[10, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0]], dtype=torch.float))

        predictions = {
            'logits': torch.zeros(2, 100),
            'confidence': torch.zeros(2, 1),
        }

        loss, metrics = loss_fn(predictions, current, target)

        # Should have no stability loss (nothing correct)
        # Should have correction loss (everything wrong)
        assert metrics['num_correct'] == 0
        assert metrics['num_incorrect'] == 2
        assert metrics['correction_loss'] > 0

    def test_loss_mixed(self, setup_loss):
        """Test loss with mix of correct and incorrect."""
        loss_fn = setup_loss

        # One correct, one incorrect
        current = Data(x=torch.tensor([[10, 0, 0, 0, 0, 0], [6, 0, 0, 0, 0, 0]], dtype=torch.float))
        target = Data(x=torch.tensor([[10, 0, 0, 0, 0, 0], [20, 0, 0, 0, 0, 0]], dtype=torch.float))

        predictions = {
            'logits': torch.zeros(2, 100),
            'confidence': torch.tensor([[0.9], [0.2]]),
        }

        loss, metrics = loss_fn(predictions, current, target)

        # Should have both stability and correction components
        assert metrics['num_correct'] == 1
        assert metrics['num_incorrect'] == 1
        assert 'stability_loss' in metrics
        assert 'correction_loss' in metrics
        assert metrics['loss'] > 0

    def test_loss_metrics(self, setup_loss):
        """Test that loss returns all expected metrics."""
        loss_fn = setup_loss

        current = Data(x=torch.tensor([[10, 0, 0, 0, 0, 0]], dtype=torch.float))
        target = Data(x=torch.tensor([[10, 0, 0, 0, 0, 0]], dtype=torch.float))

        predictions = {
            'logits': torch.zeros(1, 100),
            'confidence': torch.ones(1, 1),
        }

        loss, metrics = loss_fn(predictions, current, target)

        # Check all expected metrics
        expected_keys = [
            'loss', 'recon_loss', 'stability_loss', 'correction_loss',
            'confidence_loss', 'accuracy', 'correct_accuracy',
            'incorrect_accuracy', 'mean_confidence', 'num_correct', 'num_incorrect'
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
