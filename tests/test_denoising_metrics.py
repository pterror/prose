"""Tests for denoising and iterative refinement metrics."""

import pytest
import torch
from torch_geometric.data import Data

from src.training.denoising_metrics import IterativeRefinementMetrics


class TestIterativeRefinementMetrics:
    """Test Phase 1.5 iterative refinement metrics."""

    @pytest.fixture
    def metrics(self):
        """Create metrics calculator."""
        return IterativeRefinementMetrics(vocab_size=100)

    @pytest.fixture
    def sample_graphs(self):
        """Create sample current and target graphs."""
        # Target graph with 5 nodes
        target = Data(
            x=torch.tensor([
                [10, 0, 0, 0, 0, 0.0],
                [20, 0, 1, 0, 0, 0.0],
                [30, 0, 1, 1, 0, 0.0],
                [40, 0, 2, 0, 0, 0.0],
                [50, 0, 2, 1, 0, 0.0],
            ], dtype=torch.float),
            edge_index=torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.long),
        )

        # Current graph with some corruption (nodes 1 and 3 incorrect)
        current = target.clone()
        current.x[1, 0] = 99  # Wrong token
        current.x[3, 0] = 88  # Wrong token

        return current, target

    def test_metrics_initialization(self, metrics):
        """Test metrics calculator initialization."""
        assert metrics.vocab_size == 100

    def test_compute_iteration_metrics_perfect(self, metrics):
        """Test metrics when predictions are perfect."""
        # Create identical current and target
        target = Data(
            x=torch.tensor([
                [10, 0, 0, 0, 0, 0.0],
                [20, 0, 1, 0, 0, 0.0],
                [30, 0, 1, 1, 0, 0.0],
            ], dtype=torch.float)
        )
        current = target.clone()

        # Perfect predictions
        predictions = {
            'logits': torch.randn(3, 100),  # Dummy logits
            'confidence': torch.tensor([0.99, 0.98, 0.97]),
        }
        # Make logits predict correct tokens
        predictions['logits'][0, 10] = 10.0
        predictions['logits'][1, 20] = 10.0
        predictions['logits'][2, 30] = 10.0

        result = metrics.compute_iteration_metrics(current, target, predictions)

        # Should have perfect scores
        assert result['accuracy'] == 1.0
        assert result['recall'] == 1.0
        assert result['stability'] == 1.0
        assert result['num_correct'] == 3
        assert result['num_incorrect'] == 0
        assert result['num_changed'] == 0  # No changes needed

    def test_compute_iteration_metrics_with_errors(self, metrics, sample_graphs):
        """Test metrics with some incorrect predictions."""
        current, target = sample_graphs

        # Create predictions that fix node 1 but leave node 3 wrong
        predictions = {
            'logits': torch.randn(5, 100),
            'confidence': torch.tensor([0.95, 0.90, 0.88, 0.70, 0.92]),
        }
        # Set logits to predict specific tokens
        predictions['logits'][0, 10] = 10.0  # Correct (was already correct)
        predictions['logits'][1, 20] = 10.0  # Fixed (was 99, should be 20)
        predictions['logits'][2, 30] = 10.0  # Correct (was already correct)
        predictions['logits'][3, 77] = 10.0  # Still wrong (is 88, should be 40, predicting 77)
        predictions['logits'][4, 50] = 10.0  # Correct (was already correct)

        result = metrics.compute_iteration_metrics(current, target, predictions)

        # Check basic metrics
        assert 0.0 <= result['accuracy'] <= 1.0
        assert 0.0 <= result['precision'] <= 1.0
        assert 0.0 <= result['recall'] <= 1.0
        assert 0.0 <= result['f1'] <= 1.0
        assert 0.0 <= result['stability'] <= 1.0

        # Should have some correct and some incorrect
        assert result['num_correct'] > 0
        assert result['num_incorrect'] > 0

        # Confidence metrics should be in valid range
        assert 0.0 <= result['mean_confidence'] <= 1.0
        assert 0.0 <= result['correct_confidence'] <= 1.0
        assert 0.0 <= result['incorrect_confidence'] <= 1.0

    def test_compute_iteration_metrics_all_wrong(self, metrics):
        """Test metrics when all predictions are wrong."""
        target = Data(
            x=torch.tensor([
                [10, 0, 0, 0, 0, 0.0],
                [20, 0, 1, 0, 0, 0.0],
            ], dtype=torch.float)
        )
        current = Data(
            x=torch.tensor([
                [99, 0, 0, 0, 0, 0.0],
                [88, 0, 1, 0, 0, 0.0],
            ], dtype=torch.float)
        )

        # Predictions that don't match target
        predictions = {
            'logits': torch.randn(2, 100),
            'confidence': torch.tensor([0.50, 0.60]),
        }
        predictions['logits'][0, 77] = 10.0  # Wrong
        predictions['logits'][1, 55] = 10.0  # Wrong

        result = metrics.compute_iteration_metrics(current, target, predictions)

        # Should have zero accuracy
        assert result['accuracy'] == 0.0
        assert result['num_correct'] == 0
        assert result['num_incorrect'] == 2

    def test_compute_iteration_metrics_no_changes(self, metrics):
        """Test metrics when model makes no changes."""
        target = Data(
            x=torch.tensor([
                [10, 0, 0, 0, 0, 0.0],
                [20, 0, 1, 0, 0, 0.0],
            ], dtype=torch.float)
        )
        current = Data(
            x=torch.tensor([
                [99, 0, 0, 0, 0, 0.0],  # Wrong
                [20, 0, 1, 0, 0, 0.0],  # Correct
            ], dtype=torch.float)
        )

        # Predictions that keep current state
        predictions = {
            'logits': torch.randn(2, 100),
            'confidence': torch.tensor([0.80, 0.90]),
        }
        predictions['logits'][0, 99] = 10.0  # Keep wrong
        predictions['logits'][1, 20] = 10.0  # Keep correct

        result = metrics.compute_iteration_metrics(current, target, predictions)

        # Should have no changes
        assert result['num_changed'] == 0
        assert result['num_correct'] == 1
        assert result['num_incorrect'] == 1

    def test_compute_trajectory_metrics_empty(self, metrics):
        """Test trajectory metrics with empty history."""
        result = metrics.compute_trajectory_metrics([])
        assert result == {}

    def test_compute_trajectory_metrics_single_iteration(self, metrics):
        """Test trajectory metrics with single iteration."""
        history = [{
            'accuracy': 0.8,
            'f1': 0.75,
            'precision': 0.70,
            'recall': 0.80,
            'mean_confidence': 0.85,
            'num_changed': 5,
        }]

        result = metrics.compute_trajectory_metrics(history)

        assert result['initial_accuracy'] == 0.8
        assert result['final_accuracy'] == 0.8
        assert result['improvement'] == 0.0
        assert result['num_iterations'] == 1
        assert result['converged'] == False  # Had changes
        assert result['perfect'] == False  # Not 100%
        assert result['avg_confidence'] == 0.85
        assert result['final_f1'] == 0.75

    def test_compute_trajectory_metrics_improvement(self, metrics):
        """Test trajectory metrics showing improvement."""
        history = [
            {
                'accuracy': 0.3,
                'f1': 0.4,
                'precision': 0.5,
                'recall': 0.35,
                'mean_confidence': 0.60,
                'num_changed': 10,
            },
            {
                'accuracy': 0.6,
                'f1': 0.65,
                'precision': 0.70,
                'recall': 0.60,
                'mean_confidence': 0.75,
                'num_changed': 5,
            },
            {
                'accuracy': 0.9,
                'f1': 0.88,
                'precision': 0.85,
                'recall': 0.91,
                'mean_confidence': 0.92,
                'num_changed': 2,
            },
        ]

        result = metrics.compute_trajectory_metrics(history)

        assert result['initial_accuracy'] == 0.3
        assert result['final_accuracy'] == 0.9
        assert result['improvement'] == pytest.approx(0.6)  # 0.9 - 0.3
        assert result['num_iterations'] == 3
        assert result['converged'] == False  # Still had changes in last iteration
        assert result['perfect'] == False  # 0.9 < 1.0
        assert result['avg_confidence'] == pytest.approx((0.60 + 0.75 + 0.92) / 3)
        assert result['final_f1'] == 0.88
        assert result['final_precision'] == 0.85
        assert result['final_recall'] == 0.91

    def test_compute_trajectory_metrics_convergence(self, metrics):
        """Test trajectory metrics for convergence."""
        history = [
            {
                'accuracy': 0.7,
                'f1': 0.75,
                'precision': 0.80,
                'recall': 0.70,
                'mean_confidence': 0.85,
                'num_changed': 3,
            },
            {
                'accuracy': 0.95,
                'f1': 0.93,
                'precision': 0.92,
                'recall': 0.94,
                'mean_confidence': 0.96,
                'num_changed': 0,  # Converged
            },
        ]

        result = metrics.compute_trajectory_metrics(history)

        assert result['converged'] == True  # No changes in final iteration
        assert result['perfect'] == False  # 0.95 < 1.0
        assert result['improvement'] == 0.25

    def test_compute_trajectory_metrics_perfect_final(self, metrics):
        """Test trajectory metrics reaching perfect accuracy."""
        history = [
            {
                'accuracy': 0.6,
                'f1': 0.65,
                'precision': 0.70,
                'recall': 0.60,
                'mean_confidence': 0.80,
                'num_changed': 5,
            },
            {
                'accuracy': 1.0,  # Perfect!
                'f1': 1.0,
                'precision': 1.0,
                'recall': 1.0,
                'mean_confidence': 0.98,
                'num_changed': 4,
            },
        ]

        result = metrics.compute_trajectory_metrics(history)

        assert result['perfect'] == True  # 1.0 accuracy
        assert result['final_accuracy'] == 1.0
        assert result['improvement'] == 0.4

    def test_metrics_with_confidence_calibration(self, metrics):
        """Test that confidence metrics are properly tracked."""
        target = Data(
            x=torch.tensor([
                [10, 0, 0, 0, 0, 0.0],
                [20, 0, 1, 0, 0, 0.0],
                [30, 0, 1, 1, 0, 0.0],
            ], dtype=torch.float)
        )
        current = Data(
            x=torch.tensor([
                [10, 0, 0, 0, 0, 0.0],  # Correct
                [99, 0, 1, 0, 0, 0.0],  # Wrong
                [30, 0, 1, 1, 0, 0.0],  # Correct
            ], dtype=torch.float)
        )

        # High confidence on correct, low on incorrect
        predictions = {
            'logits': torch.randn(3, 100),
            'confidence': torch.tensor([0.95, 0.40, 0.98]),  # High, low, high
        }
        predictions['logits'][0, 10] = 10.0  # Predict correct
        predictions['logits'][1, 77] = 10.0  # Predict wrong
        predictions['logits'][2, 30] = 10.0  # Predict correct

        result = metrics.compute_iteration_metrics(current, target, predictions)

        # Correct predictions should have higher average confidence
        assert result['correct_confidence'] > result['incorrect_confidence']
        assert result['mean_confidence'] == pytest.approx((0.95 + 0.40 + 0.98) / 3)

    def test_precision_recall_calculation(self, metrics):
        """Test precision/recall calculation for changes."""
        target = Data(
            x=torch.tensor([
                [10, 0, 0, 0, 0, 0.0],  # Correct
                [20, 0, 1, 0, 0, 0.0],  # Wrong (current=99)
                [30, 0, 1, 1, 0, 0.0],  # Wrong (current=88)
                [40, 0, 2, 0, 0, 0.0],  # Correct
            ], dtype=torch.float)
        )
        current = Data(
            x=torch.tensor([
                [10, 0, 0, 0, 0, 0.0],  # Correct
                [99, 0, 1, 0, 0, 0.0],  # Wrong
                [88, 0, 1, 1, 0, 0.0],  # Wrong
                [40, 0, 2, 0, 0, 0.0],  # Correct
            ], dtype=torch.float)
        )

        # Model fixes one error (node 1) but breaks one correct (node 0)
        predictions = {
            'logits': torch.randn(4, 100),
            'confidence': torch.tensor([0.80, 0.85, 0.70, 0.90]),
        }
        predictions['logits'][0, 99] = 10.0  # Changed 10→99 (break correct)
        predictions['logits'][1, 20] = 10.0  # Changed 99→20 (fix error) ✓
        predictions['logits'][2, 88] = 10.0  # Keep 88 (still wrong)
        predictions['logits'][3, 40] = 10.0  # Keep 40 (still correct)

        result = metrics.compute_iteration_metrics(current, target, predictions)

        # Changed 2 nodes total: node 0 (broke) and node 1 (fixed)
        assert result['num_changed'] == 2
        # Precision: of 2 changes, 1 was good (node 1 fixed)
        # So precision = 1/2 = 0.5
        assert result['precision'] == pytest.approx(0.5, rel=0.01)
        # Recall: of 2 wrong nodes (1 and 2), we fixed 1
        # So recall = 1/2 = 0.5
        assert result['recall'] == pytest.approx(0.5, rel=0.01)
