"""Trajectory generation for iterative refinement training."""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.data.asg_builder import ASGBuilder
from src.data.dataset import TestCase
from src.runtime.interpreter import MiniLispInterpreter


@dataclass
class TrajectoryStep:
    """Single step in a refinement trajectory."""
    input_graph: Data  # Current state (corrupted)
    prev_graph: Data  # Previous iteration state
    target_graph: Data  # Ground truth (clean program)
    test_signals: torch.Tensor  # Test failure signals [num_nodes]
    iteration: int  # Current iteration (0-indexed)
    tests_passing: bool  # Whether all tests pass


class TrajectoryGenerator:
    """Generates training trajectories for iterative refinement."""

    def __init__(
        self,
        builder: ASGBuilder,
        interpreter: MiniLispInterpreter,
        mask_token_id: int = 0,
        scheduled_sampling_max: float = 0.95,
        scheduled_sampling_warmup: float = 0.5,
    ):
        """
        Initialize trajectory generator.

        Args:
            builder: ASGBuilder with vocabulary for creating graphs
            interpreter: MiniLispInterpreter for test execution
            mask_token_id: Token ID for masking (typically vocabulary.mask_token_id)
            scheduled_sampling_max: Maximum probability of using model predictions (0.95 = 95% model, 5% ground truth)
            scheduled_sampling_warmup: Initial probability at iteration 0 (0.5 = 50% model, 50% ground truth)
        """
        self.builder = builder
        self.interpreter = interpreter
        self.mask_token_id = mask_token_id
        self.scheduled_sampling_max = scheduled_sampling_max
        self.scheduled_sampling_warmup = scheduled_sampling_warmup

    def generate_trajectory(
        self,
        clean_graph: Data,
        tests: List[TestCase],
        corruption_rate: float = 0.5,
        max_iterations: int = 5,
        model: Optional[nn.Module] = None,
        epsilon: float = 0.3,
    ) -> List[TrajectoryStep]:
        """
        Generate a training trajectory by simulating iterative refinement.

        Args:
            clean_graph: Ground truth graph
            tests: Test cases for the program
            corruption_rate: Fraction of nodes to corrupt initially
            max_iterations: Maximum refinement iterations
            model: Optional model for predictions (if None, uses random policy)
            epsilon: Epsilon-greedy exploration rate (if model provided)

        Returns:
            List of trajectory steps (one per iteration)
        """
        trajectory = []

        # 1. Initial corruption
        current_graph = self._corrupt_graph(clean_graph, corruption_rate)
        prev_graph = current_graph.clone()

        # 2. Simulate refinement iterations
        for iteration in range(max_iterations):
            # Compute test signals from test execution
            test_signals = self._compute_test_signals(current_graph, tests)

            # Store trajectory step
            trajectory.append(TrajectoryStep(
                input_graph=current_graph.clone(),
                prev_graph=prev_graph.clone(),
                target_graph=clean_graph,
                test_signals=test_signals,
                iteration=iteration,
                tests_passing=False,  # Simplified for now
            ))

            # Generate next state using scheduled sampling
            # Compute sampling probability: increases from warmup to max over iterations
            use_model_prob = min(
                self.scheduled_sampling_max,
                self.scheduled_sampling_warmup + (iteration / max_iterations) * (1.0 - self.scheduled_sampling_warmup)
            )

            prev_graph = current_graph.clone()

            if model is not None and torch.rand(1).item() < use_model_prob:
                # Use model prediction (scheduled sampling)
                with torch.no_grad():
                    # Update iteration and test signals in features
                    current_graph.x[:, 4] = iteration
                    current_graph.x[:, 5] = test_signals

                    output = model.forward_full(current_graph, iteration=iteration)
                    predictions = output['logits'].argmax(dim=-1)

                    # Update token IDs with model predictions
                    current_graph.x[:, 1] = current_graph.x[:, 0]  # prev_token_id
                    current_graph.x[:, 0] = predictions.float()  # new token_id
            else:
                # Use ground truth (for stability and exploration)
                # Move toward clean graph incrementally
                current_tokens = current_graph.x[:, 0].long()
                target_tokens = clean_graph.x[:, 0].long()

                # Identify incorrect nodes
                incorrect_mask = (current_tokens != target_tokens)

                if incorrect_mask.any():
                    # Fix a random subset of incorrect nodes (gradual improvement)
                    incorrect_indices = incorrect_mask.nonzero(as_tuple=True)[0]
                    num_to_fix = max(1, len(incorrect_indices) // 2)  # Fix half
                    fix_indices = incorrect_indices[torch.randperm(len(incorrect_indices))[:num_to_fix]]

                    # Update with ground truth
                    current_graph.x[:, 1] = current_graph.x[:, 0]  # prev_token_id
                    current_graph.x[fix_indices, 0] = target_tokens[fix_indices].float()

            # Check if we've converged (simplified)
            if torch.allclose(current_graph.x[:, 0], clean_graph.x[:, 0]):
                # Mark final step as passing
                trajectory[-1] = TrajectoryStep(
                    input_graph=trajectory[-1].input_graph,
                    prev_graph=trajectory[-1].prev_graph,
                    target_graph=clean_graph,
                    test_signals=test_signals,
                    iteration=iteration,
                    tests_passing=True,
                )
                break

        return trajectory

    def _corrupt_graph(
        self,
        graph: Data,
        corruption_rate: float,
        keep_structure: bool = True
    ) -> Data:
        """
        Corrupt a graph by masking tokens.

        Args:
            graph: Original graph
            corruption_rate: Fraction of nodes to corrupt
            keep_structure: If True and rate < 0.9, preserve structural keywords

        Returns:
            Corrupted graph
        """
        corrupted = Data(
            x=graph.x.clone(),
            edge_index=graph.edge_index.clone(),
            edge_attr=graph.edge_attr.clone() if hasattr(graph, 'edge_attr') else None,
        )

        # Copy node_type if present
        if hasattr(graph, 'node_type'):
            corrupted.node_type = graph.node_type.clone()

        num_nodes = corrupted.x.size(0)

        if corruption_rate >= 0.9:
            # Full generation mode: corrupt all nodes
            corrupt_indices = list(range(num_nodes))
        else:
            # Partial corruption
            num_corrupt = max(1, int(num_nodes * corruption_rate))

            if keep_structure and hasattr(corrupted, 'node_type'):
                # Structural types: DEFINE=3, LAMBDA=4, IF=5, LET=6
                structural_types = {3, 4, 5, 6}
                non_structural = [
                    i for i in range(num_nodes)
                    if corrupted.node_type[i].item() not in structural_types
                ]
                if len(non_structural) >= num_corrupt:
                    import random
                    corrupt_indices = random.sample(non_structural, num_corrupt)
                else:
                    import random
                    corrupt_indices = random.sample(range(num_nodes), num_corrupt)
            else:
                import random
                corrupt_indices = random.sample(range(num_nodes), num_corrupt)

        # Mask selected nodes
        for idx in corrupt_indices:
            corrupted.x[idx, 0] = self.mask_token_id  # token_id
            corrupted.x[idx, 1] = self.mask_token_id  # prev_token_id

        return corrupted

    def _compute_test_signals(
        self,
        graph: Data,
        tests: List[TestCase],
    ) -> torch.Tensor:
        """
        Execute tests and compute Tarantula fault localization scores.

        Uses the Tarantula formula to compute suspiciousness scores for each node
        based on how many failing vs passing tests execute that node.

        Args:
            graph: Current program graph
            tests: Test cases to execute

        Returns:
            Tensor [num_nodes] with suspiciousness scores in [0, 1]
            Higher scores = more suspicious (likely buggy)
        """
        num_nodes = graph.x.size(0)

        # Track execution statistics per node
        from collections import defaultdict
        node_stats = defaultdict(lambda: {'failed': 0, 'passed': 0})
        total_failed = 0
        total_passed = 0

        # Try to reconstruct AST from graph
        try:
            from src.data.asg_reconstructor import ASGReconstructor, ReconstructionError

            reconstructor = ASGReconstructor(self.builder.vocabulary)
            ast_root, graph_to_ast_map = reconstructor.reconstruct(graph)
        except (ReconstructionError, Exception) as e:
            # Graph too corrupted to reconstruct - return zeros
            return torch.zeros(num_nodes)

        # Build reverse mapping: AST object ID → graph index
        ast_to_graph_map = {v: k for k, v in graph_to_ast_map.items()}

        # Execute all tests and collect statistics
        for test in tests:
            try:
                # Run test
                self.interpreter.trace_mode = False
                test_results = self.interpreter.run_tests(ast_root, [test])
                test_passed = test_results[0]

                # Trace which nodes were executed
                self.interpreter.trace_mode = True
                self.interpreter.traced_nodes.clear()
                self.interpreter.node_id_map.clear()

                traced_ast_ids = self.interpreter.trace_execution(ast_root, test.inputs)

                # Map AST object IDs back to graph indices
                traced_graph_indices = []
                for ast_id in traced_ast_ids:
                    if ast_id in ast_to_graph_map:
                        graph_idx = ast_to_graph_map[ast_id]
                        if 0 <= graph_idx < num_nodes:
                            traced_graph_indices.append(graph_idx)

                # Update statistics
                if test_passed:
                    total_passed += 1
                    for graph_idx in traced_graph_indices:
                        node_stats[graph_idx]['passed'] += 1
                else:
                    total_failed += 1
                    for graph_idx in traced_graph_indices:
                        node_stats[graph_idx]['failed'] += 1

            except Exception:
                # Test execution failed - skip this test
                continue
            finally:
                # Reset trace mode
                self.interpreter.trace_mode = False

        # Compute Tarantula suspiciousness scores
        test_signals = torch.zeros(num_nodes)

        if total_failed > 0:
            for graph_idx, stats in node_stats.items():
                failed = stats['failed']
                passed = stats['passed']

                # Tarantula formula:
                # suspiciousness = (failed/total_failed) / ((failed/total_failed) + (passed/total_passed))
                fail_rate = failed / total_failed
                pass_rate = passed / total_passed if total_passed > 0 else 0.0

                # Avoid division by zero
                denominator = fail_rate + pass_rate
                if denominator > 0:
                    suspiciousness = fail_rate / denominator
                else:
                    suspiciousness = 0.0

                test_signals[graph_idx] = suspiciousness

        return test_signals


@dataclass
class GuidedTrajectoryStep:
    """Single step in a guided refinement trajectory (with cross-attention)."""
    input_graph: Data  # Current state [num_nodes, 5] - no test_signal feature
    prev_graph: Data  # Previous iteration state
    target_graph: Data  # Ground truth (clean program)
    test_feedback: dict  # Test feedback for cross-attention
    iteration: int  # Current iteration (0-indexed)
    tests_passing: bool  # Whether all tests pass


class GuidedTrajectoryGenerator:
    """
    Generates training trajectories with test feedback for cross-attention guidance.

    This is the new approach that doesn't use test signals as node features,
    but instead passes them as separate guidance for cross-attention.
    """

    def __init__(
        self,
        builder: ASGBuilder,
        interpreter: MiniLispInterpreter,
        mask_token_id: int = 0,
        max_tests: int = 100,
        max_nodes: int = 1000,
        scheduled_sampling_max: float = 0.95,
        scheduled_sampling_warmup: float = 0.5,
    ):
        """
        Initialize guided trajectory generator.

        Args:
            builder: ASGBuilder with vocabulary
            interpreter: MiniLispInterpreter for test execution
            mask_token_id: Token ID for masking
            max_tests: Maximum number of tests (for padding)
            max_nodes: Maximum number of nodes (for trace encoding)
            scheduled_sampling_max: Maximum prob of using model predictions
            scheduled_sampling_warmup: Initial prob at iteration 0
        """
        self.builder = builder
        self.interpreter = interpreter
        self.mask_token_id = mask_token_id
        self.max_tests = max_tests
        self.max_nodes = max_nodes
        self.scheduled_sampling_max = scheduled_sampling_max
        self.scheduled_sampling_warmup = scheduled_sampling_warmup

    def generate_trajectory(
        self,
        clean_graph: Data,
        tests: List[TestCase],
        corruption_rate: float = 0.5,
        max_iterations: int = 5,
        model: Optional[nn.Module] = None,
    ) -> List[GuidedTrajectoryStep]:
        """
        Generate a training trajectory with test feedback for cross-attention.

        Args:
            clean_graph: Ground truth graph [num_nodes, 6]
            tests: Test cases
            corruption_rate: Fraction of nodes to corrupt
            max_iterations: Maximum refinement iterations
            model: Optional model for predictions (GuidedIterativeGraphUNet)

        Returns:
            List of guided trajectory steps
        """
        trajectory = []

        # 1. Initial corruption (5 features: token, prev_token, depth, sibling, iteration)
        current_graph = self._corrupt_graph(clean_graph, corruption_rate)
        prev_graph = current_graph.clone()

        # 2. Simulate refinement iterations
        for iteration in range(max_iterations):
            # Compute test feedback (format for cross-attention)
            # IMPORTANT: Skip test feedback at iteration 0 to avoid dependency during one-shot prediction
            if iteration == 0:
                test_feedback = None  # No test feedback for one-shot (iteration 0)
            else:
                test_feedback = self._compute_test_feedback(current_graph, tests)

            # Store trajectory step
            trajectory.append(GuidedTrajectoryStep(
                input_graph=current_graph.clone(),
                prev_graph=prev_graph.clone(),
                target_graph=clean_graph,
                test_feedback=test_feedback,
                iteration=iteration,
                tests_passing=False,
            ))

            # Generate next state using scheduled sampling
            use_model_prob = min(
                self.scheduled_sampling_max,
                self.scheduled_sampling_warmup + (iteration / max_iterations) * (1.0 - self.scheduled_sampling_warmup)
            )

            prev_graph = current_graph.clone()

            if model is not None and torch.rand(1).item() < use_model_prob:
                # Use model prediction
                with torch.no_grad():
                    # Update iteration feature
                    current_graph.x[:, 4] = iteration

                    # Move test feedback to same device as model (if present)
                    device = next(model.parameters()).device
                    if test_feedback is not None:
                        test_feedback_device = {
                            'test_ids': test_feedback['test_ids'].to(device),
                            'test_statuses': test_feedback['test_statuses'].to(device),
                            'test_traces': test_feedback['test_traces'].to(device),
                        }
                    else:
                        test_feedback_device = None  # No test feedback at iteration 0

                    output = model.forward_full(
                        current_graph,
                        iteration=iteration,
                        test_feedback=test_feedback_device
                    )
                    predictions = output['logits'].argmax(dim=-1)

                    # Update tokens
                    current_graph.x[:, 1] = current_graph.x[:, 0]  # prev_token_id
                    current_graph.x[:, 0] = predictions.float()  # new token_id
            else:
                # Use ground truth (gradual improvement)
                current_tokens = current_graph.x[:, 0].long()
                target_tokens = clean_graph.x[:, 0].long()

                incorrect_mask = (current_tokens != target_tokens)

                if incorrect_mask.any():
                    incorrect_indices = incorrect_mask.nonzero(as_tuple=True)[0]
                    num_to_fix = max(1, len(incorrect_indices) // 2)
                    fix_indices = incorrect_indices[torch.randperm(len(incorrect_indices))[:num_to_fix]]

                    current_graph.x[:, 1] = current_graph.x[:, 0]  # prev_token_id
                    current_graph.x[fix_indices, 0] = target_tokens[fix_indices].float()

            # Check convergence
            if torch.allclose(current_graph.x[:, 0], clean_graph.x[:, 0]):
                trajectory[-1] = GuidedTrajectoryStep(
                    input_graph=trajectory[-1].input_graph,
                    prev_graph=trajectory[-1].prev_graph,
                    target_graph=clean_graph,
                    test_feedback=test_feedback,
                    iteration=iteration,
                    tests_passing=True,
                )
                break

        return trajectory

    def _corrupt_graph(self, graph: Data, corruption_rate: float) -> Data:
        """
        Corrupt graph (5 features: token, prev_token, depth, sibling, iteration).

        NOTE: No test_signal feature in this version!
        """
        # Extract only first 5 features (remove test_signal if present)
        if graph.x.size(1) > 5:
            x_clean = graph.x[:, :5].clone()
        else:
            x_clean = graph.x.clone()

        corrupted = Data(
            x=x_clean,
            edge_index=graph.edge_index.clone(),
            edge_attr=graph.edge_attr.clone() if hasattr(graph, 'edge_attr') else None,
        )

        if hasattr(graph, 'node_type'):
            corrupted.node_type = graph.node_type.clone()

        num_nodes = corrupted.x.size(0)

        # Select nodes to corrupt
        if corruption_rate >= 0.9:
            corrupt_indices = list(range(num_nodes))
        else:
            num_corrupt = max(1, int(num_nodes * corruption_rate))
            import random
            corrupt_indices = random.sample(range(num_nodes), num_corrupt)

        # Mask selected nodes
        for idx in corrupt_indices:
            corrupted.x[idx, 0] = self.mask_token_id  # token_id
            corrupted.x[idx, 1] = self.mask_token_id  # prev_token_id

        return corrupted

    def _compute_test_feedback(
        self,
        graph: Data,
        tests: List[TestCase],
    ) -> dict:
        """
        Compute test feedback for cross-attention.

        Returns dict with:
            - 'test_ids': [num_tests] - Test identifiers (padded to max_tests)
            - 'test_statuses': [num_tests, 1] - Pass/fail (0=pass, 1=fail)
            - 'test_traces': [num_tests, max_nodes] - Execution trace masks
        """
        num_nodes = graph.x.size(0)
        num_tests = len(tests)

        # Initialize padded arrays
        test_ids = torch.full((self.max_tests,), -1, dtype=torch.long)  # -1 = padding
        test_statuses = torch.zeros(self.max_tests, 1)
        test_traces = torch.zeros(self.max_tests, self.max_nodes)

        # Try to reconstruct AST
        try:
            from src.data.asg_reconstructor import ASGReconstructor, ReconstructionError
            reconstructor = ASGReconstructor(self.builder.vocabulary)
            ast_root, graph_to_ast_map = reconstructor.reconstruct(graph)
        except (ReconstructionError, Exception):
            # Return empty feedback if reconstruction fails
            return {
                'test_ids': test_ids,
                'test_statuses': test_statuses,
                'test_traces': test_traces,
            }

        # Build reverse mapping
        ast_to_graph_map = {v: k for k, v in graph_to_ast_map.items()}

        # Execute each test
        for test_idx, test in enumerate(tests[:self.max_tests]):
            test_ids[test_idx] = test_idx

            try:
                # Run test
                self.interpreter.trace_mode = False
                test_results = self.interpreter.run_tests(ast_root, [test])

                # Record status (0=pass, 1=fail)
                test_passed = test_results[0]
                test_statuses[test_idx, 0] = 0.0 if test_passed else 1.0

                # If failed, trace execution
                if not test_passed:
                    self.interpreter.trace_mode = True
                    self.interpreter.traced_nodes.clear()
                    self.interpreter.node_id_map.clear()

                    traced_ast_ids = self.interpreter.trace_execution(ast_root, test.inputs)

                    # Map to graph indices and create trace mask
                    for ast_id in traced_ast_ids:
                        if ast_id in ast_to_graph_map:
                            graph_idx = ast_to_graph_map[ast_id]
                            if 0 <= graph_idx < min(num_nodes, self.max_nodes):
                                test_traces[test_idx, graph_idx] = 1.0

            except Exception:
                # Test execution failed - mark as failing
                test_statuses[test_idx, 0] = 1.0
                continue
            finally:
                self.interpreter.trace_mode = False

        return {
            'test_ids': test_ids,
            'test_statuses': test_statuses,
            'test_traces': test_traces,
        }


def corrupt_program_curriculum(
    program: Data,
    epoch: int,
    total_epochs: int = 50,
    mask_token_id: int = 0,
) -> Data:
    """
    Corrupt program with curriculum learning schedule.

    Corruption rate increases from 20% to 100% over training.

    Args:
        program: Clean program graph
        epoch: Current epoch (0-indexed)
        total_epochs: Total training epochs
        mask_token_id: Token ID for masking

    Returns:
        Corrupted program graph
    """
    # Curriculum stages (from phase1.5.md)
    # Stage 1 (0-5): 20%
    # Stage 2 (6-15): 50%
    # Stage 3 (16-25): 75%
    # Stage 4 (26-40): 90%
    # Stage 5 (41-50): 100%

    if epoch < 6:
        corruption_rate = 0.2
        keep_structure = True
    elif epoch < 16:
        corruption_rate = 0.5
        keep_structure = True
    elif epoch < 26:
        corruption_rate = 0.75
        keep_structure = True
    elif epoch < 41:
        corruption_rate = 0.9
        keep_structure = False
    else:
        corruption_rate = 1.0
        keep_structure = False

    # Create corrupted version
    corrupted = Data(
        x=program.x.clone(),
        edge_index=program.edge_index.clone(),
        edge_attr=program.edge_attr.clone() if hasattr(program, 'edge_attr') and program.edge_attr is not None else None,
    )

    if hasattr(program, 'node_type'):
        corrupted.node_type = program.node_type.clone()

    num_nodes = corrupted.x.size(0)

    if corruption_rate >= 0.9:
        corrupt_indices = list(range(num_nodes))
    else:
        num_corrupt = max(1, int(num_nodes * corruption_rate))

        if keep_structure and hasattr(corrupted, 'node_type'):
            structural_types = {3, 4, 5, 6}
            non_structural = [
                i for i in range(num_nodes)
                if corrupted.node_type[i].item() not in structural_types
            ]
            if len(non_structural) >= num_corrupt:
                import random
                corrupt_indices = random.sample(non_structural, num_corrupt)
            else:
                import random
                corrupt_indices = random.sample(range(num_nodes), num_corrupt)
        else:
            import random
            corrupt_indices = random.sample(range(num_nodes), num_corrupt)

    # Mask tokens
    for idx in corrupt_indices:
        corrupted.x[idx, 0] = mask_token_id
        corrupted.x[idx, 1] = mask_token_id

    return corrupted
