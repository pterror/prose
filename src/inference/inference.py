"""Iterative refinement inference for Phase 1.5."""

from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.data.dataset import TestCase
from src.data.asg_builder import ASGBuilder
from src.data.asg_reconstructor import ASGReconstructor, ReconstructionError
from src.runtime.interpreter import MiniLispInterpreter


class IterativeRefinementInference:
    """
    Inference engine for iterative refinement.

    Generates programs from tests using iterative refinement with test feedback.
    """

    def __init__(
        self,
        model: nn.Module,
        interpreter: MiniLispInterpreter,
        vocabulary,
        mask_token_id: int = 0,
        max_iterations: int = 10,
        confidence_threshold: float = 0.95,
        use_test_execution: bool = True,
    ):
        """
        Initialize inference engine.

        Args:
            model: Trained IterativeGraphUNet model
            interpreter: MiniLispInterpreter for test execution
            vocabulary: Vocabulary for ASG reconstruction
            mask_token_id: Token ID for masking
            max_iterations: Maximum refinement iterations
            confidence_threshold: Stop if mean confidence exceeds this
            use_test_execution: Use real test execution for stopping (recommended)
        """
        self.model = model
        self.interpreter = interpreter
        self.vocabulary = vocabulary
        self.mask_token_id = mask_token_id
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.use_test_execution = use_test_execution

        # Create reconstructor for ASG→AST conversion
        if use_test_execution:
            self.reconstructor = ASGReconstructor(vocabulary)

    def generate_from_tests(
        self,
        initial_graph: Data,
        tests: List[TestCase],
        verbose: bool = False,
    ) -> tuple[Data, dict]:
        """
        Generate program from tests using iterative refinement.

        Args:
            initial_graph: Initial graph (can be fully masked or partially filled)
            tests: Test cases to satisfy
            verbose: Print iteration progress

        Returns:
            (final_graph, metadata) tuple with:
                - final_graph: Refined program graph
                - metadata: Dict with iteration info, confidence, etc.
        """
        current_graph = initial_graph.clone()
        iteration_history = []

        for iteration in range(self.max_iterations):
            # Run model
            with torch.no_grad():
                output = self.model.forward_full(current_graph, iteration=iteration)

            predictions = output['logits'].argmax(dim=-1)
            confidence = output['confidence'].mean()

            # Update graph features
            # Store previous tokens
            prev_tokens = current_graph.x[:, 0].clone()

            # Update current tokens with predictions
            current_graph.x[:, 0] = predictions.float()
            current_graph.x[:, 1] = prev_tokens  # prev_token_id
            current_graph.x[:, 4] = iteration  # iteration number

            # Execute tests and compute test signals
            test_signals = torch.zeros(current_graph.x.size(0))
            tests_passing = False
            num_tests_passed = 0

            if self.use_test_execution and tests:
                test_signals, tests_passing, num_tests_passed = self._execute_tests(
                    current_graph, tests
                )

            current_graph.x[:, 5] = test_signals

            # Track iteration
            iteration_history.append({
                'iteration': iteration,
                'confidence': confidence.item(),
                'num_changed': (predictions != prev_tokens.long()).sum().item(),
                'tests_passed': num_tests_passed,
                'tests_total': len(tests) if tests else 0,
                'all_tests_passed': tests_passing,
            })

            if verbose:
                print(f"  Iteration {iteration}: confidence={confidence:.3f}, "
                      f"changed={iteration_history[-1]['num_changed']} nodes, "
                      f"tests={num_tests_passed}/{len(tests) if tests else 0}")

            # Check stopping criteria
            # 1. All tests pass (PRIMARY criterion)
            if self.use_test_execution and tests_passing:
                if verbose:
                    print(f"  ✓ All tests passing, stopping")
                break

            # 2. High confidence AND some tests passing
            if self.use_test_execution and confidence > self.confidence_threshold and num_tests_passed > 0:
                if verbose:
                    print(f"  ✓ High confidence ({confidence:.3f}) with {num_tests_passed}/{len(tests)} tests passing, stopping")
                break

            # 3. No changes for multiple iterations (only if not using test execution)
            # When using test execution, don't stop on "no changes" - keep trying!
            if not self.use_test_execution and iteration > 0 and iteration_history[-1]['num_changed'] == 0:
                if verbose:
                    print(f"  ✓ Converged (no changes), stopping")
                break

        metadata = {
            'iterations': iteration + 1,
            'final_confidence': confidence.item(),
            'history': iteration_history,
            'converged': iteration_history[-1]['num_changed'] == 0 if iteration_history else False,
        }

        return current_graph, metadata

    def refine_program(
        self,
        initial_graph: Data,
        target_graph: Data,
        max_iterations: Optional[int] = None,
        verbose: bool = False,
    ) -> tuple[Data, dict]:
        """
        Refine a corrupted program toward target.

        Useful for evaluation: start with corrupted program, refine toward clean.

        Args:
            initial_graph: Corrupted program graph
            target_graph: Ground truth (for evaluation only, not used by model)
            max_iterations: Override max iterations
            verbose: Print progress

        Returns:
            (refined_graph, metadata) tuple
        """
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        current_graph = initial_graph.clone()
        iteration_history = []

        for iteration in range(max_iter):
            # Run model
            with torch.no_grad():
                output = self.model.forward_full(current_graph, iteration=iteration)

            predictions = output['logits'].argmax(dim=-1)
            confidence = output['confidence'].mean()

            # Compute accuracy against target
            target_tokens = target_graph.x[:, 0].long()
            accuracy = (predictions == target_tokens).float().mean()

            # Update graph
            prev_tokens = current_graph.x[:, 0].clone()
            current_graph.x[:, 0] = predictions.float()
            current_graph.x[:, 1] = prev_tokens
            current_graph.x[:, 4] = iteration

            # Compute test signals from errors (nodes that don't match target)
            # In real setting, this would come from test execution
            test_signals = (predictions != target_tokens).float()
            current_graph.x[:, 5] = test_signals

            # Track iteration
            num_correct = (predictions == target_tokens).sum().item()
            num_changed = (predictions != prev_tokens.long()).sum().item()

            iteration_history.append({
                'iteration': iteration,
                'accuracy': accuracy.item(),
                'confidence': confidence.item(),
                'num_correct': num_correct,
                'num_changed': num_changed,
            })

            if verbose:
                print(f"  Iteration {iteration}: accuracy={accuracy:.3f}, "
                      f"confidence={confidence:.3f}, correct={num_correct}/{len(predictions)}")

            # Check stopping criteria
            # 1. Perfect accuracy
            if accuracy == 1.0:
                if verbose:
                    print(f"  ✓ Perfect accuracy, stopping")
                break

            # 2. High confidence
            if confidence > self.confidence_threshold:
                if verbose:
                    print(f"  ✓ High confidence ({confidence:.3f}), stopping")
                break

            # 3. No changes
            if num_changed == 0:
                if verbose:
                    print(f"  ✓ Converged (no changes), stopping")
                break

        metadata = {
            'iterations': iteration + 1,
            'final_accuracy': accuracy.item() if iteration_history else 0.0,
            'final_confidence': confidence.item() if iteration_history else 0.0,
            'history': iteration_history,
            'converged': iteration_history[-1]['num_changed'] == 0 if iteration_history else False,
            'perfect': iteration_history[-1]['accuracy'] == 1.0 if iteration_history else False,
        }

        return current_graph, metadata

    def _execute_tests(
        self,
        graph: Data,
        tests: List[TestCase],
    ) -> tuple[torch.Tensor, bool, int]:
        """
        Execute tests on reconstructed program and compute failure signals.

        Args:
            graph: Current program graph
            tests: Test cases to execute

        Returns:
            Tuple of (test_signals, all_tests_passed, num_tests_passed) where:
            - test_signals: Tensor [num_nodes] with 1.0 for nodes on failing test paths
            - all_tests_passed: True if all tests passed
            - num_tests_passed: Number of tests that passed
        """
        num_nodes = graph.x.size(0)
        test_signals = torch.zeros(num_nodes)

        # Try to reconstruct AST from graph
        try:
            ast_root, graph_to_ast_map = self.reconstructor.reconstruct(graph)
        except (ReconstructionError, Exception):
            # Graph too corrupted to reconstruct - return zeros
            return test_signals, False, 0

        # Build reverse mapping: AST object ID → graph index
        ast_to_graph_map = {v: k for k, v in graph_to_ast_map.items()}

        # Execute each test
        num_tests_passed = 0
        for test in tests:
            try:
                # Run test
                self.interpreter.trace_mode = False
                test_results = self.interpreter.run_tests(ast_root, [test])

                if test_results[0]:
                    # Test passed
                    num_tests_passed += 1
                else:
                    # Test failed - trace which nodes were executed
                    self.interpreter.trace_mode = True
                    self.interpreter.traced_nodes.clear()
                    self.interpreter.node_id_map.clear()

                    try:
                        traced_ast_ids = self.interpreter.trace_execution(
                            ast_root, test.inputs
                        )

                        # Map AST object IDs back to graph indices
                        for ast_id in traced_ast_ids:
                            if ast_id in ast_to_graph_map:
                                graph_idx = ast_to_graph_map[ast_id]
                                if 0 <= graph_idx < num_nodes:
                                    test_signals[graph_idx] = 1.0
                    except Exception:
                        # Tracing failed, skip
                        pass

            except Exception:
                # Test execution failed - count as failed test
                continue
            finally:
                # Reset trace mode
                self.interpreter.trace_mode = False

        all_tests_passed = num_tests_passed == len(tests)
        return test_signals, all_tests_passed, num_tests_passed


def create_masked_graph(
    template_graph: Data,
    mask_token_id: int = 0,
    full_mask: bool = True,
) -> Data:
    """
    Create a masked graph for generation from scratch.

    Args:
        template_graph: Graph to use as template (for structure)
        mask_token_id: Token ID to use for masking
        full_mask: If True, mask all tokens; if False, keep some structure

    Returns:
        Fully masked graph
    """
    masked = Data(
        x=template_graph.x.clone(),
        edge_index=template_graph.edge_index.clone(),
        edge_attr=template_graph.edge_attr.clone() if hasattr(template_graph, 'edge_attr') and template_graph.edge_attr is not None else None,
    )

    if hasattr(template_graph, 'node_type'):
        masked.node_type = template_graph.node_type.clone()

    # Mask tokens
    if full_mask:
        masked.x[:, 0] = mask_token_id  # token_id
        masked.x[:, 1] = mask_token_id  # prev_token_id
    else:
        # Keep structural keywords (DEFINE=3, LAMBDA=4, IF=5, LET=6)
        if hasattr(masked, 'node_type'):
            structural_types = {3, 4, 5, 6}
            for i in range(masked.x.size(0)):
                if masked.node_type[i].item() not in structural_types:
                    masked.x[i, 0] = mask_token_id
                    masked.x[i, 1] = mask_token_id

    # Initialize other features
    masked.x[:, 4] = 0  # iteration = 0
    masked.x[:, 5] = 0.0  # test_signal = 0

    return masked
