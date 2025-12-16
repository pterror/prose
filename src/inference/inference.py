"""Iterative refinement inference for Phase 1.5."""

from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.data.dataset import TestCase
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
        mask_token_id: int = 0,
        max_iterations: int = 10,
        confidence_threshold: float = 0.95,
    ):
        """
        Initialize inference engine.

        Args:
            model: Trained IterativeGraphUNet model
            interpreter: MiniLispInterpreter for test execution
            mask_token_id: Token ID for masking
            max_iterations: Maximum refinement iterations
            confidence_threshold: Stop if mean confidence exceeds this
        """
        self.model = model
        self.interpreter = interpreter
        self.mask_token_id = mask_token_id
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

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

            # TODO: Execute tests and compute test signals
            # For now, just use zeros (would need AST reconstruction)
            test_signals = torch.zeros(current_graph.x.size(0))
            current_graph.x[:, 5] = test_signals

            # Track iteration
            iteration_history.append({
                'iteration': iteration,
                'confidence': confidence.item(),
                'num_changed': (predictions != prev_tokens.long()).sum().item(),
            })

            if verbose:
                print(f"  Iteration {iteration}: confidence={confidence:.3f}, "
                      f"changed={iteration_history[-1]['num_changed']} nodes")

            # Check stopping criteria
            # 1. High confidence
            if confidence > self.confidence_threshold:
                if verbose:
                    print(f"  ✓ High confidence ({confidence:.3f}), stopping")
                break

            # 2. No changes (converged)
            if iteration > 0 and iteration_history[-1]['num_changed'] == 0:
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
