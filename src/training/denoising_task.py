"""Training infrastructure for Graph U-Net."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data


class DenoisingLoss(nn.Module):
    """Loss function for graph denoising task."""

    def __init__(self, num_node_types: int) -> None:
        super().__init__()
        self.num_node_types = num_node_types

    def forward(
        self, predictions: torch.Tensor, corrupted: Data, targets: Data, mask_token_id: int = 8
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute denoising loss.

        Args:
            predictions: Predicted node type logits [num_nodes, num_node_types]
            corrupted: Corrupted graph Data object (to identify masked nodes)
            targets: Original graph Data object with x[:, 0] = node types
            mask_token_id: ID of mask token (nodes to predict)

        Returns:
            (loss, metrics_dict) tuple
        """
        # Extract node type IDs from first column of node features
        # targets.x shape: [num_nodes, 3] where columns are [node_type, depth, sibling_index]
        # OR [num_nodes] if corrupted (1D)
        if targets.x.dim() == 1:
            target_labels = targets.x.long()
        else:
            target_labels = targets.x[:, 0].long()

        # Identify masked nodes from corrupted graph
        if corrupted.x.dim() == 1:
            corrupted_types = corrupted.x.long()
        else:
            corrupted_types = corrupted.x[:, 0].long()

        masked_nodes_mask = corrupted_types == mask_token_id

        # Filter to only masked nodes
        masked_predictions = predictions[masked_nodes_mask]
        masked_target_labels = target_labels[masked_nodes_mask]

        # If no nodes  are masked, return 0 loss
        if masked_predictions.numel() == 0:
            return torch.tensor(0.0, device=predictions.device), {"loss": 0.0, "accuracy": 1.0}

        # Compute loss only on masked nodes
        node_loss = F.cross_entropy(masked_predictions, masked_target_labels)

        # Compute accuracy only on masked nodes
        predicted_labels = masked_predictions.argmax(dim=-1)
        accuracy = (predicted_labels == masked_target_labels).float().mean()

        metrics = {
            "loss": node_loss.item(),
            "accuracy": accuracy.item(),
        }

        return node_loss, metrics


def collate_graph_pairs(
    batch: list[tuple[Data, Data]],
) -> tuple[Batch, Batch]:
    """
    Custom collate function for (corrupted, original) graph pairs.

    Args:
        batch: List of (corrupted, original) tuples

    Returns:
        (corrupted_batch, original_batch) as PyG Batch objects
    """
    from torch_geometric.data import Batch

    corrupted_graphs = [item[0] for item in batch]
    original_graphs = [item[1] for item in batch]

    corrupted_batch = Batch.from_data_list(corrupted_graphs)
    original_batch = Batch.from_data_list(original_graphs)

    return corrupted_batch, original_batch


class IterativeRefinementLoss(nn.Module):
    """
    Multi-objective loss for iterative refinement (Phase 1.5).

    Combines:
    1. Reconstruction: predict correct tokens
    2. Stability: don't change already-correct nodes
    3. Correction: fix incorrect nodes (higher weight)
    4. Confidence: calibrate confidence scores
    """

    def __init__(
        self,
        vocab_size: int,
        reconstruction_weight: float = 1.0,
        stability_weight: float = 0.1,
        correction_weight: float = 0.5,
        confidence_weight: float = 0.2,
        test_following_weight: float = 0.3,
    ):
        """
        Initialize multi-objective loss.

        Args:
            vocab_size: Size of token vocabulary
            reconstruction_weight: Weight for overall reconstruction loss
            stability_weight: Weight for stability loss (don't change correct)
            correction_weight: Weight for correction loss (fix incorrect)
            confidence_weight: Weight for confidence calibration
            test_following_weight: Weight for test-following loss (change failing nodes)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.reconstruction_weight = reconstruction_weight
        self.stability_weight = stability_weight
        self.correction_weight = correction_weight
        self.confidence_weight = confidence_weight
        self.test_following_weight = test_following_weight

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        current_graph: Data,
        target_graph: Data,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute multi-objective loss.

        Args:
            predictions: Dict with 'logits' [num_nodes, vocab_size] and
                        'confidence' [num_nodes, 1]
            current_graph: Current corrupted state
            target_graph: Ground truth graph

        Returns:
            (total_loss, metrics_dict)
        """
        logits = predictions['logits']
        confidence = predictions['confidence'].squeeze(-1)  # Only squeeze last dim

        # Extract token IDs (feature 0)
        current_tokens = current_graph.x[:, 0].long()
        target_tokens = target_graph.x[:, 0].long()

        # 1. Reconstruction loss (standard cross-entropy)
        recon_loss = F.cross_entropy(logits, target_tokens)

        # 2. Identify correct vs incorrect nodes
        correct_mask = (current_tokens == target_tokens)
        incorrect_mask = ~correct_mask

        # 3. Stability loss: penalize changing already-correct nodes
        if correct_mask.any():
            stability_loss = F.cross_entropy(
                logits[correct_mask],
                current_tokens[correct_mask]  # Should predict same token
            )
        else:
            stability_loss = torch.tensor(0.0, device=logits.device)

        # 4. Correction loss: reward fixing incorrect nodes
        if incorrect_mask.any():
            correction_loss = F.cross_entropy(
                logits[incorrect_mask],
                target_tokens[incorrect_mask]
            )
        else:
            correction_loss = torch.tensor(0.0, device=logits.device)

        # 5. Confidence calibration: high confidence on correct, low on incorrect
        # Target: 1.0 for correct nodes, 0.0 for incorrect
        confidence_targets = correct_mask.float()
        confidence_loss = F.binary_cross_entropy(
            confidence,
            confidence_targets
        )

        # 6. Test-following loss: encourage changes on nodes with failing tests
        # Extract test signals (feature 5)
        test_signals = current_graph.x[:, 5] if current_graph.x.size(1) > 5 else torch.zeros_like(current_tokens).float()

        # Penalize predicting same token on nodes marked by test failures
        predicted_tokens = logits.argmax(dim=-1)
        unchanged_mask = (predicted_tokens == current_tokens)

        # Loss: higher when we keep same token on failing test nodes
        test_following_loss = (test_signals * unchanged_mask.float()).mean()

        # Total weighted loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.stability_weight * stability_loss +
            self.correction_weight * correction_loss +
            self.confidence_weight * confidence_loss +
            self.test_following_weight * test_following_loss
        )

        # Metrics
        predicted_tokens = logits.argmax(dim=-1)
        accuracy = (predicted_tokens == target_tokens).float().mean()

        # Ensure tensors for accuracy metrics
        if correct_mask.any():
            correct_accuracy = (predicted_tokens[correct_mask] == target_tokens[correct_mask]).float().mean()
        else:
            correct_accuracy = torch.tensor(1.0, device=logits.device)

        if incorrect_mask.any():
            incorrect_accuracy = (predicted_tokens[incorrect_mask] == target_tokens[incorrect_mask]).float().mean()
        else:
            incorrect_accuracy = torch.tensor(0.0, device=logits.device)

        metrics = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'stability_loss': stability_loss.item() if isinstance(stability_loss, torch.Tensor) else 0.0,
            'correction_loss': correction_loss.item() if isinstance(correction_loss, torch.Tensor) else 0.0,
            'confidence_loss': confidence_loss.item(),
            'test_following_loss': test_following_loss.item(),
            'accuracy': accuracy.item(),
            'correct_accuracy': correct_accuracy.item(),
            'incorrect_accuracy': incorrect_accuracy.item(),
            'mean_confidence': confidence.mean().item(),
            'num_correct': correct_mask.sum().item(),
            'num_incorrect': incorrect_mask.sum().item(),
            'test_signals_active': (test_signals > 0).sum().item(),
        }

        return total_loss, metrics
