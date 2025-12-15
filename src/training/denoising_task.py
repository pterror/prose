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
        self, predictions: torch.Tensor, targets: Data, mask_token_id: int = 8
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute denoising loss.

        Args:
            predictions: Predicted node type logits [num_nodes, num_node_types]
            targets: Original graph Data object
            mask_token_id: ID of mask token (nodes to predict)

        Returns:
            (loss, metrics_dict) tuple
        """
        # Cross-entropy loss on node types
        target_labels = targets.x

        # Only compute loss on masked nodes (optional, for now use all)
        node_loss = F.cross_entropy(predictions, target_labels)

        # Compute accuracy
        predicted_labels = predictions.argmax(dim=-1)
        accuracy = (predicted_labels == target_labels).float().mean()

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
