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
