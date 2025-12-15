"""PyTorch Dataset for corrupted ASG graphs (denoising task)."""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class DenoisingGraphDataset(Dataset):
    """Dataset for graph denoising task."""

    def __init__(
        self,
        data_dir: Path,
        corruption_rate: float = 0.2,
        mask_token_id: int = 8,  # NodeType enum max + 1
        seed: int = 42,
    ) -> None:
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing .pt files
            corruption_rate: Fraction of nodes to corrupt (0.0-1.0)
            mask_token_id: ID to use for masked nodes
            seed: Random seed for corruption
        """
        self.data_dir = Path(data_dir)
        self.corruption_rate = corruption_rate
        self.mask_token_id = mask_token_id
        self.rng = random.Random(seed)

        # Find all .pt files
        self.samples = sorted(list(self.data_dir.glob("*.pt")))
        if not self.samples:
            raise ValueError(f"No .pt files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Data, Data]:
        """
        Get a corrupted/original graph pair.

        Returns:
            (corrupted_graph, original_graph) tuple
        """
        # Load original graph
        original = torch.load(self.samples[idx], weights_only=False)

        # Create corrupted version
        corrupted = self._corrupt_graph(original)

        return corrupted, original

    def _corrupt_graph(self, graph: Data) -> Data:
        """
        Corrupt a graph by masking random nodes.

        Args:
            graph: Original PyG Data object

        Returns:
            Corrupted copy with masked nodes
        """
        # Deep copy to avoid modifying original
        corrupted = Data(
            x=graph.x.clone(),
            edge_index=graph.edge_index.clone(),
            edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None,
        )

        # Select nodes to corrupt
        num_nodes = corrupted.x.size(0)
        num_corrupt = max(1, int(num_nodes * self.corruption_rate))

        corrupt_indices = self.rng.sample(range(num_nodes), num_corrupt)

        # Mask selected nodes
        for idx in corrupt_indices:
            corrupted.x[idx] = self.mask_token_id

        return corrupted
