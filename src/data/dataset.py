"""PyTorch Dataset for corrupted ASG graphs (denoising task)."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


@dataclass
class TestCase:
    """Test case for a program."""
    inputs: List[Any]
    expected_output: Any
    actual_output: Optional[Any] = None
    passing: Optional[bool] = None


@dataclass
class ProgramSample:
    """Program sample with graph and test cases."""
    graph: Data
    tests: List[TestCase]
    metadata: Optional[dict] = None


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

        # Mask selected nodes (only corrupt node type, keep position features)
        for idx in corrupt_indices:
            # Defensive check
            if corrupted.x.dim() == 1:
                # Fallback: if 1D, just set the value directly
                corrupted.x[idx] = self.mask_token_id
            else:
                # 2D case: only mask first column (node type)
                corrupted.x[idx, 0] = self.mask_token_id

        return corrupted


class IterativeRefinementDataset(Dataset):
    """Dataset for iterative refinement task with test cases (Phase 1.5)."""

    def __init__(
        self,
        data_dir: Path,
        corruption_rate: float = 0.5,
        mask_token_id: int = 0,
        seed: int = 42,
        keep_structure: bool = True,
    ) -> None:
        """
        Initialize Phase 1.5 dataset.

        Args:
            data_dir: Directory containing .pt files with ProgramSample objects
            corruption_rate: Fraction of nodes to corrupt (0.0-1.0)
            mask_token_id: ID to use for masked tokens (typically vocabulary.mask_token_id)
            seed: Random seed for corruption
            keep_structure: If True and rate < 0.9, preserve structural keywords
        """
        self.data_dir = Path(data_dir)
        self.corruption_rate = corruption_rate
        self.mask_token_id = mask_token_id
        self.keep_structure = keep_structure
        self.rng = random.Random(seed)

        # Find all .pt files
        self.samples = sorted(list(self.data_dir.glob("*.pt")))
        if not self.samples:
            raise ValueError(f"No .pt files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Data, Data, List[TestCase]]:
        """
        Get a corrupted/original graph pair with test cases.

        Returns:
            (corrupted_graph, original_graph, tests) tuple
        """
        # Load program sample
        sample: ProgramSample = torch.load(self.samples[idx], weights_only=False)

        # Extract original graph
        original = sample.graph

        # Create corrupted version
        corrupted = self._corrupt_graph(original)

        return corrupted, original, sample.tests

    def _corrupt_graph(self, graph: Data) -> Data:
        """
        Corrupt a graph by masking random tokens.

        For Phase 1.5, this works with 6-feature nodes:
        [token_id, prev_token_id, depth, sibling_index, iteration, test_signal]

        Args:
            graph: Original PyG Data object with 6 features

        Returns:
            Corrupted copy with masked tokens
        """
        # Deep copy to avoid modifying original
        corrupted = Data(
            x=graph.x.clone(),
            edge_index=graph.edge_index.clone(),
            edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None,
        )

        # Copy other attributes if present
        for key in ['node_type']:
            if hasattr(graph, key):
                setattr(corrupted, key, getattr(graph, key).clone())

        # Select nodes to corrupt
        num_nodes = corrupted.x.size(0)

        if self.corruption_rate >= 0.9:
            # Full generation mode: mask everything except structure
            corrupt_indices = list(range(num_nodes))
        else:
            # Partial corruption
            num_corrupt = max(1, int(num_nodes * self.corruption_rate))
            corrupt_indices = self.rng.sample(range(num_nodes), num_corrupt)

            # If keep_structure, avoid masking structural keywords
            if self.keep_structure and hasattr(corrupted, 'node_type'):
                # Structural node types: DEFINE, LAMBDA, IF, LET (values 3, 4, 5, 6)
                structural_types = {3, 4, 5, 6}
                non_structural = [
                    i for i in range(num_nodes)
                    if corrupted.node_type[i].item() not in structural_types
                ]
                if len(non_structural) >= num_corrupt:
                    corrupt_indices = self.rng.sample(non_structural, num_corrupt)

        # Mask selected nodes (feature 0 = token_id, feature 1 = prev_token_id)
        for idx in corrupt_indices:
            corrupted.x[idx, 0] = self.mask_token_id  # Current token
            corrupted.x[idx, 1] = self.mask_token_id  # Previous token

        return corrupted
