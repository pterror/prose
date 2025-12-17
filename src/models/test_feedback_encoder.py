"""Test feedback encoder for guided iterative refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TestFeedbackEncoder(nn.Module):
    """
    Encodes test execution results into embeddings for cross-attention.

    Each test result is encoded as a vector containing:
    - Test ID embedding
    - Pass/fail status
    - Execution trace (which nodes were executed)
    - Optional: Error location hints

    This allows the model to attend to relevant test failures during refinement.
    """

    def __init__(
        self,
        max_tests: int = 100,
        hidden_dim: int = 256,
        max_nodes: int = 1000,
    ):
        """
        Initialize test feedback encoder.

        Args:
            max_tests: Maximum number of tests to encode
            hidden_dim: Dimension of test embeddings
            max_nodes: Maximum number of nodes in graph (for sparse trace encoding)
        """
        super().__init__()

        self.max_tests = max_tests
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes

        # Test ID embedding (learnable)
        self.test_id_embedding = nn.Embedding(max_tests, hidden_dim // 2)

        # Status encoding (pass=0, fail=1)
        self.status_projection = nn.Linear(1, hidden_dim // 4)

        # Execution trace encoder (sparse encoding of traced nodes)
        # We'll use a simple approach: encode traced node indices
        self.trace_encoder = nn.Sequential(
            nn.Linear(max_nodes, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )

        # Final projection to combine all features
        # Total: hidden_dim/2 (test_id) + hidden_dim/4 (status) + hidden_dim/4 (trace) = hidden_dim
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        test_ids: torch.Tensor,
        test_statuses: torch.Tensor,
        test_traces: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode test feedback.

        Args:
            test_ids: Test identifiers [batch_size, num_tests] (-1 for padding)
            test_statuses: Pass/fail status [batch_size, num_tests, 1] (0=pass, 1=fail)
            test_traces: Execution traces [batch_size, num_tests, max_nodes] (binary masks)

        Returns:
            Test embeddings [batch_size, num_tests, hidden_dim]
        """
        batch_size, num_tests = test_ids.shape

        # Handle padding: Replace -1 with 0 temporarily for embedding lookup
        test_ids_clamped = test_ids.clamp(min=0)

        # Encode test IDs
        test_id_emb = self.test_id_embedding(test_ids_clamped)  # [batch, num_tests, hidden_dim/2]

        # Encode status
        status_emb = self.status_projection(test_statuses)  # [batch, num_tests, hidden_dim/4]

        # Encode execution traces
        trace_emb = self.trace_encoder(test_traces)  # [batch, num_tests, hidden_dim/4]

        # Concatenate all features
        combined = torch.cat([test_id_emb, status_emb, trace_emb], dim=-1)

        # Final projection
        test_embeddings = self.output_projection(combined)

        return test_embeddings


class TestFeedbackCrossAttention(nn.Module):
    """
    Cross-attention layer for nodes to attend to test feedback.

    This allows each node to query relevant test failures and adjust
    its predictions accordingly.
    """

    def __init__(
        self,
        node_dim: int = 256,
        test_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-attention layer.

        Args:
            node_dim: Dimension of node features
            test_dim: Dimension of test feedback embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.node_dim = node_dim
        self.test_dim = test_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"

        # Query projection (from nodes)
        self.query_proj = nn.Linear(node_dim, node_dim)

        # Key/Value projections (from test feedback)
        self.key_proj = nn.Linear(test_dim, node_dim)
        self.value_proj = nn.Linear(test_dim, node_dim)

        # Output projection
        self.output_proj = nn.Linear(node_dim, node_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        test_embeddings: torch.Tensor,
        test_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention from nodes to test feedback.

        Args:
            node_features: Node features [batch_size, num_nodes, node_dim]
            test_embeddings: Test feedback [batch_size, num_tests, test_dim]
            test_mask: Optional mask for invalid tests [batch_size, num_tests]

        Returns:
            Updated node features [batch_size, num_nodes, node_dim]
        """
        batch_size, num_nodes, _ = node_features.shape
        _, num_tests, _ = test_embeddings.shape

        # Project to Q, K, V
        Q = self.query_proj(node_features)  # [batch, num_nodes, node_dim]
        K = self.key_proj(test_embeddings)  # [batch, num_tests, node_dim]
        V = self.value_proj(test_embeddings)  # [batch, num_tests, node_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_tests, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_tests, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [batch, num_heads, num_nodes/tests, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Shape: [batch, num_heads, num_nodes, num_tests]

        # Apply mask if provided (mask out padding tests)
        if test_mask is not None:
            # test_mask: [batch, num_tests] -> [batch, 1, 1, num_tests]
            mask_expanded = test_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, num_nodes, num_tests]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # [batch, num_heads, num_nodes, head_dim]

        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.node_dim)

        # Output projection
        output = self.output_proj(attended)

        # Residual connection + layer norm
        output = self.layer_norm(node_features + self.dropout(output))

        return output


class TestGuidedGraphEncoder(nn.Module):
    """
    Graph encoder with test feedback guidance via cross-attention.

    Interleaves graph convolution layers with cross-attention to test feedback,
    allowing nodes to iteratively refine predictions based on test results.
    """

    def __init__(
        self,
        node_dim: int = 256,
        test_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize test-guided encoder.

        Args:
            node_dim: Dimension of node features
            test_dim: Dimension of test embeddings
            num_layers: Number of graph conv + cross-attention layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_layers = num_layers

        # Cross-attention layers (applied after each graph layer)
        self.cross_attention_layers = nn.ModuleList([
            TestFeedbackCrossAttention(
                node_dim=node_dim,
                test_dim=test_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        node_features: torch.Tensor,
        test_embeddings: torch.Tensor,
        test_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply test-guided encoding.

        Args:
            node_features: Node features [batch_size, num_nodes, node_dim]
            test_embeddings: Test feedback [batch_size, num_tests, test_dim]
            test_mask: Optional mask [batch_size, num_tests]

        Returns:
            Guided node features [batch_size, num_nodes, node_dim]
        """
        h = node_features

        # Apply cross-attention layers
        for i, cross_attn in enumerate(self.cross_attention_layers):
            # Cross-attend to test feedback
            h = cross_attn(h, test_embeddings, test_mask)

        return h
