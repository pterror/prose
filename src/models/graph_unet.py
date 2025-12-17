"""Graph U-Net architecture for code synthesis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool
from typing import Optional

from .test_feedback_encoder import (
    TestFeedbackEncoder,
    TestFeedbackCrossAttention,
    TestGuidedGraphEncoder,
)


class GraphUNet(nn.Module):
    """
    Hierarchical Graph U-Net for ASG processing.

    Uses Graph Attention Networks with TopK pooling for encoder,
    and unpooling with skip connections for decoder.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int = 3,
        pool_ratio: float = 0.5,
        num_node_types: int = 9,  # 8 NodeTypes + MASK token
        layer_type: str = "GAT",
    ) -> None:
        """
        Initialize Graph U-Net.

        Args:
            in_channels: Input feature dimension (embedding size)
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (for reconstruction)
            depth: Number of pooling/unpooling layers
            pool_ratio: Ratio of nodes to keep in pooling (0.0-1.0)
            num_node_types: Number of node types in vocabulary
            layer_type: "GAT" or "GCN"
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.num_node_types = num_node_types

        # Node type embedding (for categorical node types)
        self.node_embedding = nn.Embedding(num_node_types, in_channels)

        # Position encoding: Linear layer to project [depth, sibling_index] to same dim as node embedding
        # We'll concatenate normalized position features with node type embedding
        # Total input to graph layers: in_channels (node embedding) + 2 (position features)
        self.position_norm = nn.LayerNorm(2)  # Normalize position features
        self.position_projection = nn.Linear(2, in_channels // 4)  # Project to smaller dim

        # Adjust effective input channels for graph layers
        self.effective_in_channels = in_channels + in_channels // 4

        # Edge type embedding (3 types: Child, Sibling, DataFlow)
        self.edge_embedding = nn.Embedding(3, in_channels)

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        current_channels = self.effective_in_channels
        for i in range(depth):
            # Graph convolution layer
            if layer_type == "GAT":
                self.encoder_layers.append(
                    GATConv(
                        current_channels,
                        hidden_channels,
                        heads=1,
                        concat=False,
                    )
                )
            else:  # GCN
                self.encoder_layers.append(GCNConv(current_channels, hidden_channels))

            # Pooling layer
            self.pool_layers.append(TopKPooling(hidden_channels, ratio=pool_ratio))

            current_channels = hidden_channels

        # Bottleneck (deep processing at coarsest level)
        self.bottleneck = nn.ModuleList(
            [
                GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
                if layer_type == "GAT"
                else GCNConv(hidden_channels, hidden_channels)
                for _ in range(2)
            ]
        )

        # Decoder layers (reverse of encoder)
        self.decoder_layers = nn.ModuleList()
        self.unpool_layers = nn.ModuleList()

        for i in range(depth):
            # Skip connection requires 2x channels
            skip_channels = hidden_channels * 2 if i > 0 else hidden_channels

            if layer_type == "GAT":
                self.decoder_layers.append(
                    GATConv(skip_channels, hidden_channels, heads=1, concat=False)
                )
            else:
                self.decoder_layers.append(GCNConv(skip_channels, hidden_channels))

        # Output heads
        self.node_output = nn.Linear(hidden_channels, num_node_types)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through Graph U-Net.

        Args:
            data: PyG Data object with x [num_nodes, 3] = [node_type, depth, sibling_index]

        Returns:
            Reconstructed node type logits [num_nodes, num_node_types]
        """
        # Split node features into node_type and position features
        node_types = data.x[:, 0]  # [num_nodes]
        position_features = data.x[:, 1:].float()  # [num_nodes, 2] (depth, sibling_index)

        # Embed node types
        node_emb = self.node_embedding(node_types)  # [num_nodes, in_channels]

        # Process position features
        pos_norm = self.position_norm(position_features)  # [num_nodes, 2]
        pos_emb = self.position_projection(pos_norm)  # [num_nodes, in_channels // 4]

        # Concatenate node type embedding with position encoding
        x = torch.cat([node_emb, pos_emb], dim=-1)  # [num_nodes, effective_in_channels]

        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") else None

        # Store skip connections
        skip_connections = []
        pool_indices = []

        # Encoder with pooling
        for i in range(self.depth):
            # Graph convolution
            x = self.encoder_layers[i](x, edge_index)
            x = F.gelu(x)

            # Save skip connection before pooling
            skip_connections.append(x)

            # Pool
            x, edge_index, edge_attr, batch, perm, score = self.pool_layers[i](
                x, edge_index, edge_attr=None, batch=batch
            )
            pool_indices.append(perm)

        # Bottleneck
        for layer in self.bottleneck:
            x = layer(x, edge_index)
            x = F.gelu(x)

        # Decoder with unpooling (simplified version without actual unpooling)
        # For now, we'll just reverse the process using the saved indices
        # A full implementation would restore the original graph structure
        # This simplified version processes at the pooled level

        # Output layer
        node_logits = self.node_output(x)

        return node_logits

    def forward_full(self, data: Data) -> torch.Tensor:
        """
        Forward pass that returns full-resolution output.

        This is a simplified implementation that doesn't actually unpool.
        For a complete implementation, we'd need to restore the original
        graph structure using the pooling indices.

        Args:
            data: PyG Data object with x [num_nodes, 3]

        Returns:
            Node type predictions for ALL original nodes
        """
        # For Phase 1, we'll use a simpler approach:
        # Just run encoder-bottleneck-decoder at full resolution

        # Split and process node features (same as forward)
        # Handle both corrupted (1D) and uncorrupted (2D) data
        if data.x.dim() == 1:
            # Corrupted data: just node types, no position features
            node_types = data.x
            # Create dummy position features (zeros)
            position_features = torch.zeros(
                data.x.size(0), 2, device=data.x.device, dtype=torch.float32
            )
        else:
            # Uncorrupted data: [node_type, depth, sibling_index]
            node_types = data.x[:, 0]
            position_features = data.x[:, 1:].float()

        node_emb = self.node_embedding(node_types)
        pos_norm = self.position_norm(position_features)
        pos_emb = self.position_projection(pos_norm)
        x = torch.cat([node_emb, pos_emb], dim=-1)

        edge_index = data.edge_index

        # Single-level encoding
        for layer in self.encoder_layers[:1]:  # Use only first encoder layer
            x = layer(x, edge_index)
            x = F.gelu(x)

        # Bottleneck
        for layer in self.bottleneck:
            x = layer(x, edge_index)
            x = F.gelu(x)

        # Decode
        for layer in self.decoder_layers[:1]:  # Use only first decoder layer
            x = layer(x, edge_index)
            x = F.gelu(x)

        # Output
        node_logits = self.node_output(x)

        return node_logits


class IterativeGraphUNet(nn.Module):
    """
    Iterative Graph U-Net for Phase 1.5 test-driven refinement.

    Processes 6-feature nodes: [token_id, prev_token_id, depth, sibling_index, iteration, test_signal]
    Outputs token predictions and confidence scores for iterative refinement.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_channels: int = 256,
        depth: int = 3,
        max_iterations: int = 5,
        pool_ratio: float = 0.5,
        layer_type: str = "GAT",
    ) -> None:
        """
        Initialize Iterative Graph U-Net.

        Args:
            vocab_size: Size of token vocabulary
            hidden_channels: Hidden layer dimension
            depth: Number of pooling/unpooling layers
            max_iterations: Maximum refinement iterations
            pool_ratio: Ratio of nodes to keep in pooling
            layer_type: "GAT" or "GCN"
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.max_iterations = max_iterations

        # Feature embeddings
        self.token_embedding = nn.Embedding(vocab_size, 128)
        self.prev_token_embedding = nn.Embedding(vocab_size, 32)
        self.iteration_embedding = nn.Embedding(max_iterations, 32)

        # Position and test signal projections
        self.position_projection = nn.Linear(2, 32)  # depth, sibling_index
        self.test_signal_projection = nn.Linear(1, 32)

        # Total input dimension: 128 + 32 + 32 + 32 + 32 = 256
        self.input_dim = 128 + 32 + 32 + 32 + 32

        # Input projection to match hidden_channels if different
        if self.input_dim != hidden_channels:
            self.input_projection = nn.Linear(self.input_dim, hidden_channels)
        else:
            self.input_projection = nn.Identity()

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for i in range(depth):
            if layer_type == "GAT":
                self.encoder_layers.append(
                    GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
                )
            else:
                self.encoder_layers.append(GCNConv(hidden_channels, hidden_channels))

            self.pool_layers.append(TopKPooling(hidden_channels, ratio=pool_ratio))

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
            if layer_type == "GAT"
            else GCNConv(hidden_channels, hidden_channels)
            for _ in range(2)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(depth):
            skip_channels = hidden_channels * 2 if i > 0 else hidden_channels
            if layer_type == "GAT":
                self.decoder_layers.append(
                    GATConv(skip_channels, hidden_channels, heads=1, concat=False)
                )
            else:
                self.decoder_layers.append(GCNConv(skip_channels, hidden_channels))

        # Output heads
        self.token_predictor = nn.Linear(hidden_channels, vocab_size)
        self.confidence_head = nn.Linear(hidden_channels, 1)

    def forward(self, data: Data, iteration: int = 0) -> dict[str, torch.Tensor]:
        """
        Forward pass for iterative refinement.

        Args:
            data: PyG Data with x [num_nodes, 6] features:
                  [token_id, prev_token_id, depth, sibling_index, iteration, test_signal]
            iteration: Current iteration number (can override data.x[:, 4])

        Returns:
            Dict with:
                - 'logits': Token prediction logits [num_nodes, vocab_size]
                - 'confidence': Confidence scores [num_nodes, 1]
        """
        # Extract features from data
        # Handle both float and long tensors
        x_data = data.x

        token_ids = x_data[:, 0].long()  # Current token
        prev_token_ids = x_data[:, 1].long()  # Previous iteration token
        position_features = x_data[:, 2:4].float()  # depth, sibling_index
        iter_feature = x_data[:, 4].long() if iteration == 0 else torch.full((x_data.size(0),), iteration, dtype=torch.long, device=x_data.device)
        test_signals = x_data[:, 5:6].float()  # test_signal

        # Embed all features
        token_emb = self.token_embedding(token_ids)  # [num_nodes, 128]
        prev_emb = self.prev_token_embedding(prev_token_ids)  # [num_nodes, 32]
        pos_emb = self.position_projection(position_features)  # [num_nodes, 32]
        iter_emb = self.iteration_embedding(iter_feature)  # [num_nodes, 32]
        test_emb = self.test_signal_projection(test_signals)  # [num_nodes, 32]

        # Concatenate all features
        h = torch.cat([token_emb, prev_emb, pos_emb, iter_emb, test_emb], dim=-1)

        # Project to hidden dimension
        h = self.input_projection(h)
        h = F.gelu(h)

        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        # Store skip connections
        skip_connections = []
        pool_indices = []

        # Encoder with pooling
        for i in range(self.depth):
            h = self.encoder_layers[i](h, edge_index)
            h = F.gelu(h)

            skip_connections.append(h)

            # Pool
            h, edge_index, _, batch, perm, _ = self.pool_layers[i](
                h, edge_index, edge_attr=None, batch=batch
            )
            pool_indices.append(perm)

        # Bottleneck
        for layer in self.bottleneck:
            h = layer(h, edge_index)
            h = F.gelu(h)

        # For Phase 1.5, use simplified decoder without unpooling
        # Just process at pooled level and output

        # Output predictions
        logits = self.token_predictor(h)
        confidence = torch.sigmoid(self.confidence_head(h))

        return {
            'logits': logits,
            'confidence': confidence
        }

    def forward_full(self, data: Data, iteration: int = 0) -> dict[str, torch.Tensor]:
        """
        Forward pass at full resolution (no pooling) for training.

        Args:
            data: PyG Data with x [num_nodes, 6]
            iteration: Current iteration number

        Returns:
            Dict with logits and confidence for all nodes
        """
        # Extract and embed features (same as forward)
        x_data = data.x

        token_ids = x_data[:, 0].long()
        prev_token_ids = x_data[:, 1].long()
        position_features = x_data[:, 2:4].float()
        iter_feature = x_data[:, 4].long() if iteration == 0 else torch.full((x_data.size(0),), iteration, dtype=torch.long, device=x_data.device)
        test_signals = x_data[:, 5:6].float()

        token_emb = self.token_embedding(token_ids)
        prev_emb = self.prev_token_embedding(prev_token_ids)
        pos_emb = self.position_projection(position_features)
        iter_emb = self.iteration_embedding(iter_feature)
        test_emb = self.test_signal_projection(test_signals)

        h = torch.cat([token_emb, prev_emb, pos_emb, iter_emb, test_emb], dim=-1)
        h = self.input_projection(h)
        h = F.gelu(h)

        edge_index = data.edge_index

        # Single-level processing (no pooling for full resolution)
        for layer in self.encoder_layers[:1]:
            h = layer(h, edge_index)
            h = F.gelu(h)

        for layer in self.bottleneck:
            h = layer(h, edge_index)
            h = F.gelu(h)

        for layer in self.decoder_layers[:1]:
            h = layer(h, edge_index)
            h = F.gelu(h)

        # Output
        logits = self.token_predictor(h)
        confidence = torch.sigmoid(self.confidence_head(h))

        return {
            'logits': logits,
            'confidence': confidence
        }


class GuidedIterativeGraphUNet(nn.Module):
    """
    Iterative Graph U-Net with cross-attention to test feedback.

    This model processes node features AND test feedback history,
    using cross-attention to let nodes attend to relevant test failures.

    Key difference from IterativeGraphUNet:
    - Test signals are NOT node features, but separate guidance vectors
    - Nodes cross-attend to test results to get context-aware guidance
    - Scales to large codebases (1000+ tests, 10,000+ nodes)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_channels: int = 256,
        depth: int = 3,
        max_iterations: int = 5,
        max_tests: int = 100,
        max_nodes: int = 1000,
        pool_ratio: float = 0.5,
        layer_type: str = "GAT",
        num_attention_heads: int = 4,
        use_test_guidance: bool = True,
    ) -> None:
        """
        Initialize Guided Iterative Graph U-Net.

        Args:
            vocab_size: Size of token vocabulary
            hidden_channels: Hidden layer dimension
            depth: Number of pooling/unpooling layers
            max_iterations: Maximum refinement iterations
            max_tests: Maximum number of tests
            max_nodes: Maximum number of nodes (for trace encoding)
            pool_ratio: Ratio of nodes to keep in pooling
            layer_type: "GAT" or "GCN"
            num_attention_heads: Number of heads for cross-attention
            use_test_guidance: Whether to use test feedback (False = baseline)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.max_iterations = max_iterations
        self.use_test_guidance = use_test_guidance

        # Feature embeddings (NO test_signal feature anymore)
        self.token_embedding = nn.Embedding(vocab_size, 128)
        self.prev_token_embedding = nn.Embedding(vocab_size, 32)
        self.iteration_embedding = nn.Embedding(max_iterations, 32)

        # Position projection (depth, sibling_index)
        self.position_projection = nn.Linear(2, 32)

        # Total input dimension: 128 + 32 + 32 + 32 = 224
        self.input_dim = 128 + 32 + 32 + 32

        # Input projection to hidden_channels
        self.input_projection = nn.Linear(self.input_dim, hidden_channels)

        # Test feedback encoder
        if self.use_test_guidance:
            self.test_feedback_encoder = TestFeedbackEncoder(
                max_tests=max_tests,
                hidden_dim=hidden_channels,
                max_nodes=max_nodes,
            )

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        # Cross-attention layers (applied after each encoder layer)
        if self.use_test_guidance:
            self.cross_attention_layers = nn.ModuleList()

        for i in range(depth):
            if layer_type == "GAT":
                self.encoder_layers.append(
                    GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
                )
            else:
                self.encoder_layers.append(GCNConv(hidden_channels, hidden_channels))

            self.pool_layers.append(TopKPooling(hidden_channels, ratio=pool_ratio))

            # Add cross-attention after each encoder layer
            if self.use_test_guidance:
                self.cross_attention_layers.append(
                    TestFeedbackCrossAttention(
                        node_dim=hidden_channels,
                        test_dim=hidden_channels,
                        num_heads=num_attention_heads,
                        dropout=0.1,
                    )
                )

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels, heads=1, concat=False)
            if layer_type == "GAT"
            else GCNConv(hidden_channels, hidden_channels)
            for _ in range(2)
        ])

        # Cross-attention in bottleneck
        if self.use_test_guidance:
            self.bottleneck_cross_attention = TestFeedbackCrossAttention(
                node_dim=hidden_channels,
                test_dim=hidden_channels,
                num_heads=num_attention_heads,
                dropout=0.1,
            )

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(depth):
            skip_channels = hidden_channels * 2 if i > 0 else hidden_channels
            if layer_type == "GAT":
                self.decoder_layers.append(
                    GATConv(skip_channels, hidden_channels, heads=1, concat=False)
                )
            else:
                self.decoder_layers.append(GCNConv(skip_channels, hidden_channels))

        # Output heads
        self.token_predictor = nn.Linear(hidden_channels, vocab_size)
        self.confidence_head = nn.Linear(hidden_channels, 1)

    def forward_full(
        self,
        data: Data,
        iteration: int = 0,
        test_feedback: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass at full resolution with test feedback guidance.

        Args:
            data: PyG Data with x [num_nodes, 5] features:
                  [token_id, prev_token_id, depth, sibling_index, iteration]
                  (NOTE: No test_signal feature anymore!)
            iteration: Current iteration number
            test_feedback: Optional dict with:
                - 'test_ids': [num_tests] Test identifiers
                - 'test_statuses': [num_tests, 1] Pass/fail (0=pass, 1=fail)
                - 'test_traces': [num_tests, max_nodes] Execution traces

        Returns:
            Dict with:
                - 'logits': Token predictions [num_nodes, vocab_size]
                - 'confidence': Confidence scores [num_nodes, 1]
        """
        # Extract features from data
        x_data = data.x

        token_ids = x_data[:, 0].long()
        prev_token_ids = x_data[:, 1].long()
        position_features = x_data[:, 2:4].float()  # depth, sibling_index
        iter_feature = x_data[:, 4].long() if iteration == 0 else torch.full(
            (x_data.size(0),), iteration, dtype=torch.long, device=x_data.device
        )

        # Embed all features
        token_emb = self.token_embedding(token_ids)  # [num_nodes, 128]
        prev_emb = self.prev_token_embedding(prev_token_ids)  # [num_nodes, 32]
        pos_emb = self.position_projection(position_features)  # [num_nodes, 32]
        iter_emb = self.iteration_embedding(iter_feature)  # [num_nodes, 32]

        # Concatenate features (no test_signal!)
        h = torch.cat([token_emb, prev_emb, pos_emb, iter_emb], dim=-1)

        # Project to hidden dimension
        h = self.input_projection(h)
        h = F.gelu(h)

        # Encode test feedback if provided
        test_embeddings = None
        test_mask = None
        if self.use_test_guidance and test_feedback is not None:
            # Add batch dimension if missing
            test_ids = test_feedback['test_ids']
            test_statuses = test_feedback['test_statuses']
            test_traces = test_feedback['test_traces']

            if test_ids.dim() == 1:
                test_ids = test_ids.unsqueeze(0)
                test_statuses = test_statuses.unsqueeze(0)
                test_traces = test_traces.unsqueeze(0)

            # Encode test feedback
            test_embeddings = self.test_feedback_encoder(
                test_ids, test_statuses, test_traces
            )  # [1, num_tests, hidden_channels]

            # Create mask for valid tests (non-padding)
            test_mask = (test_ids != -1).float()  # [1, num_tests]

        edge_index = data.edge_index

        # Single-level processing (no pooling for full resolution)
        # Apply first encoder layer
        h = self.encoder_layers[0](h, edge_index)
        h = F.gelu(h)

        # Apply cross-attention if test feedback available
        if self.use_test_guidance and test_embeddings is not None:
            # Add batch dimension to node features: [num_nodes, hidden] -> [1, num_nodes, hidden]
            h_batched = h.unsqueeze(0)
            h_guided = self.cross_attention_layers[0](h_batched, test_embeddings, test_mask)
            h = h_guided.squeeze(0)  # [num_nodes, hidden]

        # Bottleneck
        for layer in self.bottleneck:
            h = layer(h, edge_index)
            h = F.gelu(h)

        # Cross-attention in bottleneck
        if self.use_test_guidance and test_embeddings is not None:
            h_batched = h.unsqueeze(0)
            h_guided = self.bottleneck_cross_attention(h_batched, test_embeddings, test_mask)
            h = h_guided.squeeze(0)

        # Decoder
        for layer in self.decoder_layers[:1]:
            h = layer(h, edge_index)
            h = F.gelu(h)

        # Output
        logits = self.token_predictor(h)
        confidence = torch.sigmoid(self.confidence_head(h))

        return {
            'logits': logits,
            'confidence': confidence
        }
