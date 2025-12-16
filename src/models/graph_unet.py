"""Graph U-Net architecture for code synthesis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool


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
