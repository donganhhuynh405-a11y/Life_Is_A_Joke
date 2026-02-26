"""
Advanced ML Architectures for Crypto Trading
State-of-the-art models for maximum profitability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (Google, 2019)
    Best-in-class for multi-horizon time series forecasting
    
    Features:
    - Variable selection network (learns which features matter)
    - Temporal processing with LSTM and attention
    - Multi-horizon predictions
    - Interpretable attention weights
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 160,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        num_lstm_layers: int = 2,
        output_quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.output_quantiles = output_quantiles
        
        # Variable Selection Network (VSN)
        self.vsn = VariableSelectionNetwork(input_size, hidden_size)
        
        # Temporal processing with LSTM
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_lstm_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Static enrichment
        self.static_enrichment = GatedResidualNetwork(hidden_size)
        
        # Multi-head self-attention
        self.attention = InterpretableMultiHeadAttention(
            hidden_size,
            num_attention_heads
        )
        
        # Position-wise feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Output layer for quantile predictions
        self.output_layer = nn.Linear(hidden_size, len(output_quantiles))
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch, seq_len, input_size]
            
        Returns:
            predictions: [batch, seq_len, num_quantiles]
            interpretability: Dict with attention weights and variable importance
        """
        batch_size, seq_len, _ = x.shape
        
        # Variable selection (learn which features are important)
        selected_features, variable_weights = self.vsn(x)
        
        # Temporal processing
        lstm_out, _ = self.lstm(selected_features)
        
        # Static enrichment (context from entire sequence)
        enriched = self.static_enrichment(lstm_out)
        
        # Multi-head attention (temporal dependencies)
        attended, attention_weights = self.attention(enriched)
        attended = self.layer_norm1(attended + enriched)
        
        # Feed-forward
        ff_out = self.feed_forward(attended)
        output = self.layer_norm2(ff_out + attended)
        
        # Quantile predictions
        predictions = self.output_layer(output)
        
        # Interpretability information
        interpretability = {
            'variable_weights': variable_weights,
            'attention_weights': attention_weights
        }
        
        return predictions, interpretability


class VariableSelectionNetwork(nn.Module):
    """
    Learns which input variables are important
    Uses Gated Residual Networks (GRN)
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Per-variable GRNs
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_size)
            for _ in range(input_size)
        ])
        
        # Variable selection weights
        self.variable_selection = nn.Sequential(
            nn.Linear(input_size * hidden_size, input_size),
            nn.Softmax(dim=-1)
        )
        
        # Feature transformation
        self.feature_transform = nn.Linear(1, hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_size]
            
        Returns:
            selected: [batch, seq_len, hidden_size]
            weights: [batch, seq_len, input_size]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Transform each variable
        transformed = []
        for i in range(input_size):
            var_input = x[:, :, i:i+1]  # [batch, seq_len, 1]
            var_hidden = self.feature_transform(var_input)  # [batch, seq_len, hidden]
            var_processed = self.variable_grns[i](var_hidden)
            transformed.append(var_processed)
        
        transformed = torch.stack(transformed, dim=2)  # [batch, seq_len, input_size, hidden]
        
        # Compute variable selection weights
        flat_transformed = transformed.view(batch_size, seq_len, -1)
        weights = self.variable_selection(flat_transformed)  # [batch, seq_len, input_size]
        
        # Apply weights
        weights_expanded = weights.unsqueeze(-1)  # [batch, seq_len, input_size, 1]
        selected = (transformed * weights_expanded).sum(dim=2)  # [batch, seq_len, hidden]
        
        return selected, weights


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN)
    Provides skip connections with gating mechanisms
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Gating layer
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feed-forward
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        # Gating mechanism
        gate = self.sigmoid(self.gate(x))
        
        # Gated skip connection
        out = gate * out + (1 - gate) * x
        out = self.layer_norm(out)
        
        return out


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretable attention weights
    """
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_size)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.output(attended)
        
        # Average attention weights across heads for interpretability
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network (GAT) for crypto asset correlations
    Models relationships between different cryptocurrencies
    """
    
    def __init__(
        self,
        num_assets: int,
        input_features: int,
        hidden_features: int = 64,
        num_attention_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_assets = num_assets
        self.num_attention_heads = num_attention_heads
        
        # Multi-head graph attention layers
        self.gat1 = MultiHeadGraphAttention(
            input_features,
            hidden_features,
            num_attention_heads,
            dropout
        )
        
        self.gat2 = MultiHeadGraphAttention(
            hidden_features * num_attention_heads,
            hidden_features,
            1,
            dropout
        )
        
        # Edge prediction (for dynamic graph learning)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_features * 2, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [batch, num_assets, input_features]
            adj_matrix: [batch, num_assets, num_assets] or None (learns structure)
            
        Returns:
            output_features: [batch, num_assets, hidden_features]
            learned_adj: [batch, num_assets, num_assets]
        """
        # If no adjacency matrix provided, learn it
        if adj_matrix is None:
            adj_matrix = self.learn_graph_structure(node_features)
        
        # First GAT layer
        x = self.gat1(node_features, adj_matrix)
        x = F.elu(x)
        
        # Second GAT layer
        x = self.gat2(x, adj_matrix)
        
        return x, adj_matrix
    
    def learn_graph_structure(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Learn dynamic graph structure between assets
        """
        batch_size, num_assets, _ = node_features.shape
        
        # Compute pairwise connections
        adj_matrix = torch.zeros(batch_size, num_assets, num_assets, device=node_features.device)
        
        for i in range(num_assets):
            for j in range(num_assets):
                if i != j:
                    # Concatenate features of node i and j
                    pair_features = torch.cat([
                        node_features[:, i, :],
                        node_features[:, j, :]
                    ], dim=-1)
                    
                    # Predict edge weight
                    edge_weight = self.edge_predictor(pair_features).squeeze(-1)
                    adj_matrix[:, i, j] = edge_weight
        
        # Add self-loops
        eye = torch.eye(num_assets, device=node_features.device).unsqueeze(0)
        adj_matrix = adj_matrix + eye
        
        return adj_matrix


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head attention mechanism for graph data
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        
        # Attention mechanism per head
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout)
            for _ in range(num_heads)
        ])
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        # Apply each attention head
        outputs = [att(x, adj_matrix) for att in self.attentions]
        
        # Concatenate heads
        return torch.cat(outputs, dim=-1)


class GraphAttentionLayer(nn.Module):
    """
    Single graph attention layer
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        
        # Linear transformation
        h = self.W(x)  # [batch, num_nodes, out_features]
        
        # Attention mechanism
        h_i = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch, num_nodes, num_nodes, out_features]
        h_j = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [batch, num_nodes, num_nodes, out_features]
        
        # Concatenate for attention
        concat = torch.cat([h_i, h_j], dim=-1)  # [batch, num_nodes, num_nodes, 2*out_features]
        
        # Attention coefficients
        e = self.leaky_relu(self.a(concat).squeeze(-1))  # [batch, num_nodes, num_nodes]
        
        # Mask attention with adjacency matrix
        e_masked = e * adj_matrix + (1 - adj_matrix) * (-1e9)
        
        # Normalize attention coefficients
        attention = F.softmax(e_masked, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to features
        output = torch.matmul(attention, h)  # [batch, num_nodes, out_features]
        
        return output


class MetaLearningMAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML)
    Enables fast adaptation to new market regimes with few examples
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5
    ):
        super().__init__()
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
    
    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: nn.Module
    ) -> nn.Module:
        """
        Adapt model to new task using support set
        
        Args:
            support_x: Support set inputs
            support_y: Support set targets
            loss_fn: Loss function
            
        Returns:
            Adapted model
        """
        # Clone model for adaptation
        adapted_model = type(self.base_model)(
            *self.base_model.init_args
        ).to(support_x.device)
        adapted_model.load_state_dict(self.base_model.state_dict())
        
        # Inner loop: adapt to new task
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.num_inner_steps):
            predictions = adapted_model(support_x)
            loss = loss_fn(predictions, support_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def meta_update(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        loss_fn: nn.Module,
        meta_optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Meta-update using multiple tasks
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples
            loss_fn: Loss function
            meta_optimizer: Meta-optimizer
            
        Returns:
            Average meta-loss
        """
        meta_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Adapt to task
            adapted_model = self.adapt(support_x, support_y, loss_fn)
            
            # Evaluate on query set
            predictions = adapted_model(query_x)
            loss = loss_fn(predictions, query_y)
            meta_loss += loss
        
        # Meta-update
        meta_loss = meta_loss / len(tasks)
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        return meta_loss.item()


class MultiTaskLearningHead(nn.Module):
    """
    Multi-task learning for simultaneous prediction of:
    - Price direction
    - Volatility
    - Market regime
    - Optimal position size
    """
    
    def __init__(self, shared_size: int, hidden_size: int = 128):
        super().__init__()
        
        # Shared representation
        self.shared_layers = nn.Sequential(
            nn.Linear(shared_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.price_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # UP, DOWN, SIDEWAYS
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive volatility
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # BULL, BEAR, RANGING, HIGH_VOL
        )
        
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0 to 1 (percentage of capital)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Shared representation
        shared = self.shared_layers(x)
        
        # Task-specific predictions
        outputs = {
            'price_direction': self.price_head(shared),
            'volatility': self.volatility_head(shared),
            'market_regime': self.regime_head(shared),
            'position_size': self.position_head(shared)
        }
        
        return outputs


def create_ultimate_model(
    input_size: int,
    num_assets: int,
    device: torch.device
) -> nn.Module:
    """
    Create the ultimate ensemble model combining all architectures
    """
    
    class UltimateEnsemble(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Temporal Fusion Transformer for time series
            self.tft = TemporalFusionTransformer(
                input_size=input_size,
                hidden_size=160,
                num_attention_heads=4
            )
            
            # Graph Attention Network for cross-asset relationships
            self.gat = GraphAttentionNetwork(
                num_assets=num_assets,
                input_features=input_size,
                hidden_features=64,
                num_attention_heads=4
            )
            
            # Multi-task learning head
            self.mtl = MultiTaskLearningHead(
                shared_size=160 + 64,  # TFT + GAT outputs
                hidden_size=128
            )
            
            self.init_args = (input_size, num_assets)
            
        def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None):
            # Temporal analysis with TFT
            tft_out, interpretability = self.tft(x)
            
            # Graph analysis with GAT
            gat_out, learned_adj = self.gat(x, adj_matrix)
            
            # Combine representations
            combined = torch.cat([
                tft_out[:, -1, :],  # Last time step from TFT
                gat_out.mean(dim=1)  # Aggregated graph features
            ], dim=-1)
            
            # Multi-task predictions
            outputs = self.mtl(combined)
            outputs['interpretability'] = interpretability
            outputs['learned_graph'] = learned_adj
            
            return outputs
    
    model = UltimateEnsemble().to(device)
    logger.info(f"Created ultimate ensemble model with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model
