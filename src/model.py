import torch
import torch.nn as nn


class ChessTransformer(nn.Module):
    """
    Transformer for chess move prediction.

    Input:
        (batch, history, channels, 8, 8)

    Outputs:
        policy_logits : (batch, num_moves)
        value         : (batch, 1)
    """

    def __init__(
        self,
        history=8,
        channels=17,
        d_model=256,
        nhead=8,
        num_layers=6,
        dropout=0.1,
        num_moves=4160,
    ):
        super().__init__()

        self.history = history
        self.channels = channels
        self.input_dim = history * channels

        ####################################################
        # Input embedding
        ####################################################

        self.embedding = nn.Linear(self.input_dim, d_model)

        ####################################################
        # CLS Token
        ####################################################

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        ####################################################
        # Positional Embedding
        #
        # 64 board squares
        # +1 CLS token
        ####################################################

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 65, d_model)
        )

        ####################################################
        # Transformer Encoder
        ####################################################

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        ####################################################
        # Final normalization
        ####################################################

        self.norm = nn.LayerNorm(d_model)

        ####################################################
        # Policy Head
        ####################################################

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_moves),
        )

        ####################################################
        # Value Head
        ####################################################

        self.value_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        x:
            (batch, history, channels, 8, 8)
        """

        B = x.size(0)

        ####################################################
        # Rearrange dimensions
        #
        # (B,H,C,8,8)
        # ->
        # (B,8,8,H,C)
        ####################################################

        x = x.permute(0, 3, 4, 1, 2)

        ####################################################
        # Flatten history and channels
        #
        # (B,64,H*C)
        ####################################################

        x = x.reshape(
            B,
            64,
            self.input_dim,
        )

        ####################################################
        # Linear embedding
        ####################################################

        x = self.embedding(x)

        ####################################################
        # Add CLS token
        ####################################################

        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls, x], dim=1)

        ####################################################
        # Add positional embeddings
        ####################################################

        x = x + self.pos_embedding

        ####################################################
        # Transformer
        ####################################################

        x = self.transformer(x)

        ####################################################
        # CLS output
        ####################################################

        cls_output = self.norm(x[:, 0])

        ####################################################
        # Heads
        ####################################################

        policy_logits = self.policy_head(cls_output)

        value = self.value_head(cls_output)

        return policy_logits, value