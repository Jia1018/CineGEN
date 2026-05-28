"""
Alternative classifier architectures for trajectory classification.

Provides drop-in replacements for TrajClassifier to test whether
model architecture is a bottleneck for classification accuracy.

All models share the same interface:
  forward(x: (B, L, D), seq_lens: (B,)) -> logits: (B, C)
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM + mean pooling + linear head."""

    def __init__(
        self,
        input_dim:   int = 8,
        hidden_dim:  int = 128,
        num_layers:  int = 2,
        num_classes: int = 10,
        dropout:     float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        device = x.device

        # Pack padded sequences for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=L)

        # Mean pooling over valid timesteps
        idx = torch.arange(L, device=device).unsqueeze(0)
        mask = (idx < seq_lens.unsqueeze(1)).float().unsqueeze(-1)
        h = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        h = self.norm(h)
        return self.head(h)


class MLPClassifier(nn.Module):
    """Global statistics pooling + 3-layer MLP.

    Pools the trajectory into fixed-size statistics (mean, std, min, max
    over valid timesteps) and classifies with an MLP. No sequence modeling.
    """

    def __init__(
        self,
        input_dim:   int = 8,
        hidden_dim:  int = 256,
        num_classes: int = 10,
        dropout:     float = 0.1,
        **kwargs,
    ):
        super().__init__()
        # Pool: mean + std + min + max → 4 * input_dim
        pool_dim = input_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        device = x.device

        idx = torch.arange(L, device=device).unsqueeze(0)
        mask = (idx < seq_lens.unsqueeze(1))  # (B, L)
        mask_f = mask.float().unsqueeze(-1)   # (B, L, 1)

        # Replace padded positions with large/small values for min/max
        x_masked = x.clone()
        x_masked[~mask] = 0.0

        mean = (x_masked * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        diff_sq = ((x_masked - mean.unsqueeze(1)) ** 2) * mask_f
        std = (diff_sq.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)).sqrt()

        # For min/max, use masked_fill
        big = torch.finfo(x.dtype).max
        x_for_min = x.masked_fill(~mask.unsqueeze(-1), big)
        x_for_max = x.masked_fill(~mask.unsqueeze(-1), -big)
        x_min = x_for_min.min(dim=1).values
        x_max = x_for_max.max(dim=1).values

        pooled = torch.cat([mean, std, x_min, x_max], dim=-1)  # (B, 4D)
        return self.mlp(pooled)


class DeepTransformerClassifier(nn.Module):
    """Deeper/wider Transformer encoder for comparison."""

    def __init__(
        self,
        input_dim:   int = 8,
        d_model:     int = 256,
        nhead:       int = 8,
        num_layers:  int = 8,
        max_len:     int = 300,
        num_classes: int = 10,
        dropout:     float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        device = x.device

        idx = torch.arange(L, device=device).unsqueeze(0)
        pad_mask = ~(idx < seq_lens.unsqueeze(1))

        pos = torch.arange(L, device=device)
        h = self.input_proj(x) + self.pos_embed(pos)
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        valid = (~pad_mask).float().unsqueeze(-1)
        h = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        h = self.norm(h)
        return self.head(h)


CLASSIFIER_REGISTRY = {
    "transformer":      None,  # uses TrajClassifier from classifier.py
    "deep_transformer": DeepTransformerClassifier,
    "lstm":             LSTMClassifier,
    "mlp":              MLPClassifier,
}
