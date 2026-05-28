"""
Trajectory encoder for contrastive alignment training.

Two pooling modes:
  "mean"  — mean pooling over valid positions (default)
  "cls"   — learnable CLS token prepended; output taken from CLS position
            (inspired by CLaTr/ACTOR encoder)

Two positional encoding modes:
  "learned"    — nn.Embedding (default)
  "sinusoidal" — fixed sinusoidal (CLaTr-style, generalises to unseen lengths)

Input shapes:
  direction mode:        (B, T, 6)
  speed mode:            (B, T, 2)
  direction+speed mode:  (B, T, 8)
  trajectory mode:       (B, T, 9)

Output: (B, embed_dim)  L2-normalized
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (same as CLaTr/ACTOR)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return self.dropout(x + self.pe[:x.shape[1]].unsqueeze(0))


class TrajEncoder(nn.Module):
    """
    Args:
        input_dim:   Feature dimension per timestep.
        d_model:     Internal Transformer hidden dim.
        nhead:       Number of attention heads.
        num_layers:  Number of Transformer encoder layers.
        max_len:     Maximum sequence length.
        embed_dim:   Output embedding dimension (shared with text encoder).
        dropout:     Dropout rate.
        pooling:     "mean" or "cls".
        pos_enc:     "learned" or "sinusoidal".
    """

    def __init__(
        self,
        input_dim:  int   = 6,
        d_model:    int   = 256,
        nhead:      int   = 4,
        num_layers: int   = 4,
        max_len:    int   = 195,
        embed_dim:  int   = 256,
        dropout:    float = 0.1,
        pooling:    str   = "mean",
        pos_enc:    str   = "learned",
    ):
        super().__init__()
        self.pooling = pooling
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        if pos_enc == "sinusoidal":
            self.pos_enc = SinusoidalPE(d_model, max_len=5000, dropout=dropout)
            self.pos_embed = None
        else:
            self.pos_embed = nn.Embedding(max_len + 1, d_model)  # +1 for CLS
            self.pos_enc = None

        # CLS token
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        else:
            self.cls_token = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, embed_dim),
        )

    def forward(self, x: torch.Tensor, seq_lens: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Args:
            x:         (B, max_len, input_dim)  padded trajectory features
            seq_lens:  (B,)                     actual valid lengths
            normalize: If True, L2-normalize output (for retrieval/contrastive).
                       If False, return raw embeddings (for FCD computation).
        Returns:
            embed: (B, embed_dim)
        """
        B, L, _ = x.shape
        device  = x.device

        h = self.input_proj(x)                                     # (B, L, d_model)

        if self.pooling == "cls":
            # Prepend CLS token
            cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, d_model)
            h   = torch.cat([cls, h], dim=1)                       # (B, 1+L, d_model)
            seq_len_with_cls = L + 1

            # Padding mask: CLS token is always valid (False = attend)
            idx       = torch.arange(seq_len_with_cls, device=device).unsqueeze(0)
            # CLS at pos 0 is valid; positions 1..L are valid if < seq_lens+1
            pad_mask  = idx >= (seq_lens.unsqueeze(1) + 1)         # (B, 1+L)

            # Positional encoding
            if self.pos_enc is not None:
                h = self.pos_enc(h)
            else:
                pos = torch.arange(seq_len_with_cls, device=device)
                h   = h + self.pos_embed(pos)

            h = self.encoder(h, src_key_padding_mask=pad_mask)     # (B, 1+L, d_model)
            h = h[:, 0]                                            # (B, d_model) — CLS output

        else:
            # Mean pooling path (original)
            idx      = torch.arange(L, device=device).unsqueeze(0)
            pad_mask = ~(idx < seq_lens.unsqueeze(1))              # (B, L)

            if self.pos_enc is not None:
                h = self.pos_enc(h)
            else:
                pos = torch.arange(L, device=device)
                h   = h + self.pos_embed(pos)

            h = self.encoder(h, src_key_padding_mask=pad_mask)     # (B, L, d_model)

            valid = (~pad_mask).float().unsqueeze(-1)              # (B, L, 1)
            h     = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        h = self.norm(h)
        h = self.proj(h)                                           # (B, embed_dim)
        return F.normalize(h, dim=-1) if normalize else h
