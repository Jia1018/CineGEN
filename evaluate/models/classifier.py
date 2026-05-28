"""
Trajectory classifier for cinematic aspect classification.

Encodes a variable-length sequence of trajectory features into class logits
via a lightweight Transformer encoder + mean pooling + linear head.

Architecture mirrors TrajEncoder but replaces the L2-norm projection head
with a single linear classification layer.

Input shapes (depending on traj_type):
  trajectory: (B, T, 9)   rot6D(6) + rel_trans(3)
  velocity:   (B, T, 6)   trans_vel(3) + rot_vel(3)
  direction:  (B, T, 6)   trans_dir(3) + rot_dir(3)
  speed:      (B, T, 2)   [log_trans_speed, log_rot_speed]

Output: (B, num_classes)  raw logits (no softmax applied)
"""

import torch
import torch.nn as nn


class TrajClassifier(nn.Module):
    """
    Args:
        input_dim:   Feature dimension per timestep.
        d_model:     Internal Transformer hidden dim.
        nhead:       Number of attention heads.
        num_layers:  Number of Transformer encoder layers.
        max_len:     Maximum sequence length for positional embedding.
        num_classes: Number of output classes.
        dropout:     Dropout rate.
    """

    def __init__(
        self,
        input_dim:   int   = 6,
        d_model:     int   = 128,
        nhead:       int   = 4,
        num_layers:  int   = 4,
        max_len:     int   = 300,
        num_classes: int   = 10,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed  = nn.Embedding(max_len, d_model)

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
        """
        Args:
            x:        (B, max_len, input_dim)  padded trajectory features
            seq_lens: (B,)                     actual valid lengths
        Returns:
            logits: (B, num_classes)
        """
        B, L, _ = x.shape
        device  = x.device

        idx      = torch.arange(L, device=device).unsqueeze(0)
        pad_mask = ~(idx < seq_lens.unsqueeze(1))                  # (B, L)

        pos = torch.arange(L, device=device)
        h   = self.input_proj(x) + self.pos_embed(pos)            # (B, L, d_model)
        h   = self.encoder(h, src_key_padding_mask=pad_mask)

        valid = (~pad_mask).float().unsqueeze(-1)                  # (B, L, 1)
        h     = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        h = self.norm(h)
        return self.head(h)                                        # (B, num_classes)
