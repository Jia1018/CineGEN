"""
Contrastive alignment model: learns a shared embedding space for trajectory
features (direction or speed) and text (motion caption or content aspect).

Training objective: symmetric InfoNCE (CLIP-style) loss.

The model intentionally keeps text and trajectory encoders separate so they can
be frozen/fine-tuned independently and reused across multiple alignment tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate.models.traj_encoder import TrajEncoder
from evaluate.models.encoders import TextEncoder


class AlignModel(nn.Module):
    """
    Args:
        traj_encoder:       Encodes trajectory features → (B, embed_dim) L2-normed.
        text_encoder:       Encodes text strings → (B, embed_dim) L2-normed.
        init_temperature:   Initial softmax temperature (log scale internally).
        learn_temperature:  Whether to make temperature a learnable parameter.
    """

    def __init__(
        self,
        traj_encoder:       TrajEncoder,
        text_encoder:       TextEncoder,
        init_temperature:   float = 0.07,
        learn_temperature:  bool  = True,
        threshold_selfsim:  float = 0.99,   # mask text pairs above this cosine sim as false negatives
    ):
        super().__init__()
        self.traj_encoder      = traj_encoder
        self.text_encoder      = text_encoder
        self.threshold_selfsim = threshold_selfsim

        self.log_temp = nn.Parameter(
            torch.tensor(init_temperature).log(),
            requires_grad=learn_temperature,
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temp.exp()

    def encode_traj(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """(B, T, D), (B,) → (B, embed_dim) L2-normalized."""
        return self.traj_encoder(x, seq_lens)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """list[str] → (B, embed_dim) L2-normalized."""
        feat = self.text_encoder(texts)   # (B, embed_dim) — pooled
        return F.normalize(feat, dim=-1)

    def encode_traj_raw(self, x: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """(B, T, D), (B,) → (B, embed_dim) raw (unnormalized). Use for FCD."""
        return self.traj_encoder(x, seq_lens, normalize=False)

    def encode_text_raw(self, texts: list[str]) -> torch.Tensor:
        """list[str] → (B, embed_dim) raw (unnormalized). Use for FCD."""
        return self.text_encoder(texts)   # no L2 norm

    def forward(
        self,
        traj:     torch.Tensor,   # (B, T, D)
        seq_lens: torch.Tensor,   # (B,)
        texts:    list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (traj_embed, text_embed) both (B, embed_dim) L2-normalized."""
        return self.encode_traj(traj, seq_lens), self.encode_text(texts)

    def compute_loss(
        self,
        traj_embed: torch.Tensor,   # (B, embed_dim) L2-normalized
        text_embed: torch.Tensor,   # (B, embed_dim) L2-normalized
    ) -> torch.Tensor:
        """
        Symmetric InfoNCE loss with false-negative filtering.

        Pairs whose text embeddings have cosine similarity > threshold_selfsim
        are masked out (treated as neither positive nor negative), so the model
        is not penalized for correctly recognising near-duplicate captions
        (e.g. two clips both labelled "Interior" or both described as "slow pan").
        """
        B      = traj_embed.shape[0]
        device = traj_embed.device

        logits = (traj_embed @ text_embed.T) / self.temperature   # (B, B)

        # False-negative filtering
        if self.threshold_selfsim < 1.0:
            text_sim = text_embed @ text_embed.T                   # (B, B) cosine sim
            eye      = torch.eye(B, dtype=torch.bool, device=device)
            false_neg = (text_sim > self.threshold_selfsim) & ~eye
            logits = logits.masked_fill(false_neg, float("-inf"))

        labels = torch.arange(B, device=device)
        loss_t2m = F.cross_entropy(logits,   labels)   # trajectory → text
        loss_m2t = F.cross_entropy(logits.T, labels)   # text → trajectory
        return (loss_t2m + loss_m2t) / 2


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

VALID_TEXT_TYPES = frozenset([
    "motion",
    "logline_script",
    "macro_type",
    "setting_class",
    "subject_composition",
    "genre_vibe",
])

VALID_TRAJ_TYPES = frozenset([
    "trajectory", "velocity", "direction", "speed", "direction+speed"
])

TRAJ_INPUT_DIM = {
    "trajectory":      9,
    "velocity":        6,
    "direction":       6,
    "speed":           2,
    "direction+speed": 8,
}


def build_align_model(
    traj_type:          str   = "direction",
    embed_dim:          int   = 256,
    traj_d_model:       int   = 256,
    traj_nhead:         int   = 4,
    traj_num_layers:    int   = 4,
    max_vel_len:        int   = 195,
    dropout:            float = 0.1,
    clip_model_id:      str   = "openai/clip-vit-large-patch14",
    freeze_clip:        bool  = True,
    init_temperature:   float = 0.07,
    learn_temperature:  bool  = True,
    threshold_selfsim:  float = 0.99,
    pooling:            str   = "mean",
    pos_enc:            str   = "learned",
) -> AlignModel:
    """
    Convenience constructor that wires up TrajEncoder + TextEncoder into AlignModel.
    """
    assert traj_type in VALID_TRAJ_TYPES, f"traj_type must be one of {VALID_TRAJ_TYPES}"

    traj_enc = TrajEncoder(
        input_dim  = TRAJ_INPUT_DIM[traj_type],
        d_model    = traj_d_model,
        nhead      = traj_nhead,
        num_layers = traj_num_layers,
        max_len    = max_vel_len,
        embed_dim  = embed_dim,
        dropout    = dropout,
        pooling    = pooling,
        pos_enc    = pos_enc,
    )

    # Pooled text encoder: one vector per caption
    text_enc = TextEncoder(
        model_id   = clip_model_id,
        out_dim    = embed_dim,
        max_length = 77,
        pooled     = True,
        freeze     = freeze_clip,
    )

    return AlignModel(
        traj_encoder      = traj_enc,
        text_encoder      = text_enc,
        init_temperature  = init_temperature,
        learn_temperature = learn_temperature,
        threshold_selfsim = threshold_selfsim,
    )
