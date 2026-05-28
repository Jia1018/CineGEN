"""
CineGen: text-conditioned camera trajectory generation.

Architecture
------------
- Sequencer: TransformerAdaLN (1-layer, d=512, nhead=8, ff=4096)
- Diffuser:  SimpleMLPAdaLN (3 ResBlocks, d=1024)

Conditioning (all injected through AdaLN modulation, never as prefix tokens):
- Motion caption (CLIP ViT-B/32, frozen) → 512-D
- Optional first_pose (encoded via AE encoder — real PulpAE or IdentityAE)
- Optional scene logline (CLIP, frozen) → projected via logline_proj MLP
- Optional cinematic-aspect embedding (per-category embedding)

The published variant uses:
  text_mode="combined", use_logline=True, use_first_pose=True, IdentityAE
  → "separate-encoded logline + first_pose + no AE compression"

See `IdentityAE` below for the no-AE variant. Use the regular `PulpAE`
(not included here) for the original AE-compressed setting.
"""

import copy
import math
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def modulate(x, shift, scale):
    """AdaLN modulation — handles both 2D (N, D) and 3D (B, L, D)."""
    if x.dim() == 3 and shift.dim() == 2:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift


def approx_gelu():
    return nn.GELU(approximate="tanh")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class InputProcess(nn.Module):
    """Linear projection of input latent to model dim."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.poseEmbedding = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        return self.poseEmbedding(x)


class TimeEmbedder(nn.Module):
    """Sinusoidal time embedding ."""
    def __init__(self, dim, time_scaling=1000.0):
        super().__init__()
        self.dim = dim
        self.time_scaling = time_scaling
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim),
        )

    def forward(self, t):
        t = t * self.time_scaling
        args = t.unsqueeze(-1) * self.freqs
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.mlp(emb)


# ─────────────────────────────────────────────────────────────────────────────
# Aspect Encoder
# ─────────────────────────────────────────────────────────────────────────────

STYLE_ASPECT_KEYS = ["macro_type", "setting_class", "subject_composition", "genre_vibe"]
ASPECT_VOCABS = {
    "macro_type": ["Interior (Restricted)", "Exterior (Open)", "Semi-Open (Hybrid)"],
    "setting_class": ["Domestic/Residential", "Urban/City", "Public/Commercial/Institutional",
                       "Nature", "Industrial", "Abstract/Void", "Rural/Nature"],
    "subject_composition": ["Single-Character", "Two-Shot", "Group/Crowd",
                            "Object-Centric", "Scenery-Only"],
    "genre_vibe": ["Drama / Romance / Emotion", "Thriller / Horror / Mystery",
                    "Standard Narrative / Dialogue", "Comedy / Lighthearted",
                    "Action / Chase / Combat", "Adventure / Fantasy / Sci-Fi",
                    "Epic / Historical / Heroic", "Lifestyle / Everyday / Commercial",
                    "Documentary / Realism", "Ethereal / Dreamy / Surreal"],
}


class AspectEncoder(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        per_aspect_dim = emb_dim  # each aspect gets same dim
        self.embeddings = nn.ModuleDict()
        for key in STYLE_ASPECT_KEYS:
            n_classes = len(ASPECT_VOCABS[key]) + 1
            self.embeddings[key] = nn.Embedding(n_classes, per_aspect_dim)
        self.fusion = nn.Sequential(
            nn.Linear(per_aspect_dim * len(STYLE_ASPECT_KEYS), emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, aspects, device, B):
        embs = []
        for key in STYLE_ASPECT_KEYS:
            vocab = ASPECT_VOCABS[key]
            vals = aspects.get(key, [""] * B) if aspects else [""] * B
            indices = []
            for v in vals:
                try: idx = vocab.index(v)
                except ValueError: idx = len(vocab)
                indices.append(idx)
            embs.append(self.embeddings[key](torch.tensor(indices, device=device)))
        return self.fusion(torch.cat(embs, dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# Sequencer: TransformerAdaLN
# ─────────────────────────────────────────────────────────────────────────────

class AdaLNBlock(nn.Module):
    """Transformer block with AdaLN modulation ."""
    def __init__(self, hidden_size, num_heads, mlp_size=4096, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_size,
                       act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, key_padding_mask=padding_mask)
        x = x + gate_msa.unsqueeze(1) * h
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CineGenSequencer(nn.Module):
    """TransformerAdaLN sequencer.

    Conditioning (text + first_pose + optional aspects) all go through AdaLN,
    NOT as prefix tokens.
    """
    def __init__(self, ae_dim, latent_dim=512, num_heads=8, ff_size=4096,
                 num_layers=1, dropout=0.2, clip_dim=512, cond_drop_prob=0.1,
                 use_first_pose=False, fp_emb_dim=64,
                 use_aspects=False, aspect_emb_dim=64,
                 use_logline=False, logline_emb_dim=64):
        super().__init__()
        self.ae_dim = ae_dim
        self.latent_dim = latent_dim
        self.cond_drop_prob = cond_drop_prob
        self.use_first_pose = use_first_pose
        self.use_aspects = use_aspects
        self.use_logline = use_logline
        # fp_emb_dim = ae_dim when using AE encoder (set externally)
        self.fp_emb_dim = fp_emb_dim
        self.logline_emb_dim = logline_emb_dim

        self.input_process = InputProcess(ae_dim, latent_dim)
        self.position_enc = PositionalEncoding(latent_dim, dropout)
        self.mask_latent = nn.Parameter(torch.zeros(1, 1, ae_dim))

        # Text conditioning (CLIP ViT-B/32 → 512D)
        self.text_proj = nn.Linear(clip_dim, clip_dim)  # identity-like, for consistency

        # First pose: encoded through frozen AE (set via set_ae_encoder)
        # No learnable MLP — uses the same AE encoder that encodes trajectories.
        # AE is set externally after construction via set_ae_encoder().
        self._ae_encoder = None  # set later

        # Aspect styles → small embedding (concat to cond)
        if use_aspects:
            self.aspect_encoder = AspectEncoder(aspect_emb_dim)

        # Logline (separately CLIP-encoded scene text) → small projection
        if use_logline:
            self.logline_proj = nn.Sequential(
                nn.Linear(clip_dim, logline_emb_dim),
                nn.SiLU(),
                nn.Linear(logline_emb_dim, logline_emb_dim),
            )

        # Conditioning projection: concatenated [text, fp?, aspects?, logline?] → latent_dim
        cond_input_dim = clip_dim
        if use_first_pose:
            cond_input_dim += fp_emb_dim  # ae_dim (64) from AE encoder
        if use_aspects:
            cond_input_dim += aspect_emb_dim
        if use_logline:
            cond_input_dim += logline_emb_dim
        self.cond_fusion = nn.Sequential(
            nn.Linear(cond_input_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNBlock(latent_dim, num_heads, ff_size, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_process = nn.Linear(latent_dim, ae_dim)

        self._init_weights()

    def _init_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        def _basic_init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic_init)

    def set_ae_encoder(self, ae_encoder):
        """Set the frozen AE encoder for first_pose encoding."""
        self._ae_encoder = ae_encoder

    @torch.no_grad()
    def _encode_first_pose(self, first_pose):
        """Encode first_pose (B, D) through frozen AE encoder → (B, ae_dim).

        Repeats the pose to T=4 (minimum for AE's stride-2 downsampling),
        encodes, and takes the single output latent token.
        """
        fp_seq = first_pose.unsqueeze(1).expand(-1, 4, -1)  # (B, 4, D)
        z = self._ae_encoder(fp_seq.permute(0, 2, 1))       # (B, ae_dim, 1)
        return z.squeeze(-1)                                  # (B, ae_dim)

    def _build_cond(self, text_emb, first_pose, aspects, logline_emb,
                    force_mask, device, B):
        """Build AdaLN conditioning via concatenation: [text, fp?, aspects?, logline?] → fused."""
        if force_mask or (self.training and torch.rand(1).item() < self.cond_drop_prob):
            return torch.zeros(B, self.latent_dim, device=device)

        parts = [self.text_proj(text_emb)]  # (B, 512)

        if self.use_first_pose:
            if first_pose is not None and self._ae_encoder is not None:
                parts.append(self._encode_first_pose(first_pose))  # (B, ae_dim=64)
            else:
                parts.append(torch.zeros(B, self.fp_emb_dim,
                                         device=device, dtype=text_emb.dtype))

        if self.use_aspects:
            if aspects is not None:
                parts.append(self.aspect_encoder(aspects, device, B))  # (B, 64)
            else:
                parts.append(torch.zeros(B, self.aspect_encoder.emb_dim,
                                         device=device, dtype=text_emb.dtype))

        if self.use_logline:
            if logline_emb is not None:
                parts.append(self.logline_proj(logline_emb))  # (B, logline_emb_dim)
            else:
                parts.append(torch.zeros(B, self.logline_emb_dim,
                                         device=device, dtype=text_emb.dtype))

        return self.cond_fusion(torch.cat(parts, dim=-1))  # (B, latent_dim)

    def forward(self, latents, text_emb, padding_mask=None, force_mask=False,
                first_pose=None, aspects=None, logline_emb=None):
        B, L, _ = latents.shape
        device = latents.device

        cond = self._build_cond(text_emb, first_pose, aspects, logline_emb,
                                force_mask, device, B)

        x = self.input_process(latents)
        x = self.position_enc(x)

        for block in self.blocks:
            x = block(x, cond, padding_mask=padding_mask)

        # Return both projected output (ae_dim) and hidden states (latent_dim)
        return self.output_process(x), x


# ─────────────────────────────────────────────────────────────────────────────
# Diffuser: SimpleMLPAdaLN 
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """AdaLN residual block ."""
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True),
        )

    def forward(self, x, y):
        shift, scale, gate = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift, scale)
        return x + gate * self.mlp(h)


class FinalLayer(nn.Module):
    """Final layer with AdaLN ."""
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class CineGenDiffuser(nn.Module):
    """SimpleMLPAdaLN diffuser ."""
    def __init__(self, in_channels, model_channels=1024, out_channels=None,
                 z_channels=512, num_res_blocks=3):
        super().__init__()
        out_channels = out_channels or in_channels

        self.time_embed = TimeEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList([ResBlock(model_channels) for _ in range(num_res_blocks)])
        self.final_layer = FinalLayer(model_channels, out_channels)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)


# ─────────────────────────────────────────────────────────────────────────────
# EMA ( callbacks/ema.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    """Update EMA parameters."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


# ─────────────────────────────────────────────────────────────────────────────
# DDPM Noise Schedule
# ─────────────────────────────────────────────────────────────────────────────

class CosineScheduler:
    """Cosine noise schedule for DDPM."""
    def __init__(self, T=100, s=0.008):
        t = torch.arange(T + 1)
        f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = betas.clamp(max=0.999)

        self.T = T
        self.betas = betas
        self.alphas = 1 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t]
        while ab.dim() < x0.dim():
            ab = ab.unsqueeze(-1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    def posterior_sample(self, x_t, x_0_pred, t, t_prev):
        ab_t = self.alpha_bar[t]
        ab_tp = self.alpha_bar[t_prev]
        while ab_t.dim() < x_t.dim():
            ab_t = ab_t.unsqueeze(-1)
            ab_tp = ab_tp.unsqueeze(-1)
        mean = (ab_tp.sqrt() * (1 - ab_t / ab_tp) * x_0_pred +
                (ab_t / ab_tp).sqrt() * (1 - ab_tp) * x_t) / (1 - ab_t)
        var = (1 - ab_tp) * (1 - ab_t / ab_tp) / (1 - ab_t)
        noise = torch.randn_like(x_t)
        mask = (t > 0).float()
        while mask.dim() < x_t.dim():
            mask = mask.unsqueeze(-1)
        return mean + mask * var.sqrt() * noise


# ─────────────────────────────────────────────────────────────────────────────
# Full CineGen
# ─────────────────────────────────────────────────────────────────────────────

class CineGen(nn.Module):
    """
    MAR-style MAR: Sequencer + Diffuser + EMA.

    All conditioning (text, first_pose, aspects) goes through AdaLN.
    """
    def __init__(
        self,
        ae_dim: int = 64,
        # Sequencer
        seq_latent_dim: int = 512,
        seq_nhead: int = 8,
        seq_ff_size: int = 4096,
        seq_num_layers: int = 1,
        seq_dropout: float = 0.2,
        # Diffuser
        diff_model_channels: int = 1024,
        diff_num_res_blocks: int = 3,
        # Conditioning
        clip_dim: int = 512,
        cond_drop_prob: float = 0.1,
        use_first_pose: bool = False,
        use_aspects: bool = False,
        use_logline: bool = False,
        logline_emb_dim: int = 64,
        # DDPM
        ddpm_T: int = 100,
        # MAR
        mask_ratio_min: float = 0.5,
        n_ar_steps: int = 18,
        # Text
        text_mode: str = "motion",
        # EMA
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.ae_dim = ae_dim
        self.ddpm_T = ddpm_T
        self.text_mode = text_mode
        self.mask_ratio_min = mask_ratio_min
        self.n_ar_steps = n_ar_steps
        self.ema_decay = ema_decay
        self.use_logline = use_logline

        # CLIP text encoder (frozen) — ViT-B/32 (512D)
        from transformers import CLIPModel, CLIPTokenizer
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = clip.text_model
        self.clip_text_proj = clip.text_projection
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.clip_text_proj.parameters():
            p.requires_grad_(False)

        # Sequencer
        self.sequencer = CineGenSequencer(
            ae_dim=ae_dim, latent_dim=seq_latent_dim, num_heads=seq_nhead,
            ff_size=seq_ff_size, num_layers=seq_num_layers, dropout=seq_dropout,
            clip_dim=clip_dim, cond_drop_prob=cond_drop_prob,
            use_first_pose=use_first_pose, fp_emb_dim=ae_dim,  # AE latent dim
            use_aspects=use_aspects,
            use_logline=use_logline, logline_emb_dim=logline_emb_dim,
        )

        # Diffuser
        self.diffuser = CineGenDiffuser(
            in_channels=ae_dim, model_channels=diff_model_channels,
            z_channels=seq_latent_dim, num_res_blocks=diff_num_res_blocks,
        )

        # EMA copies
        self.ema_sequencer = copy.deepcopy(self.sequencer)
        self.ema_diffuser = copy.deepcopy(self.diffuser)
        for p in self.ema_sequencer.parameters():
            p.requires_grad_(False)
        for p in self.ema_diffuser.parameters():
            p.requires_grad_(False)

        # Noise schedule
        self.scheduler = CosineScheduler(T=ddpm_T)

    def set_ae_encoder(self, ae_encoder):
        """Set frozen AE encoder for first_pose encoding (shared across sequencer + EMA)."""
        self.sequencer.set_ae_encoder(ae_encoder)
        self.ema_sequencer.set_ae_encoder(ae_encoder)

    @torch.no_grad()
    def update_ema(self):
        update_ema(self.ema_sequencer, self.sequencer, self.ema_decay)
        update_ema(self.ema_diffuser, self.diffuser, self.ema_decay)

    @torch.no_grad()
    def encode_text(self, texts):
        device = next(self.text_encoder.parameters()).device
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True,
                                truncation=True, max_length=77).to(device)
        out = self.text_encoder(**tokens)
        emb = self.clip_text_proj(out.pooler_output)  # (B, 512)
        return F.normalize(emb, dim=-1)

    def _build_text(self, motion_captions, cinematic_aspects):
        # Sep-encode mode: primary text channel is motion-only; logline goes through
        # a separate CLIP encode path injected into AdaLN conditioning.
        if self.use_logline:
            return motion_captions
        if self.text_mode == "logline":
            if cinematic_aspects and "logline_script" in cinematic_aspects:
                return cinematic_aspects["logline_script"]
            return [""] * len(motion_captions)
        elif self.text_mode == "combined":
            if cinematic_aspects and "logline_script" in cinematic_aspects:
                loglines = cinematic_aspects["logline_script"]
                return [f"[MOTION] {m} [SCRIPT] {l}" for m, l in zip(motion_captions, loglines)]
            return motion_captions
        return motion_captions

    def _build_logline_emb(self, motion_captions, cinematic_aspects):
        """Encode logline_script separately via CLIP for sep-encode mode.

        Returns None if use_logline is False or no logline available.
        """
        if not self.use_logline:
            return None
        B = len(motion_captions)
        if cinematic_aspects and "logline_script" in cinematic_aspects:
            loglines = cinematic_aspects["logline_script"]
        else:
            loglines = [""] * B
        # Reuse the same frozen CLIP encoder — distinct embedding from motion text
        return self.encode_text(loglines)

    def set_ae_decoder(self, ae_decoder):
        """Set frozen AE decoder for pose-space loss computation."""
        self.ae_decoder = ae_decoder
        for p in self.ae_decoder.parameters():
            p.requires_grad_(False)

    def compute_loss(self, z, motion_captions, cinematic_aspects=None,
                     first_pose=None, gt_traj_feat=None, traj_type="direction+speed",
                     pose_loss_weight=0.0, batch_multiplier=5, **kwargs):
        """
        Args:
            z: (B, L, ae_dim) encoded trajectory latent
            gt_traj_feat: (B, T, D) original trajectory features (for pose loss)
            pose_loss_weight: weight for pose-space loss (0=disabled)
            batch_multiplier: repeat each masked token with N different noise levels
                              , improves noise-level diversity)
        """
        B, L, D = z.shape
        device = z.device
        self.scheduler.to(device)

        texts = self._build_text(motion_captions, cinematic_aspects)
        text_emb = self.encode_text(texts)
        logline_emb = self._build_logline_emb(motion_captions, cinematic_aspects)

        # Random masking
        mask_ratio = torch.rand(1).item() * (1 - self.mask_ratio_min) + self.mask_ratio_min
        n_mask = max(1, int(L * mask_ratio))
        mask_indices = torch.rand(B, L, device=device).argsort(dim=1)[:, :n_mask]

        z_masked = z.clone()
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for b in range(B):
            z_masked[b, mask_indices[b]] = self.sequencer.mask_latent.squeeze()
            mask[b, mask_indices[b]] = True

        # Sequencer → hidden states for diffuser conditioning
        _, seq_hidden = self.sequencer(z_masked, text_emb, first_pose=first_pose,
                                        aspects=cinematic_aspects,
                                        logline_emb=logline_emb)

        # DDPM loss on masked positions with batch_multiplier
        # (each masked token gets batch_multiplier different noise levels)
        ctx = seq_hidden[mask]   # (N_masked, seq_latent_dim)
        target = z[mask]         # (N_masked, ae_dim)

        N_masked = ctx.shape[0]
        if batch_multiplier > 1:
            ctx = ctx.repeat(batch_multiplier, 1)
            target = target.repeat(batch_multiplier, 1)
            N_total = N_masked * batch_multiplier
        else:
            N_total = N_masked

        t = torch.randint(0, self.ddpm_T, (N_total,), device=device)
        noise = torch.randn_like(target)
        z_noisy, _ = self.scheduler.q_sample(target, t, noise)

        noise_pred = self.diffuser(z_noisy, t.float() / self.ddpm_T, ctx)
        ddpm_loss = F.mse_loss(noise_pred, noise)

        result = {"loss": ddpm_loss, "ddpm_loss": ddpm_loss}

        # Optional pose-space loss: decode predicted latents → reconstruct trajectory.
        # Only apply pose loss for samples where the noise level t is low enough that
        # one-step x0 prediction is reliable (t < ddpm_T / 3). At high t, ab.sqrt() in
        # the denominator becomes tiny and x0_pred explodes.
        if pose_loss_weight > 0 and hasattr(self, 'ae_decoder') and gt_traj_feat is not None:
            # Predict clean x0 from noise prediction (one-step denoising)
            ab = self.scheduler.alpha_bar[t]
            while ab.dim() < z_noisy.dim():
                ab = ab.unsqueeze(-1)
            x0_pred = (z_noisy - (1 - ab).sqrt() * noise_pred) / ab.sqrt()

            # Take only first N_masked rows when batch_multiplier > 1 (skip noise replicas)
            x0_pred_first = x0_pred[:N_masked] if batch_multiplier > 1 else x0_pred
            t_first = t[:N_masked] if batch_multiplier > 1 else t

            # Apply pose loss only on samples with t < T / 3 (low-noise → reliable x0 pred)
            low_t_mask = t_first < (self.ddpm_T // 3)
            if low_t_mask.any():
                # Subset to low-t predictions
                # x0_pred_first is of shape (N_masked, ae_dim) — we need to map back to (B, L, ae_dim)
                # via the original "mask" tensor. Build mask_lowt by intersecting position-mask with t-mask
                mask_lowt = mask.clone()
                # Find which (b, l) positions correspond to low_t_mask=True
                # mask is (B, L) bool. The N_masked positions are in row-major order matching mask.nonzero()
                pos_idx = mask.nonzero(as_tuple=False)  # (N_masked, 2) [b, l]
                kept_pos = pos_idx[low_t_mask.cpu()]
                # Reset mask_lowt and only set kept positions
                mask_lowt = torch.zeros_like(mask)
                mask_lowt[kept_pos[:, 0], kept_pos[:, 1]] = True

                # Reconstruct full latent only at low-t positions
                z_full = z.clone()
                z_full[mask_lowt] = x0_pred_first[low_t_mask]

                # Decode through frozen AE
                z_for_decode = z_full.permute(0, 2, 1)  # (B, ae_dim, L)
                decoded = self.ae_decoder(z_for_decode)  # (B, T_decoded, D)
                T_target = gt_traj_feat.shape[1]
                if decoded.shape[1] > T_target:
                    decoded = decoded[:, :T_target, :]
                elif decoded.shape[1] < T_target:
                    decoded = F.pad(decoded, (0, 0, 0, T_target - decoded.shape[1]))

                if traj_type == "direction+speed":
                    from utils.traj_reconstruct import compute_pose_loss
                    pose_loss = compute_pose_loss(decoded, gt_traj_feat, first_pose)
                else:
                    from utils.traj_reconstruct import compute_traj_pose_loss
                    pose_loss = compute_traj_pose_loss(decoded, gt_traj_feat)

                # Guard against NaN
                if torch.isfinite(pose_loss):
                    result["pose_loss"] = pose_loss
                    result["loss"] = ddpm_loss + pose_loss_weight * pose_loss

        # Update EMA
        if self.training:
            self.update_ema()

        return result

    @torch.no_grad()
    def sample(self, motion_captions, cinematic_aspects=None, L_z=None,
               n_ar_steps=None, ddpm_steps=50, cfg_scale=3.5,
               device="cuda", first_pose=None, **kwargs):
        """
        MAR-style sampling with dual CFG applied in the diffuser.

        1. Sequencer produces cond_logits and uncond_logits (text-conditioned vs not)
        2. Diffuser applies CFG: noise_pred = uncond + cfg_scale * (cond - uncond)
        """
        n_ar_steps = n_ar_steps or self.n_ar_steps
        B = len(motion_captions)
        self.scheduler.to(device)

        texts = self._build_text(motion_captions, cinematic_aspects)
        text_emb = self.encode_text(texts)
        logline_emb = self._build_logline_emb(motion_captions, cinematic_aspects)

        # Use EMA models for sampling
        seq = self.ema_sequencer
        diff = self.ema_diffuser

        z = seq.mask_latent.expand(B, L_z, -1).clone()

        for step in range(n_ar_steps):
            n_target = round(L_z * (step + 1) / n_ar_steps)
            n_unmasked = round(L_z * step / n_ar_steps)
            n_new = n_target - n_unmasked
            if n_new <= 0:
                continue

            # Sequencer: get BOTH conditional and unconditional outputs
            _, cond_hidden = seq(z, text_emb, first_pose=first_pose,
                                 aspects=cinematic_aspects,
                                 logline_emb=logline_emb)
            _, uncond_hidden = seq(z, text_emb, first_pose=first_pose,
                                   force_mask=True)

            # Find masked positions using cond_hidden variance
            is_masked = torch.all(
                torch.abs(z - seq.mask_latent.squeeze()) < 1e-6, dim=-1
            )
            scores = cond_hidden.var(dim=-1)
            scores[~is_masked] = float('inf')
            _, idx = scores.topk(n_new, dim=1, largest=False)

            # DDPM refine with CFG applied INSIDE the diffuser loop
            for b in range(B):
                pos = idx[b]
                cond_ctx = cond_hidden[b, pos]      # (n_new, seq_d_model)
                uncond_ctx = uncond_hidden[b, pos]   # (n_new, seq_d_model)
                x = torch.randn(n_new, self.ae_dim, device=device)

                T = self.ddpm_T
                step_size = max(1, T // ddpm_steps)
                for t_val in range(T - 1, -1, -step_size):
                    t = torch.full((n_new,), t_val, device=device, dtype=torch.long)
                    t_prev = torch.clamp(t - step_size, min=0)
                    t_float = t.float() / T

                    # Dual forward: conditional and unconditional
                    cond_pred = diff(x, t_float, cond_ctx)
                    uncond_pred = diff(x, t_float, uncond_ctx)

                    # CFG in diffuser 
                    noise_pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)

                    # DDPM posterior using the guided noise prediction
                    # x_0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)
                    ab = self.scheduler.alpha_bar[t_val]
                    x_0_pred = (x - (1 - ab).sqrt() * noise_pred) / ab.sqrt()
                    x = self.scheduler.posterior_sample(x, x_0_pred, t, t_prev)

                z[b, pos] = x

        return z


# ─────────────────────────────────────────────────────────────────────────────
# IdentityAE — pass-through "autoencoder" used by the published no-AE variant.
# Replaces PulpAE; the diffusion target stays in the raw 8-D dirspd / 9-D
# trajectory feature space instead of a learned 64-D latent.
# ─────────────────────────────────────────────────────────────────────────────

class IdentityEncoder(nn.Module):
    """Mean-pools (B, D, T) → (B, D, 1) over time.

    Used only by ``CineGenSequencer._encode_first_pose``: first_pose is replicated
    to 4 frames and fed through this encoder so its output shape matches what
    PulpAE's encoder would produce (single token of dim D). Mean-pooling 4 copies
    of the same vector returns the vector itself, so this is effectively a
    no-op in the no-AE setting.
    """
    def forward(self, x):  # x: (B, D, T)
        return x.mean(dim=-1, keepdim=True)


class IdentityAE(nn.Module):
    """Stand-in for the trajectory autoencoder when no compression is used.

    - ``encode``: just permutes (B, T, D) → (B, D, T); no temporal stride, no
      channel projection. CineGen then operates directly on the 8-D dirspd
      (or 9-D trajectory) feature stream at full sequence length.
    - ``decode``: permute back.
    - ``encoder`` / ``decoder`` attributes are present for API compatibility
      with the real PulpAE wrapper.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = input_dim
        self.encoder = IdentityEncoder()
        self.decoder = nn.Identity()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, D, T)."""
        return x.permute(0, 2, 1)

    def decode(self, z: torch.Tensor, target_len: int = None) -> torch.Tensor:
        """z: (B, D, T) → (B, T, D)."""
        return z.permute(0, 2, 1)
