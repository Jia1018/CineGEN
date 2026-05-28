"""
CLaTr alignment model — exact reimplementation of E.T.'s architecture.

Architecture (from E.T./DIRECTOR/clatr/):
  - Trajectory encoder: ACTOR-style transformer VAE (9D input → 256D latent)
  - Text encoder:       ACTOR-style transformer VAE (clip_dim input → 256D latent)
  - Trajectory decoder: ACTOR-style transformer (256D latent → 9D output)
  - Loss: reconstruction (SmoothL1) + KL + latent alignment (SmoothL1) + InfoNCE

Both encoders map to the same shared 256-dim latent space, enabling
cross-modal retrieval (text ↔ trajectory).

References:
  - E.T. CLaTr: https://github.com/exceptional-trajectories
  - TEMOS:      https://mathis.petrovich.fr/temos/
  - ACTOR:      https://mathis.petrovich.fr/actor/
  - InfoNCE:    https://arxiv.org/abs/1807.03748
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)   # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:x.shape[1]].unsqueeze(0)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# ACTOR-style encoder
# ---------------------------------------------------------------------------

class ACTOREncoder(nn.Module):
    """
    Transformer encoder that produces a latent distribution (μ, log σ²).

    Input tokens:  [learnable_tokens(1 or 2) | projected_input]
    The first 1 (det) or 2 (VAE) output tokens are taken as the latent.
    """

    def __init__(
        self,
        input_dim:  int,
        latent_dim: int = 256,
        ff_size:    int = 1024,
        num_layers: int = 6,
        num_heads:  int = 4,
        dropout:    float = 0.1,
        activation: str = "gelu",
        vae:        bool = True,
    ):
        super().__init__()
        self.vae       = vae
        self.n_tokens  = 2 if vae else 1
        self.latent_dim = latent_dim

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.pos_enc    = PositionalEncoding(latent_dim, dropout=dropout)
        self.latent_tokens = nn.Parameter(torch.randn(self.n_tokens, latent_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation=activation, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        x:    torch.Tensor,       # (B, T, input_dim)
        mask: torch.Tensor,       # (B, T) True = valid
    ) -> torch.Tensor:            # (B, n_tokens, latent_dim)
        B, T, _ = x.shape

        tokens = self.input_proj(x)                                # (B, T, D)
        tokens = self.pos_enc(tokens)

        # Prepend learnable latent tokens
        lat = self.latent_tokens.unsqueeze(0).expand(B, -1, -1)   # (B, n_tokens, D)
        seq = torch.cat([lat, tokens], dim=1)                     # (B, n_tokens+T, D)

        # src_key_padding_mask: True = IGNORE (PyTorch convention)
        # Our mask: True = valid → invert, prepend False for latent tokens
        pad_mask = ~mask                                           # (B, T) True=pad
        lat_mask = torch.zeros(B, self.n_tokens, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([lat_mask, pad_mask], dim=1)        # (B, n_tokens+T)

        out = self.transformer(seq, src_key_padding_mask=full_mask)
        return out[:, :self.n_tokens]                             # (B, n_tokens, D)


# ---------------------------------------------------------------------------
# ACTOR-style decoder
# ---------------------------------------------------------------------------

class ACTORDecoder(nn.Module):
    """
    Transformer decoder that reconstructs a sequence from a latent vector.

    Memory: latent vector (1 token).
    Queries: learned positional time queries.
    """

    def __init__(
        self,
        output_dim: int,
        latent_dim: int = 256,
        ff_size:    int = 1024,
        num_layers: int = 6,
        num_heads:  int = 4,
        dropout:    float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 300,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.pos_enc   = PositionalEncoding(latent_dim, max_len=max_seq_len, dropout=dropout)
        self.time_query = nn.Parameter(torch.randn(1, max_seq_len, latent_dim) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
            dropout=dropout, activation=activation, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_proj    = nn.Linear(latent_dim, output_dim)

    def forward(
        self,
        z:    torch.Tensor,   # (B, latent_dim)
        mask: torch.Tensor,   # (B, T) True = valid
    ) -> torch.Tensor:        # (B, T, output_dim)
        B, T = mask.shape

        # Queries: positional time queries
        queries = self.time_query[:, :T].expand(B, -1, -1)     # (B, T, D)
        queries = self.pos_enc(queries)

        # Memory: latent vector as single token
        memory = z.unsqueeze(1)                                 # (B, 1, D)

        out = self.transformer(queries, memory)                 # (B, T, D)
        out = self.out_proj(out)                                # (B, T, output_dim)

        # Zero out padded positions
        out = out * mask.unsqueeze(-1).float()
        return out


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class KLLoss(nn.Module):
    """Closed-form KL divergence between two diagonal Gaussians."""
    def forward(
        self,
        q: tuple[torch.Tensor, torch.Tensor],   # (mu_q, logvar_q)
        p: tuple[torch.Tensor, torch.Tensor],   # (mu_p, logvar_p)
    ) -> torch.Tensor:
        mu_q, lv_q = q
        mu_p, lv_p = p
        kl = 0.5 * (
            lv_p - lv_q
            + (lv_q.exp() + (mu_q - mu_p).pow(2)) / lv_p.exp().clamp(min=1e-8)
            - 1.0
        )
        return kl.mean()


class InfoNCEFiltered(nn.Module):
    """
    Bidirectional InfoNCE loss with false-negative filtering via text self-similarity.
    Matches E.T.'s InfoNCE_with_filtering implementation.
    """

    def __init__(self, temperature: float = 0.1, threshold_selfsim: float = 0.995):
        super().__init__()
        self.temperature = temperature
        # Convert from [0,1] cosine score space to [-1,1] similarity space
        self.threshold   = 2.0 * threshold_selfsim - 1.0

    def forward(
        self,
        t_latents: torch.Tensor,    # (B, D)  text latents
        m_latents: torch.Tensor,    # (B, D)  traj latents
        sent_token: torch.Tensor,   # (B, 77) CLIP token ids for false-neg filtering
    ) -> torch.Tensor:
        B = t_latents.shape[0]

        t = F.normalize(t_latents, dim=-1)
        m = F.normalize(m_latents, dim=-1)
        sim = (t @ m.T) / self.temperature                     # (B, B)

        # False-negative filtering: mask pairs with very similar captions
        if sent_token is not None and self.threshold < 1.0:
            tok = sent_token.float()
            tok = F.normalize(tok, dim=-1)
            selfsim = tok @ tok.T                              # (B, B)
            # Mask out near-duplicate captions (excluding diagonal = always kept)
            diag_mask = torch.eye(B, dtype=torch.bool, device=sim.device)
            false_neg = (selfsim > self.threshold) & ~diag_mask
            sim = sim.masked_fill(false_neg, float("-inf"))

        labels = torch.arange(B, device=sim.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0
        return loss


# ---------------------------------------------------------------------------
# CLaTr model
# ---------------------------------------------------------------------------

class CLaTr(nn.Module):
    """
    CLaTr: Contrastive Language-Trajectory model.

    Trains two cross-modal VAE encoders (trajectory, text) in a shared
    256-dim latent space with reconstruction + KL + latent + InfoNCE losses.

    Args:
        traj_input_dim: Input dim per frame (9 for rot6D+trans).
        text_input_dim: CLIP feature dim (768 for ViT-L/14, 512 for ViT-B/32).
        latent_dim:     Shared latent space dimension.
        ff_size:        Transformer feedforward hidden size.
        num_layers:     Transformer layers in encoder and decoder.
        num_heads:      Attention heads.
        dropout:        Dropout rate.
        max_seq_len:    Maximum trajectory length (for decoder queries).
        temperature:    InfoNCE temperature.
        threshold_selfsim: False-negative filtering threshold.
        lmd:            Loss weights dict.
        fact:           VAE sampling variance scale (1.0 = full).
        sample_mean:    Use μ instead of sampling (for eval).
    """

    def __init__(
        self,
        traj_input_dim: int   = 9,
        text_input_dim: int   = 768,
        latent_dim:     int   = 256,
        ff_size:        int   = 1024,
        num_layers:     int   = 6,
        num_heads:      int   = 4,
        dropout:        float = 0.1,
        max_seq_len:    int   = 300,
        temperature:    float = 0.1,
        threshold_selfsim: float = 0.995,
        lmd: dict = None,
        fact:        float = 1.0,
        sample_mean: bool  = False,
    ):
        super().__init__()
        self.latent_dim  = latent_dim
        self.fact        = fact
        self.sample_mean = sample_mean
        self.lmd = lmd or {
            "recons":      1.0,
            "kl":          1e-5,
            "latent":      1e-5,
            "contrastive": 0.1,
        }

        # Encoders
        self.traj_encoder = ACTOREncoder(
            input_dim=traj_input_dim, latent_dim=latent_dim,
            ff_size=ff_size, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout, vae=True,
        )
        self.text_encoder = ACTOREncoder(
            input_dim=text_input_dim, latent_dim=latent_dim,
            ff_size=ff_size, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout, vae=True,
        )

        # Decoder (trajectory only)
        self.traj_decoder = ACTORDecoder(
            output_dim=traj_input_dim, latent_dim=latent_dim,
            ff_size=ff_size, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout, max_seq_len=max_seq_len,
        )

        # Losses
        self.recons_loss     = nn.SmoothL1Loss()
        self.kl_loss         = KLLoss()
        self.contrastive_loss = InfoNCEFiltered(temperature, threshold_selfsim)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _sample(
        self,
        encoded: torch.Tensor,   # (B, 2, latent_dim)
    ) -> tuple[torch.Tensor, tuple]:
        mu, logvar = encoded.unbind(dim=1)                     # each (B, D)
        if self.sample_mean or not self.training:
            return mu, (mu, logvar)
        std = (0.5 * logvar).exp()
        z   = mu + self.fact * std * torch.randn_like(std)
        return z, (mu, logvar)

    def encode_traj(
        self,
        traj_feat:    torch.Tensor,   # (B, T, 9)
        padding_mask: torch.Tensor,   # (B, T) True=valid
    ) -> tuple[torch.Tensor, tuple]:
        enc = self.traj_encoder(traj_feat, padding_mask)       # (B, 2, D)
        return self._sample(enc)

    def encode_text(
        self,
        caption_feat: torch.Tensor,   # (B, 77, clip_dim)
    ) -> tuple[torch.Tensor, tuple]:
        # Text has no padding (always 77 tokens, padded by CLIP tokenizer)
        B = caption_feat.shape[0]
        mask = torch.ones(B, caption_feat.shape[1], dtype=torch.bool,
                          device=caption_feat.device)
        enc = self.text_encoder(caption_feat, mask)            # (B, 2, D)
        return self._sample(enc)

    # ------------------------------------------------------------------
    # Forward / loss
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict) -> dict:
        traj      = batch["traj_feat"]      # (B, T, 9)
        mask      = batch["padding_mask"]   # (B, T)
        cap_feat  = batch["caption_feat"]   # (B, 77, clip_dim)
        sent_tok  = batch["sent_token"]     # (B, 77)

        # Encode both modalities
        m_z, m_dist = self.encode_traj(traj, mask)    # traj → latent
        t_z, t_dist = self.encode_text(cap_feat)       # text → latent

        # Decode both latents back to trajectory
        m_recon = self.traj_decoder(m_z, mask)         # traj→traj recon
        t_recon = self.traj_decoder(t_z, mask)         # text→traj recon

        losses = {}

        # Reconstruction: both decodings should match the real trajectory
        losses["recons"] = (
            self.recons_loss(m_recon, traj) +
            self.recons_loss(t_recon, traj)
        )

        # KL divergence (4-way symmetric, same as TEMOS)
        ref_mu  = torch.zeros_like(m_dist[0])
        ref_lv  = torch.zeros_like(m_dist[1])
        ref_dist = (ref_mu, ref_lv)
        losses["kl"] = (
            self.kl_loss(t_dist, m_dist) +
            self.kl_loss(m_dist, t_dist) +
            self.kl_loss(m_dist, ref_dist) +
            self.kl_loss(t_dist, ref_dist)
        )

        # Latent manifold alignment
        losses["latent"] = F.smooth_l1_loss(t_z, m_z)

        # Contrastive (InfoNCE with false-neg filtering)
        losses["contrastive"] = self.contrastive_loss(t_z, m_z, sent_tok)

        losses["loss"] = sum(
            self.lmd.get(k, 0.0) * v for k, v in losses.items() if k != "loss"
        )
        return losses

    # ------------------------------------------------------------------
    # Retrieval helpers (eval)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_traj_embedding(
        self,
        traj_feat:    torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.sample_mean = True
        z, _ = self.encode_traj(traj_feat, padding_mask)
        self.sample_mean = False
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def get_text_embedding(
        self,
        caption_feat: torch.Tensor,
    ) -> torch.Tensor:
        self.sample_mean = True
        z, _ = self.encode_text(caption_feat)
        self.sample_mean = False
        return F.normalize(z, dim=-1)
