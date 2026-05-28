"""
Text and image encoders for conditioning signals.

TextEncoder:   CLIP text model → (B, seq_len, dim) or (B, dim) pooled
ImageEncoder:  CLIP vision model → (B, num_patches+1, dim) or (B, dim) pooled
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor


CLIP_MODEL_ID = "openai/clip-vit-large-patch14"   # 768-dim text, 1024-dim vision


class TextEncoder(nn.Module):
    """
    Encodes text prompts via CLIP text encoder.

    Returns token-level features (B, seq_len, text_dim) or pooled (B, text_dim).
    A linear projection maps to out_dim if specified.
    """

    def __init__(
        self,
        model_id: str = CLIP_MODEL_ID,
        out_dim: int = 512,
        max_length: int = 77,
        pooled: bool = False,
        freeze: bool = True,
    ):
        super().__init__()
        self.max_length = max_length
        self.pooled = pooled

        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)
        self.model     = CLIPTextModel.from_pretrained(model_id)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        text_dim = self.model.config.hidden_size   # 768 for ViT-L/14
        self.proj = nn.Linear(text_dim, out_dim) if out_dim != text_dim else nn.Identity()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Args:  texts: list of B strings
        Returns:
            If pooled: (B, out_dim)
            Else:      (B, max_length, out_dim)
        """
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.model.parameters())):
            out = self.model(**tokens)

        if self.pooled:
            feat = out.pooler_output   # (B, text_dim)
        else:
            feat = out.last_hidden_state  # (B, seq_len, text_dim)

        return self.proj(feat)


class ImageEncoder(nn.Module):
    """
    Encodes an RGB image via CLIP vision encoder.

    Returns patch-level features (B, num_patches+1, vision_dim) or pooled (B, vision_dim).
    """

    def __init__(
        self,
        model_id: str = CLIP_MODEL_ID,
        out_dim: int = 512,
        pooled: bool = False,
        freeze: bool = True,
    ):
        super().__init__()
        self.pooled = pooled

        self.processor = CLIPImageProcessor.from_pretrained(model_id)
        self.model      = CLIPVisionModel.from_pretrained(model_id)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        vision_dim = self.model.config.hidden_size   # 1024 for ViT-L/14
        self.proj = nn.Linear(vision_dim, out_dim) if out_dim != vision_dim else nn.Identity()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:  images: (B, 3, H, W) float [0,1] or (B, 3, H, W) uint8
        Returns:
            If pooled: (B, out_dim)
            Else:      (B, num_patches+1, out_dim)
        """
        # Preprocess using the CLIP processor norms
        pixel_values = self.processor(
            images=[img for img in images.cpu()],
            return_tensors="pt",
            do_rescale=False,
        ).pixel_values.to(self.device)

        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.model.parameters())):
            out = self.model(pixel_values=pixel_values)

        if self.pooled:
            feat = out.pooler_output       # (B, vision_dim)
        else:
            feat = out.last_hidden_state   # (B, num_patches+1, vision_dim)

        return self.proj(feat)


class DepthEncoder(nn.Module):
    """
    Encodes a single-channel depth map via CLIP vision encoder.

    Depth is a 1-channel float map (relative or metric). It is normalised to
    [0, 1] and repeated to 3 channels so the pretrained CLIP ViT can process it
    without any architectural changes — the same strategy used in GenDoP.

    Returns patch-level features (B, num_patches+1, out_dim) or pooled (B, out_dim).
    The non-pooled output (default) preserves spatial layout, which is useful for
    direction generation (e.g. depth gradient indicates scene structure and
    likely camera motion direction).
    """

    def __init__(
        self,
        model_id: str = CLIP_MODEL_ID,
        out_dim:  int = 512,
        pooled:   bool = False,
        freeze:   bool = True,
    ):
        super().__init__()
        self.pooled = pooled
        # Reuse the same CLIP vision architecture; depth is fed as pseudo-RGB
        self.processor = CLIPImageProcessor.from_pretrained(model_id)
        self.model      = CLIPVisionModel.from_pretrained(model_id)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        vision_dim = self.model.config.hidden_size
        self.proj = nn.Linear(vision_dim, out_dim) if out_dim != vision_dim else nn.Identity()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: (B, 1, H, W) or (B, H, W) float — raw depth values (any scale)
        Returns:
            If pooled: (B, out_dim)
            Else:      (B, num_patches+1, out_dim)
        """
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)           # (B, 1, H, W)

        # Normalise each map independently to [0, 1]
        B = depth.shape[0]
        d_flat = depth.reshape(B, -1)
        d_min  = d_flat.min(dim=1).values.reshape(B, 1, 1, 1)
        d_max  = d_flat.max(dim=1).values.reshape(B, 1, 1, 1)
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)   # (B, 1, H, W) in [0,1]

        # Repeat to 3 channels to match CLIP expected input
        depth_rgb = depth_norm.repeat(1, 3, 1, 1)               # (B, 3, H, W)

        pixel_values = self.processor(
            images=[img for img in depth_rgb.cpu()],
            return_tensors="pt",
            do_rescale=False,
        ).pixel_values.to(self.device)

        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.model.parameters())):
            out = self.model(pixel_values=pixel_values)

        feat = out.pooler_output if self.pooled else out.last_hidden_state
        return self.proj(feat)


class MultiAspectContentEncoder(nn.Module):
    """
    Encodes each cinematic aspect as a separate token via a shared CLIP text encoder.

    The 5 aspects (from cinematic_data) are encoded independently so the
    SpeedModel's cross-attention can learn which aspects matter most for speed:

        logline_script      — rich scene description (most informative but noisy)
        macro_type          — "Interior (Restricted)" / "Exterior (Open)"
                              → predicts translation scale
        setting_class       — "Urban/City", "Domestic/Residential", ...
                              → predicts environment scale
        subject_composition — "Single-Character", "Group/Crowd", ...
                              → predicts what is being tracked
        genre_vibe          — "Action / Thriller", "Drama / Romance", ...
                              → strong prior on camera speed

    Returns:
        (B, num_aspects, out_dim)  — one token per aspect, in the order above
    """

    ASPECT_KEYS = [
        "logline_script",
        "macro_type",
        "setting_class",
        "subject_composition",
        "genre_vibe",
    ]

    def __init__(
        self,
        model_id: str = CLIP_MODEL_ID,
        out_dim:  int = 512,
        freeze:   bool = True,
    ):
        super().__init__()
        # Shared text encoder for all aspects (weights shared, batched together)
        self.encoder = TextEncoder(
            model_id=model_id, out_dim=out_dim, pooled=True, freeze=freeze
        )
        self.num_aspects = len(self.ASPECT_KEYS)

    def forward(
        self,
        aspects: dict[str, list[str]],
        keys: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            aspects: dict mapping aspect_key → list of B strings
                     (as returned by dataset collate_fn for "cinematic_aspects")
            keys:    Optional subset of ASPECT_KEYS to encode.
                     If None, all 5 ASPECT_KEYS are encoded.
        Returns:
            (B, n_keys, out_dim)  where n_keys = len(keys or ASPECT_KEYS)
        """
        use_keys = keys if keys is not None else self.ASPECT_KEYS
        B = len(next(iter(aspects.values())))
        all_texts = []
        for key in use_keys:
            all_texts.extend(aspects.get(key, [""] * B))

        all_feats = self.encoder(all_texts)                         # (B*n_keys, out_dim)
        all_feats = all_feats.reshape(len(use_keys), B, -1).permute(1, 0, 2)
        return all_feats                                             # (B, n_keys, out_dim)
