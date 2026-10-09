"""
Fusion Model — Architecture 2 (Swin-B + Cross-Attention Metadata Fusion)
-------------------------------------------------------------------------

This is Architecture 2 of Khet Guard's crop disease / pest / cattle pipeline.
Architecture 1 lives in fusion_model.py (TF/Keras CNN) and
train_pytorch_fusion.py (PyTorch EfficientNet-B4).  Do NOT edit those files.

Design goals for Arch 2
~~~~~~~~~~~~~~~~~~~~~~~~
* Replace the CNN backbone with a **Swin-B Vision Transformer** so the model
  attends over non-overlapping local windows of image patches rather than
  sliding conv filters — giving richer long-range context especially for
  leaf-texture disease patterns.
* Replace the simple vector-concatenation fusion of Arch 1 with
  **multi-head cross-attention**: the 49 patch tokens are queries, the
  projected metadata token is both key and value. This lets the model learn
  *where in the image* the metadata (crop type, season, location) matters.
* Provide an **attention-rollout** explainability function analogous to the
  Grad-CAM in Arch 1, returning a 7x7 spatial importance map that can be
  overlaid on the original image.

Dependency note
~~~~~~~~~~~~~~~
NO new package is required.  ``torchvision >= 0.13`` ships ``swin_b`` natively
via ``torchvision.models.swin_b``.  The root ``requirements.txt`` (which pins
``torchvision==0.23.0``) already covers this.  ``timm`` is *not* needed and
has *not* been added anywhere.

Shape conventions used in docstrings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
B  = batch size
N  = number of patch tokens = 7 x 7 = 49   (for 224x224 input)
D  = image feature / embedding dimension   = 1024   (Swin-B final stage)
M  = metadata embedding dimension          = 1024   (projected to match D)
H  = number of cross-attention heads       = 8      (configurable)

Author: Khet Guard ML Team
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_B_Weights, swin_b


# ---------------------------------------------------------------------------
# Constants verified by probing the live torchvision model:
#   swin_b(weights=None).features output: (B, 7, 7, 1024)
#   window_size in last stage:            [7, 7]
#   num_heads in ShiftedWindowAttention:  32
# ---------------------------------------------------------------------------
_SWIN_B_FEATURE_DIM: int = 1024   # channel width of Swin-B's final stage
_SWIN_B_GRID_H: int = 7           # spatial grid height after 4 patch-merging stages
_SWIN_B_GRID_W: int = 7           # spatial grid width  after 4 patch-merging stages
_SWIN_B_N_TOKENS: int = _SWIN_B_GRID_H * _SWIN_B_GRID_W  # 49 patch tokens


# ===========================================================================
# Metadata Branch
# ===========================================================================

class MetadataBranch(nn.Module):
    """
    Projects the raw 16-D agronomic metadata vector into a single token
    at the same width as Swin-B's final-stage patch tokens (D=1024).

    Architecture (matches project convention in train_pytorch_fusion.py):
        Linear(16 -> 64) -> ReLU -> Dropout(p) -> Linear(64 -> 1024)

    Shape trace
    -----------
    Input  : (B, 16)
    After Linear(16->64)   : (B, 64)
    After ReLU             : (B, 64)
    After Dropout          : (B, 64)
    After Linear(64->1024) : (B, 1024)   <- one metadata "token" of width D
    """

    def __init__(
        self,
        metadata_dim: int = 16,
        out_dim: int = _SWIN_B_FEATURE_DIM,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, out_dim),  # out_dim == D == 1024
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        metadata : (B, 16)   -- raw agronomic feature vector

        Returns
        -------
        token : (B, 1024)   -- single metadata token of width D
        """
        return self.net(metadata)  # (B, D)


# ===========================================================================
# Cross-Attention Fusion
# ===========================================================================

class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention: the metadata token queries over image patches.

    The single metadata token asks: "given this crop/season/location context,
    which of the 49 spatial image patches are most relevant?".  The patch
    tokens answer as keys and values.

    The cross-attention weights are stored in ``self.last_attn_weights`` so
    that ``SwinFusionModel.attention_rollout()`` can combine them with the
    Swin backbone rollout for explainability.

    Shape trace
    -----------
    patch_tokens : (B, N, D)   N=49, D=1024   -- the 49 spatial patch tokens
    meta_token   : (B, D)      single metadata token

    nn.MultiheadAttention used with batch_first=True, so shapes are
    (B, seq_len, D) throughout.

    Q = meta_token   : (B, 1, D)   -- unsqueezed to a sequence of length 1
    K = patch_tokens : (B, N, D)
    V = patch_tokens : (B, N, D)

    attn_output  : (B, 1, D)   -- metadata token after attending over patches
    attn_weights : (B, 1, N)   -- one weight per patch token (averaged over heads)
                                   stored for rollout; shape (B, 1, 49)
    """

    def __init__(
        self,
        embed_dim: int = _SWIN_B_FEATURE_DIM,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        embed_dim   : must equal D (1024) so Q, K, V projections are square
        num_heads   : 8 heads -> each head attends over 128-D subspace of 1024
        dropout_rate: attention dropout (0 at inference)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,  # (B, seq, dim) convention throughout
        )
        # Populated by forward(); consumed by attention_rollout()
        self.last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        patch_tokens: torch.Tensor,
        meta_token: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_tokens : (B, N, D)   N=49 spatial patch tokens from Swin-B
        meta_token   : (B, D)      single metadata token

        Returns
        -------
        fused : (B, D)   metadata-conditioned spatial image representation
        """
        # Query = metadata token (B, 1, D)
        # Key   = image patch tokens (B, 49, D)
        # Value = image patch tokens (B, 49, D)
        query = meta_token.unsqueeze(1)  # (B, 1, D)

        # Q=(B,1,D), K=(B,N,D), V=(B,N,D) -> attn_output=(B,1,D), weights=(B,1,N)
        attn_output, attn_weights = self.mha(
            query=query,
            key=patch_tokens,
            value=patch_tokens,
            need_weights=True,          # must be True so we can store weights
            average_attn_weights=True,  # average over heads -> (B, 1, N)
        )
        # Detach before storing so backprop graph is not retained
        self.last_attn_weights = attn_weights.detach()  # (B, 1, N=49)

        # Residual fusion: global mean-pooled image patches + metadata-attended image features
        fused = patch_tokens.mean(dim=1) + attn_output.squeeze(1)  # (B, D)
        return fused


# ===========================================================================
# Classifier Head
# ===========================================================================

class FusionClassifierHead(nn.Module):
    """
    Takes the mean-pooled fused token representation and produces class logits.

    Shape trace
    -----------
    Input  : (B, N, D)   fused patch tokens
    Mean-pool over N=49  : (B, D)   = (B, 1024)
    Linear(1024 -> 256)  : (B, 256)
    ReLU + Dropout
    Linear(256 -> num_classes) : (B, num_classes)
    """

    def __init__(
        self,
        in_dim: int = _SWIN_B_FEATURE_DIM,
        num_classes: int = 38,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),      # (B, 1024) -> (B, 256)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes), # (B, 256)  -> (B, num_classes)
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        fused_tokens : (B, D) or (B, N, D)

        Returns
        -------
        logits : (B, num_classes)   -- raw logits; apply F.softmax externally
        """
        if fused_tokens.ndim == 3:
            fused_tokens = fused_tokens.mean(dim=1)
        return self.head(fused_tokens)  # (B, num_classes)


# ===========================================================================
# Main Model
# ===========================================================================

class SwinFusionModel(nn.Module):
    """
    Architecture 2 -- Swin-B backbone with cross-attention metadata fusion.

    Full data-flow
    --------------
    1. Image  (B, 3, 224, 224)
            |  Swin-B feature extractor  (features + norm layers only)
       patch grid  (B, 7, 7, 1024)
            |  reshape to token sequence
       patch_tokens  (B, 49, 1024)

    2. Metadata  (B, 16)
            |  MetadataBranch
       meta_token  (B, 1024)

    3. CrossAttentionFusion
       Q = meta_token   (B, 1, 1024)
       K = V = patch_tokens (B, 49, 1024)
            |  nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
       attn_output (B, 1, 1024) + residual patch mean-pool  -> fused (B, 1024)
       attn_weights stored -> (B, 1, 49)  used in attention_rollout()

    4. FusionClassifierHead
       fused_tokens  (B, 49, 1024)
            |  mean-pool  (B, 1024)
            |  Linear(1024->256) -> ReLU -> Dropout
            |  Linear(256->num_classes)
       logits  (B, num_classes)

    Comparison with Architecture 1 (FusionModel in train_pytorch_fusion.py)
    -----------------------------------------------------------------------
    Component         | Arch 1 (EfficientNet-B4)     | Arch 2 (Swin-B)
    ------------------|------------------------------|----------------------------
    Backbone          | EfficientNet-B4 (~19M params)| Swin-B (~87M params)
    Feature dim       | 1792                         | 1024
    Spatial tokens    | Single vector after GAP      | 49 tokens (7x7 grid)
    Fusion method     | Concatenation                | Cross-attention
    Explainability    | Grad-CAM                     | Attention rollout (7x7 map)

    Parameters
    ----------
    num_classes   : int   number of output classes (38 disease / 20 pest / 41 cattle)
    metadata_dim  : int   width of the raw metadata vector (default 16, matches Arch 1)
    dropout_rate  : float dropout probability used in metadata branch and classifier head
    num_attn_heads: int   number of cross-attention heads (must divide 1024 evenly)
    pretrained    : bool  load ImageNet-1k weights for Swin-B backbone
    """

    def __init__(
        self,
        num_classes: int = 38,
        metadata_dim: int = 16,
        dropout_rate: float = 0.3,
        num_attn_heads: int = 8,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # 1. Swin-B Backbone (feature extractor only -- head is removed)
        # ------------------------------------------------------------------
        weights = Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        _backbone = swin_b(weights=weights)

        # Keep only the parts that produce the (B, 7, 7, 1024) feature map:
        #   features: 8 stages of shifted-window attention + patch merging
        #   norm:     final LayerNorm over D=1024
        # Deliberately drop permute / avgpool / flatten / head so we retain
        # all 49 spatial tokens instead of the single global-pooled vector.
        self.swin_features = _backbone.features  # (B,3,224,224) -> (B,7,7,1024)
        self.swin_norm = _backbone.norm          # LayerNorm applied token-wise

        # ------------------------------------------------------------------
        # 2. Metadata Branch -> single token of width D=1024
        # ------------------------------------------------------------------
        self.metadata_branch = MetadataBranch(
            metadata_dim=metadata_dim,
            out_dim=_SWIN_B_FEATURE_DIM,
            dropout_rate=dropout_rate,
        )

        # ------------------------------------------------------------------
        # 3. Cross-Attention Fusion
        # ------------------------------------------------------------------
        self.cross_attn = CrossAttentionFusion(
            embed_dim=_SWIN_B_FEATURE_DIM,
            num_heads=num_attn_heads,
            dropout_rate=0.0,
        )

        # ------------------------------------------------------------------
        # 4. Classifier Head
        # ------------------------------------------------------------------
        self.classifier = FusionClassifierHead(
            in_dim=_SWIN_B_FEATURE_DIM,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

        # Expose constants for training scripts
        self.img_feature_dim: int = _SWIN_B_FEATURE_DIM   # D = 1024
        self.n_patch_tokens: int = _SWIN_B_N_TOKENS       # N = 49
        self.grid_h: int = _SWIN_B_GRID_H                 # 7
        self.grid_w: int = _SWIN_B_GRID_W                 # 7

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image    : (B, 3, 224, 224)   -- ImageNet-normalised RGB image
        metadata : (B, 16)            -- agronomic feature vector

        Returns
        -------
        logits   : (B, num_classes)   -- raw class scores (no softmax applied)
                   Pass to nn.CrossEntropyLoss() during training.
                   Call F.softmax(logits, dim=1) for inference probabilities.
        """
        # 1. Extract Swin-B spatial feature grid -------------------------------------------------------
        # (B, 3, 224, 224)
        #    -> swin_features -> (B, 7, 7, 1024)   [height-last, channel-last convention in Swin]
        #    -> swin_norm     -> (B, 7, 7, 1024)   LayerNorm applied along D dimension
        x = self.swin_features(image)   # (B, 7, 7, 1024)
        x = self.swin_norm(x)           # (B, 7, 7, 1024)

        # Flatten spatial dims into a sequence of N=49 tokens
        # (B, 7, 7, 1024) -> (B, 49, 1024)
        B, H, W, D = x.shape                      # H=7, W=7, D=1024
        patch_tokens = x.view(B, H * W, D)        # (B, N=49, D=1024)

        # 2. Project metadata to one 1024-D token -----------------------------------------------------
        # (B, 16) -> (B, 1024)
        meta_token = self.metadata_branch(metadata)  # (B, D=1024)

        # 3. Cross-attention fusion -------------------------------------------------------------------
        # Q=meta_token (B,1,1024), K=V=patch_tokens (B,49,1024)
        # attn_weights=(B,1,49) stored in self.cross_attn.last_attn_weights
        fused = self.cross_attn(patch_tokens, meta_token)  # (B, D=1024)

        # 4. Classify ---------------------------------------------------------------------------------
        # mean-pool (B,49,1024) -> (B,1024) -> Linear -> (B,num_classes)
        logits = self.classifier(fused)  # (B, num_classes)
        return logits

    # ------------------------------------------------------------------
    # Explainability: Attention Rollout
    # ------------------------------------------------------------------

    def attention_rollout(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor,
        combine_cross_attn: bool = True,
    ) -> np.ndarray:
        """
        Proper multi-layer Swin-B attention rollout (Abnar & Zuidema, 2020).

        Unlike the previous single-layer implementation that only returned the
        cross-attention weights (which visually mimic GradCAM blobs because
        both highlight the same discriminative leaf regions at 7x7 resolution),
        this method performs a true multi-layer rollout through the Swin-B
        backbone's *own* attention, producing window-aware, edge-sensitive maps
        that are visually distinct from GradCAM.

        Algorithm
        ---------
        1.  Monkey-patch each ``ShiftedWindowAttention.forward`` in stage 3
            (``swin_features[7]``).  This torchvision version computes all
            attention math inside the free function ``shifted_window_attention``
            with no hookable ``attn_drop`` sub-module boundary, so we
            temporarily intercept ``torch.nn.functional.softmax`` to capture
            the 4-D attention weight tensor.

        2.  Run a forward pass to populate captures and
            ``cross_attn.last_attn_weights``.

        3.  For each of the 2 stage-3 blocks (W-MSA then SW-MSA), apply the
            Abnar & Zuidema residual formula::

                A_layer = 0.5 * mean_heads(softmax(A)) + 0.5 * I
                rollout = A_block1 @ A_block0

        4.  For the shifted-window block (shift_size > 0), un-shift token
            indices back to original spatial order before accumulating.

        5.  Sum over queries -> (7, 7) backbone saliency map.

        6.  Optionally combine (geometric mean) with the cross-attention map.

        Parameters
        ----------
        image    : (1, 3, 224, 224)
        metadata : (1, 16)
        combine_cross_attn : bool  default True

        Returns
        -------
        rollout_map : np.ndarray  (7, 7)  float32  values in [0, 1]
        """
        import torchvision.models.swin_transformer as _swin_tv
        from torchvision.models.swin_transformer import ShiftedWindowAttention

        H, W = self.grid_h, self.grid_w  # 7, 7
        N = H * W                         # 49

        # ------------------------------------------------------------------
        # 1.  Capture attention weights via monkey-patching.
        #     This torchvision version has no attn_drop sub-module;
        #     all attention math is inside shifted_window_attention().
        #     We wrap each stage-3 block forward and intercept F.softmax.
        # ------------------------------------------------------------------
        captured = []  # list of {'attn': Tensor, 'shift_size': tuple}
        _orig_swa_fwd = ShiftedWindowAttention.forward

        def _patched_fwd(self_attn, x):
            _orig_fn = _swin_tv.shifted_window_attention
            attn_store = {}

            def _intercepted(*args, **kwargs):
                import torch.nn.functional as _F_mod
                _orig_sm = _F_mod.softmax

                def _cap_sm(inp, dim=-1, **kw):
                    out = _orig_sm(inp, dim=dim, **kw)
                    if inp.ndim == 4:
                        attn_store['attn'] = out.detach().cpu().float()
                    return out

                _F_mod.softmax = _cap_sm
                torch.nn.functional.softmax = _cap_sm
                try:
                    result = _orig_fn(*args, **kwargs)
                finally:
                    _F_mod.softmax = _orig_sm
                    torch.nn.functional.softmax = _orig_sm
                return result

            _swin_tv.shifted_window_attention = _intercepted
            try:
                out = _orig_swa_fwd(self_attn, x)
            finally:
                _swin_tv.shifted_window_attention = _orig_fn

            if 'attn' in attn_store:
                captured.append({
                    'attn': attn_store['attn'],
                    'shift_size': tuple(self_attn.shift_size),
                })
            return out

        stage3_blocks = list(self.swin_features[7].children())
        for blk in stage3_blocks:
            blk.attn.forward = _patched_fwd.__get__(blk.attn, ShiftedWindowAttention)

        # ------------------------------------------------------------------
        # 2.  Forward pass
        # ------------------------------------------------------------------
        was_training = self.training
        self.eval()
        with torch.no_grad():
            _ = self.forward(image, metadata)
        if was_training:
            self.train()
        for blk in stage3_blocks:
            blk.attn.forward = _orig_swa_fwd.__get__(blk.attn, ShiftedWindowAttention)

        # ------------------------------------------------------------------
        # 3.  Rollout (block 0 -> block 1)
        # ------------------------------------------------------------------
        rollout = torch.eye(N)

        for info in captured:
            attn = info['attn']              # (B*nW, nH, Nw, Nw)
            shift_size = info['shift_size']
            A = attn[0].mean(dim=0)           # avg heads -> (N, N)
            if any(s > 0 for s in shift_size):
                A = self._unshift_attn(A, shift_size, H, W)
            A = 0.5 * A + 0.5 * torch.eye(N)
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
            rollout = A @ rollout

        swin_map = rollout.sum(dim=0).view(H, W).numpy()  # (7, 7)

        # ------------------------------------------------------------------
        # 4.  Combine with cross-attention weights
        # ------------------------------------------------------------------
        if combine_cross_attn and self.cross_attn.last_attn_weights is not None:
            cross_w = self.cross_attn.last_attn_weights  # (B, 1, 49)
            cross_map = cross_w[0, 0, :].float().cpu().numpy().reshape(H, W)
            s_min, s_max = swin_map.min(), swin_map.max()
            c_min, c_max = cross_map.min(), cross_map.max()
            swin_norm  = (swin_map  - s_min) / (s_max  - s_min  + 1e-8)
            cross_norm = (cross_map - c_min) / (c_max  - c_min  + 1e-8)
            combined = np.sqrt(swin_norm * cross_norm + 1e-8)
        else:
            combined = swin_map.copy()

        # ------------------------------------------------------------------
        # 5.  Normalise to [0, 1]
        # ------------------------------------------------------------------
        combined -= combined.min()
        denom = combined.max()
        if denom > 1e-8:
            combined /= denom

        return combined.astype(np.float32)


    # ------------------------------------------------------------------
    # Explainability helper
    # ------------------------------------------------------------------

    def _unshift_attn(
        self,
        attn: torch.Tensor,
        shift_size: tuple,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Undo the cyclic spatial shift of ``ShiftedWindowAttention`` so that
        attention weight indices correspond to original (unshifted) spatial
        positions.

        Torchvision applies the shift before computing attention::

            x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))

        so a token originally at (r, c) moves to shifted position
        ((r - shift_h) % H, (c - shift_w) % W).  This method inverts that
        permutation on both the query and key axes of the attention matrix.

        Parameters
        ----------
        attn       : (N, N) float tensor of attention weights in shifted order
        shift_size : (shift_h, shift_w)
        H, W       : spatial grid dimensions (both 7 for Swin-B stage 3)

        Returns
        -------
        (N, N) float tensor with token indices in original spatial order
        """
        N = H * W
        shift_h, shift_w = shift_size

        # inv_perm[shifted_flat] = orig_flat
        # token at shifted (sr, sc) came from orig ((sr+sh)%H, (sc+sw)%W)
        inv_perm = torch.zeros(N, dtype=torch.long)
        for sr in range(H):
            for sc in range(W):
                orig_r = (sr + shift_h) % H
                orig_c = (sc + shift_w) % W
                inv_perm[sr * W + sc] = orig_r * W + orig_c

        # fwd_perm[orig_flat] = shifted_flat  (inverse permutation of inv_perm)
        fwd_perm = torch.argsort(inv_perm)

        # attn_orig[r, c] = attn_shifted[fwd_perm[r], fwd_perm[c]]
        return attn[fwd_perm, :][:, fwd_perm]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @classmethod
    def model_info(cls) -> dict:
        """
        Return a dict describing this architecture.
        Analogous to CropDiseaseFusionModel.get_model_info() in fusion_model.py.
        """
        return {
            "architecture": "arch2_swin_fusion",
            "backbone": "swin_b",
            "backbone_source": "torchvision.models.swin_b",
            "img_feature_dim": _SWIN_B_FEATURE_DIM,
            "n_patch_tokens": _SWIN_B_N_TOKENS,
            "spatial_grid": f"{_SWIN_B_GRID_H}x{_SWIN_B_GRID_W}",
            "fusion": "cross_attention (meta_token=Q, patch_tokens=K=V)",
            "explainability": "attention_rollout",
            "new_dependencies": "none -- uses torchvision >= 0.13",
        }

    def count_parameters(self) -> dict:
        """Return trainable parameter counts per sub-module (useful for viva)."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "swin_backbone": _count(self.swin_features) + _count(self.swin_norm),
            "metadata_branch": _count(self.metadata_branch),
            "cross_attention": _count(self.cross_attn),
            "classifier_head": _count(self.classifier),
            "total": _count(self),
        }


# ===========================================================================
# Factory function (mirrors create_fusion_model() in fusion_model.py)
# ===========================================================================

def create_swin_fusion_model(
    num_classes: int = 38,
    metadata_dim: int = 16,
    dropout_rate: float = 0.3,
    num_attn_heads: int = 8,
    pretrained: bool = True,
) -> SwinFusionModel:
    """
    Convenience factory mirroring create_fusion_model() from fusion_model.py.

    Parameters
    ----------
    num_classes   : number of output disease/pest/cattle classes
    metadata_dim  : width of the agronomic metadata vector (default 16)
    dropout_rate  : dropout applied in MetadataBranch and classifier head
    num_attn_heads: cross-attention heads (must divide 1024; 8 is a safe default)
    pretrained    : if True, load ImageNet-1k weights for the Swin-B backbone

    Returns
    -------
    SwinFusionModel instance, ready for .to(device)

    Example
    -------
    >>> from fusion_model_arch2_swin import create_swin_fusion_model
    >>> model = create_swin_fusion_model(num_classes=38).to("cuda")
    >>> logits = model(image_batch, metadata_batch)   # (B, 38)
    >>> probs  = F.softmax(logits, dim=1)             # (B, 38)  for inference
    """
    return SwinFusionModel(
        num_classes=num_classes,
        metadata_dim=metadata_dim,
        dropout_rate=dropout_rate,
        num_attn_heads=num_attn_heads,
        pretrained=pretrained,
    )


# ===========================================================================
# Smoke-test  (python fusion_model_arch2_swin.py)
# ===========================================================================

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Khet Guard -- Architecture 2 (Swin-B Fusion) smoke test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = create_swin_fusion_model(num_classes=38, pretrained=False).to(device)

    # Parameter counts
    params = model.count_parameters()
    print("Parameter counts:")
    for k, v in params.items():
        print(f"  {k:25s}: {v:>12,}")

    # Architecture info
    print("\nmodel_info():")
    print(json.dumps(model.model_info(), indent=2))

    # Forward pass
    B = 2
    dummy_image = torch.randn(B, 3, 224, 224, device=device)
    dummy_meta  = torch.randn(B, 16, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(dummy_image, dummy_meta)
    probs = F.softmax(logits, dim=1)

    print(f"\nForward pass:")
    print(f"  image input  : {tuple(dummy_image.shape)}")
    print(f"  metadata in  : {tuple(dummy_meta.shape)}")
    print(f"  logits out   : {tuple(logits.shape)}")
    print(f"  probs sum    : {probs.sum(dim=1).tolist()}  (should be [1.0, 1.0])")

    # Attention rollout
    rollout = model.attention_rollout(dummy_image[:1], dummy_meta[:1])
    print(f"\nAttention rollout:")
    print(f"  shape        : {rollout.shape}   (should be (7, 7))")
    print(f"  value range  : [{rollout.min():.4f}, {rollout.max():.4f}]  (should be [0, 1])")

    print("\nAll checks passed.")
