"""Stage 4: Nighttime Phone Grain & ISP Look (Applied AFTER Blur!).

Simulates a high-ISO smartphone sensor + Apple/Android ISP pipeline:
  1. Shadow-weighted shot + read noise (gritty dark walls, cleaner bright skin).
  2. Coarse/clumpy spatial correlation (so grain feels like real phone Night Mode,
     NOT crisp digital TV static).
  3. Subtle chroma noise softening in YCrCb space (real smartphone ISPs aggressively
     denoise color speckles while leaving organic luminance clumps).
  4. Light edge-preserving smoothing so flat patches have that authentic low-light
     computational smartphone texture.
"""

from __future__ import annotations

import cv2
import numpy as np

from .schema import Stage4GrainConfig


def apply_stage4_grain(srgb_u8: np.ndarray, cfg: Stage4GrainConfig) -> np.ndarray:
    """Apply realistic high-ISO nighttime phone grain & ISP texture on uint8 sRGB."""
    cfg = cfg.validated()
    if not cfg.enabled or (cfg.grain_amount <= 0.01 and cfg.isp_smoothing <= 0.01):
        return srgb_u8.copy()

    h, w = srgb_u8.shape[:2]
    img = srgb_u8.astype(np.float32) / 255.0

    # 1. Generate shadow-weighted clumpy sensor noise (luma + subdued chroma)
    if cfg.grain_amount > 0.01:
        rng = np.random.default_rng(int(cfg.seed) & 0xFFFFFFFF)

        # Clumpiness controls the grain spatial scale (smaller base grid -> larger organic clumps)
        clump_scale = 1.0 - 0.52 * cfg.clumpiness  # 1.0 (fine) -> 0.48 (clumpy Night Mode)
        nh = max(16, int(round(h * clump_scale)))
        nw = max(16, int(round(w * clump_scale)))

        # Luminance grain (primary visible smartphone texture)
        luma_coarse = rng.normal(0.0, 1.0, size=(nh, nw)).astype(np.float32)
        if nh != h or nw != w:
            luma_coarse = cv2.resize(luma_coarse, (w, h), interpolation=cv2.INTER_LINEAR)
        # Mix a touch of fine pixel-level read noise so it never looks blurry-only
        luma_fine = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        luma_noise = (0.72 * luma_coarse + 0.28 * luma_fine)

        # Chroma noise generated even coarser (matches smartphone chroma subsampling/NR)
        ch = max(12, int(round(h * clump_scale * 0.45)))
        cw = max(12, int(round(w * clump_scale * 0.45)))
        chroma_raw = rng.normal(0.0, 1.0, size=(ch, cw, 3)).astype(np.float32)
        chroma_raw -= chroma_raw.mean(axis=2, keepdims=True)  # Pure color variation
        chroma_noise = cv2.resize(chroma_raw, (w, h), interpolation=cv2.INTER_LINEAR)

        # Shadow-weighted shot + read noise envelope:
        # Dark shadows get strong low-light grain; bright skin highlights get much lighter grain.
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        shadow_env = np.power(np.clip(1.0 - lum, 0.0, 1.0), 1.45)
        # Mix uniform shot noise with shadow-weighted read noise
        weight_map = (1.0 - cfg.shadow_weight * 0.65) + (cfg.shadow_weight * 1.15) * shadow_env

        # Calibrated noise amplitude (0..1 slider -> subtle 0.015 to gritty 0.078 std)
        luma_amp = 0.074 * cfg.grain_amount
        chroma_amp = 0.026 * cfg.grain_amount * (1.0 - 0.35 * cfg.isp_smoothing)

        noisy = img + (luma_noise * weight_map * luma_amp)[..., np.newaxis] + chroma_noise * (weight_map * chroma_amp)[..., np.newaxis]
        img = np.clip(noisy, 0.0, 1.0)

    out_u8 = np.clip(np.rint(img * 255.0), 0.0, 255.0).astype(np.uint8)

    # 2. Smartphone ISP chroma softening + light edge-preserving smoothing
    if cfg.isp_smoothing > 0.02:
        # Chroma softening in YCrCb (keeps luminance clumps intact while melting color speckle)
        ycrcb = cv2.cvtColor(out_u8, cv2.COLOR_RGB2YCrCb)
        chroma_sigma = 0.6 + 1.8 * cfg.isp_smoothing
        ycrcb[..., 1] = cv2.GaussianBlur(ycrcb[..., 1], (0, 0), sigmaX=chroma_sigma)
        ycrcb[..., 2] = cv2.GaussianBlur(ycrcb[..., 2], (0, 0), sigmaX=chroma_sigma)
        chroma_soft = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        # Light edge-preserving bilateral pass (fast d=5 on CPU) blended gently
        sigma_color = 14.0 + 26.0 * cfg.isp_smoothing
        sigma_space = 2.5 + 3.5 * cfg.isp_smoothing
        smoothed = cv2.bilateralFilter(chroma_soft, d=5, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        blend = 0.25 + 0.45 * cfg.isp_smoothing
        out_u8 = cv2.addWeighted(chroma_soft, 1.0 - blend, smoothed, blend, 0.0)

    return out_u8
