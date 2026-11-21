"""Stage 4: Nighttime Phone Grain & ISP Look (Applied AFTER Blur!).

Calibrated directly against measured noise profiles of real Instagram iPhone photos:
- Shadow-weighted shot + read noise so bright skin stays clean & flattering while
  darker backgrounds/shadows pick up organic low-light smartphone texture.
- Subtle clumpy spatial correlation + heavy chroma noise suppression (matching real
  Apple/Android ISP chroma denoising — never crunchy digital TV static).
- Gentle edge-preserving smoothing for that subtle computational smartphone finish.
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

    # 1. Generate shadow-weighted clumpy sensor noise (luma + restrained chroma)
    if cfg.grain_amount > 0.01:
        rng = np.random.default_rng(int(cfg.seed) & 0xFFFFFFFF)

        # Clumpiness controls spatial correlation scale (1.0 = fine, 0.45 = soft Night Mode clumps)
        clump_scale = 1.0 - 0.50 * cfg.clumpiness
        nh = max(16, int(round(h * clump_scale)))
        nw = max(16, int(round(w * clump_scale)))

        luma_coarse = rng.normal(0.0, 1.0, size=(nh, nw)).astype(np.float32)
        if nh != h or nw != w:
            luma_coarse = cv2.resize(luma_coarse, (w, h), interpolation=cv2.INTER_LINEAR)
        luma_fine = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        luma_noise = 0.78 * luma_coarse + 0.22 * luma_fine

        # Coarser subtle chroma variation (real iPhone ISPs suppress color noise heavily)
        ch = max(12, int(round(h * clump_scale * 0.42)))
        cw = max(12, int(round(w * clump_scale * 0.42)))
        chroma_raw = rng.normal(0.0, 1.0, size=(ch, cw, 3)).astype(np.float32)
        chroma_raw -= chroma_raw.mean(axis=2, keepdims=True)
        chroma_noise = cv2.resize(chroma_raw, (w, h), interpolation=cv2.INTER_LINEAR)

        # Shadow-weighted envelope: keeps midtone/bright faces silky while shadows show real ISO texture
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        shadow_env = np.power(np.clip(1.0 - lum, 0.0, 1.0), 1.65)
        # Protect very brightest highlights (>0.85) from looking dirty
        hl_damp = np.clip((1.05 - lum) / 0.25, 0.35, 1.0)
        weight_map = ((1.0 - cfg.shadow_weight * 0.72) + (cfg.shadow_weight * 1.20) * shadow_env) * hl_damp

        # Calibrated to real iPhone Night Mode / party flash std (fine & authentic, never crunchy static)
        luma_amp = 0.038 * cfg.grain_amount
        chroma_amp = 0.012 * cfg.grain_amount * (1.0 - 0.40 * cfg.isp_smoothing)

        noisy = (
            img
            + (luma_noise * weight_map * luma_amp)[..., np.newaxis]
            + chroma_noise * (weight_map * chroma_amp)[..., np.newaxis]
        )
        img = np.clip(noisy, 0.0, 1.0)

    out_u8 = np.clip(np.rint(img * 255.0), 0.0, 255.0).astype(np.uint8)

    # 2. Smartphone ISP chroma softening + light edge-preserving smoothing
    if cfg.isp_smoothing > 0.02:
        ycrcb = cv2.cvtColor(out_u8, cv2.COLOR_RGB2YCrCb)
        chroma_sigma = 0.7 + 1.6 * cfg.isp_smoothing
        ycrcb[..., 1] = cv2.GaussianBlur(ycrcb[..., 1], (0, 0), sigmaX=chroma_sigma)
        ycrcb[..., 2] = cv2.GaussianBlur(ycrcb[..., 2], (0, 0), sigmaX=chroma_sigma)
        chroma_soft = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        sigma_color = 12.0 + 22.0 * cfg.isp_smoothing
        sigma_space = 2.0 + 3.0 * cfg.isp_smoothing
        smoothed = cv2.bilateralFilter(chroma_soft, d=5, sigmaColor=sigma_color, sigmaSpace=sigma_space)

        blend = 0.18 + 0.40 * cfg.isp_smoothing
        out_u8 = cv2.addWeighted(chroma_soft, 1.0 - blend, smoothed, blend, 0.0)

    return out_u8
