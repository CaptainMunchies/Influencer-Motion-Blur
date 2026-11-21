"""Stage 4: Nighttime Phone Grain, Digital-Zoom Sensor Crop & ISP Look (Applied AFTER Blur!).

Simulates both clean everyday low-light iPhone Night Mode AND authentic raw 3AM digital-zoom
streetlight photos:
  1. Optional Digital-Zoom Sensor Crop (`digital_zoom > 0`): downsamples to the true cropped
     sensor grid (~40%-70% resolution), injects high-ISO noise & waxy ISP denoising AT the
     cropped sensor level, then upscales back with subtle smartphone micro-sharpening —
     creating the exact unmistakable pinch-zoomed phone camera look.
  2. Shadow-weighted shot + read noise (gritty dim shadows, cleaner bright skin) with
     coarse spatial correlation & heavy ISP chroma suppression (plus authentic low-light
     blue-channel chroma mottling when `night_cast` is active).
  3. Optional 3AM Sodium-Streetlight Cast (`night_cast > 0`): muted dim night highlight ceiling
     with subtle olive-amber streetlight tint in murky shadows.
"""

from __future__ import annotations

import cv2
import numpy as np

from .schema import Stage4GrainConfig


def apply_stage4_grain(srgb_u8: np.ndarray, cfg: Stage4GrainConfig) -> np.ndarray:
    """Apply realistic high-ISO nighttime phone grain, optional digital zoom & ISP texture."""
    cfg = cfg.validated()
    if (
        not cfg.enabled
        and True
    ) or (
        cfg.grain_amount <= 0.01
        and cfg.isp_smoothing <= 0.01
        and cfg.digital_zoom <= 0.01
        and cfg.night_cast <= 0.01
    ):
        return srgb_u8.copy()

    orig_h, orig_w = srgb_u8.shape[:2]
    working_u8 = srgb_u8.copy()

    # 0. Optional Digital-Zoom Sensor Crop: drop to zoomed sensor pixel grid BEFORE noise & ISP denoise!
    zoomed = cfg.digital_zoom > 0.02
    if zoomed:
        # 0.0 -> 1.0x native, 1.0 -> 0.36x effective optical resolution (~3x-4x smartphone digital zoom)
        zoom_scale = 1.0 - 0.62 * cfg.digital_zoom
        zw = max(180, int(round(orig_w * zoom_scale)))
        zh = max(180, int(round(orig_h * zoom_scale)))
        working_u8 = cv2.resize(working_u8, (zw, zh), interpolation=cv2.INTER_AREA)

    h, w = working_u8.shape[:2]
    img = working_u8.astype(np.float32) / 255.0

    # 1. Optional 3AM Sodium-Streetlight Exposure & Murky Olive-Amber Night Cast
    if cfg.night_cast > 0.01:
        nc = cfg.night_cast
        # Compress bright highlights gently (muted dim streetlamp exposure instead of studio bright)
        # while subtly lifting the murky deep shadows with an olive-warm sodium cast.
        img = np.power(np.clip(img, 0.0, 1.0), 1.0 + 0.16 * nc)
        highlight_cap = 1.0 - 0.24 * nc
        img = img * highlight_cap
        # Soft olive-amber sodium tint in shadows & low-mids (G & R slightly warmer than B)
        lum0 = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        shadow_zone = np.clip(1.0 - lum0 * 1.4, 0.0, 1.0)[..., np.newaxis]
        # Per-channel multiplicative tint + subtle murky shadow floor
        tint = np.array([1.0 + 0.06 * nc, 1.0 + 0.05 * nc, 1.0 - 0.16 * nc], dtype=np.float32)
        floor = np.array([0.042 * nc, 0.040 * nc, 0.022 * nc], dtype=np.float32)
        img = img * tint[np.newaxis, np.newaxis, :] + shadow_zone * floor[np.newaxis, np.newaxis, :]
        # Gently desaturate slightly for raw unedited night-camera realism
        gray_m = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])[..., np.newaxis]
        img = img * (1.0 - 0.18 * nc) + gray_m * (0.18 * nc)
        img = np.clip(img, 0.0, 1.0)

    # 2. Shadow-weighted clumpy high-ISO sensor noise
    if cfg.grain_amount > 0.01:
        rng = np.random.default_rng(int(cfg.seed) & 0xFFFFFFFF)

        clump_scale = 1.0 - 0.52 * cfg.clumpiness
        nh = max(16, int(round(h * clump_scale)))
        nw = max(16, int(round(w * clump_scale)))

        luma_coarse = rng.normal(0.0, 1.0, size=(nh, nw)).astype(np.float32)
        if nh != h or nw != w:
            luma_coarse = cv2.resize(luma_coarse, (w, h), interpolation=cv2.INTER_LINEAR)
        luma_fine = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        luma_noise = 0.76 * luma_coarse + 0.24 * luma_fine

        # Coarser chroma mottle (slightly stronger blue/yellow mottle in dim sodium light!)
        ch = max(12, int(round(h * clump_scale * 0.40)))
        cw = max(12, int(round(w * clump_scale * 0.40)))
        chroma_raw = rng.normal(0.0, 1.0, size=(ch, cw, 3)).astype(np.float32)
        chroma_raw -= chroma_raw.mean(axis=2, keepdims=True)
        if cfg.night_cast > 0.05:
            # Real phones amplify B-channel hardest under yellow streetlights -> stronger blue/olive mottle
            chroma_raw[..., 2] *= (1.0 + 0.55 * cfg.night_cast)
        chroma_noise = cv2.resize(chroma_raw, (w, h), interpolation=cv2.INTER_LINEAR)

        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        shadow_env = np.power(np.clip(1.0 - lum, 0.0, 1.0), 1.55)
        hl_damp = np.clip((1.05 - lum) / 0.25, 0.35, 1.0)
        weight_map = ((1.0 - cfg.shadow_weight * 0.70) + (cfg.shadow_weight * 1.25) * shadow_env) * hl_damp

        # Boost amplitude gently when digital_zoom/night_cast active, then let ISP denoise melt it into real clumps
        iso_boost = 1.0 + 0.25 * cfg.digital_zoom + 0.22 * cfg.night_cast
        luma_amp = 0.036 * cfg.grain_amount * iso_boost
        chroma_amp = 0.011 * cfg.grain_amount * (1.0 - 0.35 * cfg.isp_smoothing + 0.35 * cfg.night_cast)

        noisy = (
            img
            + (luma_noise * weight_map * luma_amp)[..., np.newaxis]
            + chroma_noise * (weight_map * chroma_amp)[..., np.newaxis]
        )
        img = np.clip(noisy, 0.0, 1.0)

    out_u8 = np.clip(np.rint(img * 255.0), 0.0, 255.0).astype(np.uint8)

    # 3. Smartphone ISP chroma softening + edge-preserving waxy low-light smoothing
    if cfg.isp_smoothing > 0.02:
        ycrcb = cv2.cvtColor(out_u8, cv2.COLOR_RGB2YCrCb)
        chroma_sigma = 0.8 + 2.0 * cfg.isp_smoothing
        ycrcb[..., 1] = cv2.GaussianBlur(ycrcb[..., 1], (0, 0), sigmaX=chroma_sigma)
        ycrcb[..., 2] = cv2.GaussianBlur(ycrcb[..., 2], (0, 0), sigmaX=chroma_sigma)
        chroma_soft = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        # Stronger bilateral pass when digital_zoom is active -> authentic iPhone Deep Fusion watercolor smoothing!
        d_val = 9 if cfg.digital_zoom > 0.35 else 5
        sigma_color = 14.0 + 38.0 * cfg.isp_smoothing
        sigma_space = 2.5 + 5.0 * cfg.isp_smoothing
        smoothed = cv2.bilateralFilter(chroma_soft, d=d_val, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        if cfg.digital_zoom > 0.45 and cfg.isp_smoothing > 0.55:
            # Second gentle pass melts sandy sensor static into true creamy zoomed-phone patches
            smoothed = cv2.bilateralFilter(smoothed, d=7, sigmaColor=sigma_color * 0.85, sigmaSpace=sigma_space * 0.85)

        blend = 0.20 + 0.68 * cfg.isp_smoothing
        out_u8 = cv2.addWeighted(chroma_soft, 1.0 - blend, smoothed, blend, 0.0)

    # 4. Upscale digital-zoomed sensor grid back to full resolution + subtle phone ISP sharpening
    if zoomed:
        out_u8 = cv2.resize(out_u8, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        sharp_amt = 0.24 * cfg.digital_zoom * (1.0 - 0.50 * cfg.night_cast)
        if sharp_amt > 0.03:
            blur_u = cv2.GaussianBlur(out_u8, (0, 0), sigmaX=1.5)
            out_u8 = cv2.addWeighted(out_u8, 1.0 + sharp_amt, blur_u, -sharp_amt, 0.0)
        # Ultra-fine subtle post-upscale filmic read texture (never sandy static!)
        if cfg.grain_amount > 0.20:
            rng2 = np.random.default_rng((int(cfg.seed) + 777) & 0xFFFFFFFF)
            post_n = rng2.normal(0.0, 1.35 * cfg.grain_amount, size=(orig_h, orig_w, 1)).astype(np.float32)
            out_u8 = np.clip(out_u8.astype(np.float32) + post_n, 0.0, 255.0).astype(np.uint8)

    return out_u8
