"""Stage 3: Dirty Smartphone Lens Glow, Atmospheric Haze & Edge Fringing.

Runs in Linear RGB space so bright flash speculars and warm night lights bloom with
natural halation, shadows get a subtle milky smudged-lens lift, and outer edges pick up
barely-there radial chromatic aberration like a real tiny smartphone lens.
"""

from __future__ import annotations

import cv2
import numpy as np

from .schema import Stage3OpticsConfig


def _fast_wide_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Fast multi-scale wide Gaussian blur via downsample -> blur -> upsample (~3ms on CPU)."""
    h, w = img.shape[:2]
    if sigma_px <= 1.0:
        return img.copy()

    # Downsample proportionally to sigma so wide veiling glare is instant on CPU
    factor = max(1, min(8, int(sigma_px / 6.0)))
    if factor > 1:
        sw = max(8, w // factor)
        sh = max(8, h // factor)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma_px / factor, borderType=cv2.BORDER_REFLECT_101)
        return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_px, borderType=cv2.BORDER_REFLECT_101)


def _apply_radial_chromatic_aberration(linear_rgb: np.ndarray, strength: float) -> np.ndarray:
    """Very subtle radial R/B edge fringing (zero at center/face, gentle near corners)."""
    if strength <= 1e-3:
        return linear_rgb

    h, w = linear_rgb.shape[:2]
    # Max pixel shift at corners: ~0.8 to 2.6 pixels on 1080p (authentic phone lens, never cartoon 3D)
    max_shift_ratio = 0.0024 * strength

    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy

    # Quadratic radial falloff keeps the face/center untouched while fringing outer edges
    rn2 = (dx / max(1.0, cx)) ** 2 + (dy / max(1.0, cy)) ** 2
    radial_factor = np.clip(rn2 * 0.65, 0.0, 1.4)

    # Red expands slightly outward, Blue pulls slightly inward
    map_r_x = (xx - dx * max_shift_ratio * radial_factor).astype(np.float32)
    map_r_y = (yy - dy * max_shift_ratio * radial_factor).astype(np.float32)
    map_b_x = (xx + dx * max_shift_ratio * 0.85 * radial_factor).astype(np.float32)
    map_b_y = (yy + dy * max_shift_ratio * 0.85 * radial_factor).astype(np.float32)

    out = linear_rgb.copy()
    out[..., 0] = cv2.remap(
        linear_rgb[..., 0], map_r_x, map_r_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    out[..., 2] = cv2.remap(
        linear_rgb[..., 2], map_b_x, map_b_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    return out


def apply_stage3_optics(linear_rgb: np.ndarray, cfg: Stage3OpticsConfig) -> np.ndarray:
    """Apply dirty smartphone lens veiling glare, milky shadow lift & edge fringing in Linear RGB."""
    cfg = cfg.validated()
    if not cfg.enabled:
        return linear_rgb.copy()

    out = linear_rgb.copy()
    h, w = out.shape[:2]
    short_edge = float(min(h, w))

    # 1. Subtle radial edge chromatic aberration (happens at the physical lens before scatter)
    if cfg.edge_fringing > 0.01:
        out = _apply_radial_chromatic_aberration(out, cfg.edge_fringing)

    # 2. Linear-light dirty-lens veiling glare (tight highlight bloom + wide hazy scatter)
    if cfg.glow_amount > 0.01:
        lum = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        # Softly isolate brighter regions (>0.28 linear) for glowing halation
        hl_mask = np.clip((lum - 0.24) / 0.55, 0.0, 1.0)
        hl_mask = hl_mask * hl_mask * (3.0 - 2.0 * hl_mask)
        highlights = out * hl_mask[..., np.newaxis]

        # Tight specular bloom + wide smudged-fingerprint lens scatter
        tight_bloom = _fast_wide_blur(highlights, sigma_px=short_edge * 0.016)
        wide_veil = _fast_wide_blur(out, sigma_px=short_edge * 0.075)

        # Warm tint for night/flash lighting (subtle golden-amber glow on highlights)
        wt = cfg.warmth
        bloom_tint = np.array([1.0 + 0.26 * wt, 1.0 + 0.06 * wt, 1.0 - 0.20 * wt], dtype=np.float32)

        g = cfg.glow_amount
        # Blend wide veiling scatter + additive highlight halation
        veil_opacity = 0.16 * g
        bloom_strength = 0.42 * g
        out = (
            out * (1.0 - veil_opacity)
            + wide_veil * veil_opacity * bloom_tint[np.newaxis, np.newaxis, :]
            + tight_bloom * bloom_strength * bloom_tint[np.newaxis, np.newaxis, :]
        )

    # 3. Gentle lifted milky blacks (smudged smartphone glass / flash bounce in dark rooms)
    if cfg.haze_lift > 0.01:
        # Lift shadows subtly in linear space (equivalent to ~4-16 sRGB black floor lift)
        # Slightly cool-warm neutral milky lift weighted toward shadows so highlights stay punchy
        lift_floor = 0.022 * (cfg.haze_lift ** 1.35)
        wt = cfg.warmth
        lift_rgb = np.array(
            [lift_floor * (1.0 + 0.14 * wt), lift_floor * (1.0 + 0.03 * wt), lift_floor * (1.0 - 0.08 * wt)],
            dtype=np.float32,
        )
        # Compress slightly toward lifted black floor so overall exposure doesn't blow out
        out = out * (1.0 - lift_floor * 0.85) + lift_rgb[np.newaxis, np.newaxis, :]

    return np.maximum(out, 0.0).astype(np.float32)
