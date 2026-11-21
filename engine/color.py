"""Linear-light RGB conversion with highlight headroom boost.

Optical operations (defocus, motion trails, lens veiling glare) must run in linear
light with expanded highlight headroom so bright skin speculars, flash reflections,
and night lights glow organically instead of smearing into flat muddy gray.
"""

from __future__ import annotations

import numpy as np

from .schema import OpticalColorConfig

# Precomputed 8-bit sRGB -> Linear LUT for gamma 2.2 (sub-millisecond conversion)
_U8_GRID = np.linspace(0.0, 1.0, 256, dtype=np.float32)
_LUT_GAMMA_22 = np.power(_U8_GRID, 2.2, dtype=np.float32)


def srgb_u8_to_linear(img_rgb: np.ndarray, cfg: OpticalColorConfig | None = None) -> np.ndarray:
    """Convert uint8 RGB [0..255] to float32 Linear RGB with subtle highlight headroom."""
    cfg = (cfg or OpticalColorConfig()).validated()

    if img_rgb.dtype == np.uint8 and abs(cfg.gamma - 2.2) < 1e-4:
        linear = _LUT_GAMMA_22[img_rgb]
    else:
        srgb = np.clip(img_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
        linear = np.power(srgb, cfg.gamma, dtype=np.float32)

    if cfg.highlight_boost > 1e-4:
        # Threshold in sRGB space (>0.85 default) mapped to linear space
        lin_thresh = float(cfg.highlight_threshold ** cfg.gamma)
        # Smooth luminance-weighted highlight expansion so bright speculars keep energy
        lum = (
            0.2126 * linear[..., 0]
            + 0.7152 * linear[..., 1]
            + 0.0722 * linear[..., 2]
        )
        # Also catch single-channel bright highlights via max(lum, max_c * 0.85)
        peak = np.maximum(lum, np.max(linear, axis=2) * 0.90)
        excess = np.clip((peak - lin_thresh) / max(1e-4, 1.0 - lin_thresh), 0.0, 1.0)
        # Smoothstep curve for artifact-free highlight headroom
        smooth = excess * excess * (3.0 - 2.0 * excess)
        gain = 1.0 + cfg.highlight_boost * smooth
        linear = linear * gain[..., np.newaxis]

    return linear.astype(np.float32, copy=False)


def linear_to_srgb_u8(linear: np.ndarray, cfg: OpticalColorConfig | None = None) -> np.ndarray:
    """Soft-shoulder tone rolloff + Linear RGB -> uint8 sRGB [0..255]."""
    cfg = (cfg or OpticalColorConfig()).validated()
    x = np.maximum(linear, 0.0).astype(np.float32, copy=False)

    # Smooth highlight shoulder above knee=0.88 so boosted highlights compress back
    # gracefully into [0..1] like an iPhone HDR tone curve rather than hard-clipping.
    knee = 0.88
    above = x > knee
    if np.any(above):
        over = x[above] - knee
        # Rational rolloff mapping [knee, inf) -> [knee, 1.0)
        span = 1.0 - knee
        x = x.copy()
        x[above] = knee + span * (over / (over + span))

    srgb = np.power(np.clip(x, 0.0, 1.0), 1.0 / cfg.gamma, dtype=np.float32)
    return np.clip(np.rint(srgb * 255.0), 0.0, 255.0).astype(np.uint8)
