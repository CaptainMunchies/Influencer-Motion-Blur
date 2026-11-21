"""Stage 1: Focus & Lens Softness (True Circular Aperture Disc Defocus).

Replicates missed smartphone autofocus using a real circular aperture PSF (NOT plain
Gaussian blur), with resolution-normalized radius, subtle smartphone spherical rim
softness, and optional Depth/Focus Falloff + user brush mask blending.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.signal import fftconvolve

from .masks import UserBrushMask, build_focus_falloff_mask
from .schema import Stage1FocusConfig


def build_aperture_disc_kernel(radius: float, rim_softness: float = 0.18) -> np.ndarray:
    """Build an antialiased circular aperture PSF kernel (4x supersampled)."""
    r = max(0.35, float(radius))
    half = int(np.ceil(r + 1.5))
    size = 2 * half + 1

    # 4x supersampling eliminates blocky aliasing on small smartphone defocus radii
    ss = 4
    grid_size = size * ss
    coords = (np.arange(grid_size, dtype=np.float32) + 0.5) / ss - (half + 0.5)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    dist = np.sqrt(xx * xx + yy * yy)

    # Smooth circular disc edge with subtle smartphone optical rim softness
    feather = max(0.25, r * rim_softness)
    inner = max(0.0, r - feather)
    outer = r + 0.35 * feather
    disc = np.clip((outer - dist) / max(1e-4, outer - inner), 0.0, 1.0)
    disc = disc * disc * (3.0 - 2.0 * disc)

    # Slight spherical aberration energy boost near disc interior/rim (authentic lens bokeh)
    norm_r = np.clip(dist / max(r, 1e-4), 0.0, 1.0)
    bokeh_profile = 0.88 + 0.22 * (norm_r * norm_r)
    disc = disc * bokeh_profile

    kernel = disc.reshape(size, ss, size, ss).mean(axis=(1, 3)).astype(np.float32)
    total = float(kernel.sum())
    if total > 1e-8:
        kernel /= total
    else:
        kernel[half, half] = 1.0
    return kernel


def convolve_linear_rgb(linear_rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Fast 2D convolution on float32 Linear RGB (OpenCV spatial for small kernels, FFT for larger)."""
    kh, kw = kernel.shape
    if kh <= 1 and kw <= 1:
        return linear_rgb.copy()

    # OpenCV filter2D is SIMD-accelerated and fastest for compact kernels (<= 11x11)
    if kh * kw <= 121:
        return cv2.filter2D(linear_rgb, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)

    # For larger circular discs or long shake trajectories, FFT convolution with reflect padding
    # prevents dark border wrap-around and runs in ~20-45ms on an i5 CPU.
    pad_y = kh // 2
    pad_x = kw // 2
    padded = cv2.copyMakeBorder(
        linear_rgb, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REFLECT_101
    )
    out = np.empty_like(linear_rgb, dtype=np.float32)
    k_rot = kernel  # fftconvolve performs true convolution
    for c in range(3):
        out[..., c] = fftconvolve(padded[..., c], k_rot, mode="valid").astype(np.float32)
    return np.maximum(out, 0.0)


def apply_stage1_focus(
    linear_rgb: np.ndarray,
    cfg: Stage1FocusConfig,
    brush: Optional[UserBrushMask] = None,
) -> np.ndarray:
    """Apply Stage 1 circular aperture defocus in Linear RGB space."""
    cfg = cfg.validated()
    if not cfg.enabled or cfg.radius <= 0.15:
        return linear_rgb.copy()

    h, w = linear_rgb.shape[:2]
    # Normalize radius against a 1080px short edge so presets look consistent across image sizes
    scale = min(h, w) / 1080.0
    eff_radius = max(0.35, cfg.radius * scale)

    blur_map = build_focus_falloff_mask(
        hw=(h, w),
        mode=cfg.falloff_mode,
        strength=cfg.falloff_strength,
        cx=cfg.focus_center_x,
        cy=cfg.focus_center_y,
        brush=brush,
    )

    min_w = float(np.min(blur_map))
    max_w = float(np.max(blur_map))

    # Fast path: uniform blur across the entire frame
    if max_w - min_w < 0.03:
        if max_w < 0.02:
            return linear_rgb.copy()
        kernel = build_aperture_disc_kernel(eff_radius * max_w, cfg.aperture_rim_softness)
        return convolve_linear_rgb(linear_rgb, kernel)

    # Multi-depth layered circular defocus (Sharp -> Mid Disc -> Full Disc)
    # Using a mid-radius layer eliminates the harsh double-outline halo of naive 2-layer blending.
    mid_radius = max(0.35, eff_radius * 0.48)
    k_mid = build_aperture_disc_kernel(mid_radius, cfg.aperture_rim_softness)
    k_full = build_aperture_disc_kernel(eff_radius, cfg.aperture_rim_softness)

    img_mid = convolve_linear_rgb(linear_rgb, k_mid)
    img_full = convolve_linear_rgb(linear_rgb, k_full)

    # Piecewise smooth blend: [0 .. 0.5] -> sharp..mid, [0.5 .. 1.0] -> mid..full
    t_low = np.clip(blur_map * 2.0, 0.0, 1.0)[..., np.newaxis]
    t_high = np.clip((blur_map - 0.5) * 2.0, 0.0, 1.0)[..., np.newaxis]

    low_blend = linear_rgb * (1.0 - t_low) + img_mid * t_low
    high_blend = img_mid * (1.0 - t_high) + img_full * t_high

    mask_upper = (blur_map >= 0.5)[..., np.newaxis]
    return np.where(mask_upper, high_blend, low_blend).astype(np.float32)
