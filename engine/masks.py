"""Spatial falloff masks & user paint-brush mask helpers.

User brush masks follow a simple convention:
- `keep_sharp` (float32 HxW in [0..1]): 1.0 where user painted "Keep Sharp" (green).
- `extra_blur` (float32 HxW in [0..1]): 1.0 where user painted "Make Blurry" (red).

In PNG/RGBA mask uploads from the web UI:
- Green channel > Red channel encodes `keep_sharp` (scaled by alpha)
- Red channel > Green channel encodes `extra_blur` (scaled by alpha)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class UserBrushMask:
    """Decoded & resized spatial guidance masks painted by the user."""

    keep_sharp: np.ndarray  # HxW float32 [0..1]
    extra_blur: np.ndarray  # HxW float32 [0..1]

    @property
    def has_sharp(self) -> bool:
        return bool(np.max(self.keep_sharp) > 0.02)

    @property
    def has_blur(self) -> bool:
        return bool(np.max(self.extra_blur) > 0.02)


def decode_rgba_brush_mask(
    rgba_mask: Optional[np.ndarray],
    target_hw: Tuple[int, int],
    feather_ratio: float = 0.028,
) -> Optional[UserBrushMask]:
    """Decode an RGBA painted canvas (Green=keep sharp, Red=make blurry) to smooth float masks."""
    if rgba_mask is None:
        return None

    h, w = target_hw
    if rgba_mask.ndim != 3 or rgba_mask.shape[2] < 3:
        return None

    if rgba_mask.shape[0] != h or rgba_mask.shape[1] != w:
        rgba = cv2.resize(rgba_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        rgba = rgba_mask

    r = rgba[..., 0].astype(np.float32) / 255.0
    g = rgba[..., 1].astype(np.float32) / 255.0
    a = (rgba[..., 3].astype(np.float32) / 255.0) if rgba.shape[2] == 4 else np.ones((h, w), dtype=np.float32)

    # Separate green ("Keep Sharp") vs red ("Make Blurry") strokes cleanly
    keep_sharp = np.clip((g - r * 0.5) * a, 0.0, 1.0)
    extra_blur = np.clip((r - g * 0.5) * a, 0.0, 1.0)

    if np.max(keep_sharp) < 0.01 and np.max(extra_blur) < 0.01:
        return None

    sigma = max(4.0, min(h, w) * feather_ratio)
    ksize = int(sigma * 4) | 1
    ksize = max(5, min(ksize, 151))

    if np.max(keep_sharp) >= 0.01:
        keep_sharp = cv2.GaussianBlur(keep_sharp, (ksize, ksize), sigma)
        keep_sharp = np.clip(keep_sharp * 1.15, 0.0, 1.0)
    if np.max(extra_blur) >= 0.01:
        extra_blur = cv2.GaussianBlur(extra_blur, (ksize, ksize), sigma)
        extra_blur = np.clip(extra_blur * 1.15, 0.0, 1.0)

    return UserBrushMask(keep_sharp=keep_sharp, extra_blur=extra_blur)


def build_focus_falloff_mask(
    hw: Tuple[int, int],
    mode: str,
    strength: float,
    cx: float = 0.50,
    cy: float = 0.34,
    brush: Optional[UserBrushMask] = None,
) -> np.ndarray:
    """Return a float32 HxW blur-weight map in [0..1] (0 = sharp, 1 = full blur)."""
    h, w = hw
    if mode == "uniform" or strength <= 1e-4:
        blur_map = np.ones((h, w), dtype=np.float32)
    else:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xn = (xx / max(1, w - 1)) - cx
        yn = (yy / max(1, h - 1)) - cy

        if mode == "face_center":
            # Keep upper-middle portrait zone sharper; smoothly increase blur toward
            # foreground bottom (e.g. stairs/jeans/hands) and left/right outer edges.
            rx = xn / 0.34
            ry = yn / 0.28
            # Slightly stronger foreground blur below center
            below_boost = np.clip(yn * 1.45, 0.0, 1.0)
            dist = np.sqrt(rx * rx + ry * ry) + 0.45 * below_boost
            # Smooth transition from sharp face core to full blur
            falloff = np.clip((dist - 0.35) / 1.10, 0.0, 1.0)
            falloff = falloff * falloff * (3.0 - 2.0 * falloff)
            # Mix between uniform (1.0) and falloff map by strength
            min_blur = 1.0 - strength
            blur_map = min_blur + strength * falloff
        else:  # "edges"
            rx = (xx / max(1, w - 1) - 0.5) / 0.42
            ry = (yy / max(1, h - 1) - 0.5) / 0.42
            dist = np.sqrt(rx * rx + ry * ry)
            falloff = np.clip((dist - 0.30) / 0.95, 0.0, 1.0)
            falloff = falloff * falloff * (3.0 - 2.0 * falloff)
            min_blur = 1.0 - strength
            blur_map = min_blur + strength * falloff

    if brush is not None:
        if brush.has_blur:
            blur_map = np.maximum(blur_map, brush.extra_blur)
        if brush.has_sharp:
            blur_map = blur_map * (1.0 - brush.keep_sharp)

    return np.clip(blur_map, 0.0, 1.0).astype(np.float32)


def build_outer_limb_mask(
    hw: Tuple[int, int],
    brush: Optional[UserBrushMask] = None,
    face_cx: float = 0.50,
    face_cy: float = 0.32,
) -> np.ndarray:
    """Build a natural mask that keeps the face/upper-center frozen while letting outer arms/hands trail."""
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / max(1, w - 1)
    yn = yy / max(1, h - 1)

    # Elliptical protected face & upper-torso core
    dx = (xn - face_cx) / 0.23
    dy = (yn - (face_cy + 0.08)) / 0.30
    core_dist = np.sqrt(dx * dx + dy * dy)
    outer = np.clip((core_dist - 0.65) / 0.85, 0.0, 1.0)
    outer = outer * outer * (3.0 - 2.0 * outer)

    if brush is not None:
        if brush.has_blur:
            # When the user explicitly painted moving arms/hands, prioritize their brush!
            if not brush.has_sharp:
                outer = np.clip(outer * 0.25 + brush.extra_blur * 0.95, 0.0, 1.0)
            else:
                outer = np.clip(outer * 0.20 + brush.extra_blur, 0.0, 1.0)
        if brush.has_sharp:
            outer = outer * (1.0 - brush.keep_sharp)

    return np.clip(outer, 0.0, 1.0).astype(np.float32)
