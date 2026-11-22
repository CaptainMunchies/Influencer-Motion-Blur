"""Stage 5: Final Instagram Compression Polish.

Applies optional subtle downscale-then-upscale softness (matching Instagram's mobile
upload resizing) followed by an authentic 4:2:0 chroma-subsampled JPEG compression pass.
"""

from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image

from .schema import Stage5CompressionConfig


def _apply_accidental_phone_framing(
    rgb_u8: np.ndarray,
    tilt_angle: float,
    crop_offset: float,
    framing_seed: int,
) -> np.ndarray:
    """Apply realistic slight phone tilt + casual off-center framing with exact zero-black-corner zoom."""
    if abs(tilt_angle) < 0.05 and crop_offset < 0.02:
        return rgb_u8

    h, w = rgb_u8.shape[:2]
    rad = math.radians(abs(tilt_angle))
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # Exact minimum uniform zoom so rotated corners never expose out-of-frame pixels
    min_rot_zoom = max(cos_a + sin_a * (h / max(1.0, float(w))), cos_a + sin_a * (w / max(1.0, float(h))))
    extra_crop_zoom = 1.0 + 0.048 * crop_offset
    zoom = min_rot_zoom * extra_crop_zoom

    # Deterministic casual off-center shift bounded safely inside the zoom margin
    rng = np.random.default_rng(int(framing_seed) & 0xFFFFFFFF)
    dir_x = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.55, 1.0))
    dir_y = float(rng.choice([-1.0, 1.0])) * float(rng.uniform(0.35, 0.85))
    max_shift_frac = max(0.0, (extra_crop_zoom - 1.0) * 0.42)
    shift_x_px = dir_x * max_shift_frac * w
    shift_y_px = dir_y * max_shift_frac * h

    cx, cy = w * 0.5, h * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), float(tilt_angle), float(zoom))
    M[0, 2] += shift_x_px
    M[1, 2] += shift_y_px
    return cv2.warpAffine(rgb_u8, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)


def apply_stage5_compression(srgb_u8: np.ndarray, cfg: Stage5CompressionConfig) -> np.ndarray:
    """Apply optional casual phone framing (tilt + crop), upload softness, and 4:2:0 JPEG compression."""
    cfg = cfg.validated()
    if not cfg.enabled:
        return srgb_u8.copy()

    h, w = srgb_u8.shape[:2]
    working = _apply_accidental_phone_framing(
        srgb_u8,
        tilt_angle=cfg.tilt_angle,
        crop_offset=cfg.crop_offset,
        framing_seed=cfg.framing_seed,
    )

    # 1. Optional gentle downscale-then-upscale softness (keeps exact output dimensions)
    if cfg.insta_softness > 0.02:
        # 0.0 -> 1.0x (no resize), 1.0 -> 0.72x roundtrip
        factor = 1.0 - 0.28 * cfg.insta_softness
        sw = max(64, int(round(w * factor)))
        sh = max(64, int(round(h * factor)))
        if sw < w and sh < h:
            down = cv2.resize(working, (sw, sh), interpolation=cv2.INTER_AREA)
            working = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)

    # 2. Authentic 4:2:0 JPEG compression roundtrip via Pillow (subsampling=2 is 4:2:0)
    pil_img = Image.fromarray(working, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(cfg.jpeg_quality), subsampling=2, optimize=False)
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")
    return np.array(decoded, dtype=np.uint8)
