"""Stage 5: Final Instagram Compression Polish.

Applies optional subtle downscale-then-upscale softness (matching Instagram's mobile
upload resizing) followed by an authentic 4:2:0 chroma-subsampled JPEG compression pass.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from .schema import Stage5CompressionConfig


def apply_stage5_compression(srgb_u8: np.ndarray, cfg: Stage5CompressionConfig) -> np.ndarray:
    """Apply subtle upload softness + realistic 4:2:0 JPEG compression pass."""
    cfg = cfg.validated()
    if not cfg.enabled:
        return srgb_u8.copy()

    h, w = srgb_u8.shape[:2]
    working = srgb_u8

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
