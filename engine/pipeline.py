"""Deterministic 5-Stage Influencer Blur Lab Pipeline Runner.

This module is 100% headless and SaaS-ready:
- Call `run_pipeline(image, config)` to get every intermediate stage (Original + Stages 1..5)
  plus per-stage timing breakdown for side-by-side QA inspection.
- Call `process_image(image, config)` for single-shot production SaaS usage returning only
  the final processed image.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from .color import linear_to_srgb_u8, srgb_u8_to_linear
from .masks import UserBrushMask, decode_rgba_brush_mask
from .schema import CloudAIConfig, PipelineConfig
from .stage1_focus import apply_stage1_focus
from .stage2_motion import apply_stage2_motion
from .stage3_optics import apply_stage3_optics
from .stage4_grain import apply_stage4_grain
from .stage5_compression import apply_stage5_compression

ImageInput = Union[np.ndarray, Image.Image]


@dataclass
class PipelineResult:
    """Full intermediate output of all 5 stages + timing & metadata."""

    stages: Dict[str, np.ndarray]  # keys: original, stage1, stage2, stage3, stage4, stage5
    timings_ms: Dict[str, float]
    total_ms: float
    motion_provider: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def final(self) -> np.ndarray:
        return self.stages["stage5"]


def _coerce_rgb_u8(image: ImageInput) -> np.ndarray:
    """Normalize PIL Image or numpy array into contiguous uint8 RGB (H, W, 3)."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.0:
                arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def run_pipeline(
    image: ImageInput,
    config: Union[PipelineConfig, Dict[str, Any], None] = None,
    brush_rgba: Optional[np.ndarray] = None,
    brush_mask: Optional[UserBrushMask] = None,
    cloud_config: Optional[CloudAIConfig] = None,
    return_intermediates: bool = True,
) -> PipelineResult:
    """Execute the deterministic 5-stage optical + smartphone ISP pipeline.

    Pre-step (automatic inside optical Stages 1–3):
      Converts sRGB -> Linear RGB (gamma 2.2) with highlight headroom boost (>0.85)
      before any optical blur/glow, then converts back to sRGB before Stage 4 grain.
    """
    t_start = time.perf_counter()
    cfg = config.validated() if isinstance(config, PipelineConfig) else PipelineConfig.from_dict(config)
    original_u8 = _coerce_rgb_u8(image)
    h, w = original_u8.shape[:2]

    resolved_brush = brush_mask
    if resolved_brush is None and brush_rgba is not None:
        resolved_brush = decode_rgba_brush_mask(brush_rgba, (h, w))

    stages: Dict[str, np.ndarray] = {}
    if return_intermediates:
        stages["original"] = original_u8

    timings: Dict[str, float] = {}
    notes: list[str] = []

    # Pre-step: sRGB -> Linear RGB with highlight headroom boost
    t0 = time.perf_counter()
    lin = srgb_u8_to_linear(original_u8, cfg.color)
    timings["linearize"] = round((time.perf_counter() - t0) * 1000.0, 2)

    # Stage 1: Circular Aperture Defocus + Focus Falloff
    t1 = time.perf_counter()
    lin_s1 = apply_stage1_focus(lin, cfg.stage1_focus, brush=resolved_brush)
    timings["stage1"] = round((time.perf_counter() - t1) * 1000.0, 2)
    if return_intermediates:
        stages["stage1"] = linear_to_srgb_u8(lin_s1, cfg.color) if cfg.stage1_focus.enabled else original_u8

    # Stage 2: Realistic Motion Blur / Slow-Sync Flash Drag / Pivot Arm Swing / Cloud AI
    t2 = time.perf_counter()
    motion_res = apply_stage2_motion(
        lin_s1,
        cfg.stage2_motion,
        brush=resolved_brush,
        color_cfg=cfg.color,
        cloud_cfg=cloud_config,
    )
    lin_s2 = motion_res.linear_rgb
    if motion_res.note:
        notes.append(motion_res.note)
    timings["stage2"] = round((time.perf_counter() - t2) * 1000.0, 2)
    if return_intermediates:
        stages["stage2"] = linear_to_srgb_u8(lin_s2, cfg.color)

    # Stage 3: Dirty Smartphone Lens Glow, Atmospheric Haze & Edge Fringing
    t3 = time.perf_counter()
    lin_s3 = apply_stage3_optics(lin_s2, cfg.stage3_optics)
    srgb_s3 = linear_to_srgb_u8(lin_s3, cfg.color)
    timings["stage3"] = round((time.perf_counter() - t3) * 1000.0, 2)
    if return_intermediates:
        stages["stage3"] = srgb_s3

    # Stage 4: Nighttime Phone High-ISO Grain & Clumpy ISP Look (Applied AFTER blur!)
    t4 = time.perf_counter()
    srgb_s4 = apply_stage4_grain(srgb_s3, cfg.stage4_grain)
    timings["stage4"] = round((time.perf_counter() - t4) * 1000.0, 2)
    if return_intermediates:
        stages["stage4"] = srgb_s4

    # Stage 5: Final Instagram Compression Polish (softness + 4:2:0 JPEG pass)
    t5 = time.perf_counter()
    srgb_s5 = apply_stage5_compression(srgb_s4, cfg.stage5_compression)
    timings["stage5"] = round((time.perf_counter() - t5) * 1000.0, 2)
    stages["stage5"] = srgb_s5

    total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    return PipelineResult(
        stages=stages,
        timings_ms=timings,
        total_ms=total_ms,
        motion_provider=motion_res.provider_used,
        notes=notes,
    )


def process_image(
    image: ImageInput,
    config: Union[PipelineConfig, Dict[str, Any], None] = None,
    brush_rgba: Optional[np.ndarray] = None,
    cloud_config: Optional[CloudAIConfig] = None,
) -> np.ndarray:
    """Clean one-liner SaaS API: takes an image + JSON dict/PipelineConfig, returns final RGB uint8."""
    res = run_pipeline(
        image=image,
        config=config,
        brush_rgba=brush_rgba,
        cloud_config=cloud_config,
        return_intermediates=False,
    )
    return res.final
