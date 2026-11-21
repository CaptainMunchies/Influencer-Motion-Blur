"""Influencer Blur Lab — Headless Optical & Smartphone Imperfection Engine.

Zero UI / FastAPI dependencies. Embed directly into any Python web backend or SaaS:

    from PIL import Image
    from engine import process_image, get_builtin_preset

    img = Image.open("photo.jpg")
    preset = get_builtin_preset("flash_party_blur")
    result_rgb = process_image(img, preset)
"""

from .masks import UserBrushMask, decode_rgba_brush_mask
from .pipeline import PipelineResult, process_image, run_pipeline
from .presets import (
    BUILTIN_PRESETS,
    get_builtin_preset,
    list_builtin_presets,
    list_custom_presets,
    load_preset_json,
    save_preset_json,
)
from .schema import (
    CloudAIConfig,
    OpticalColorConfig,
    PipelineConfig,
    Stage1FocusConfig,
    Stage2MotionConfig,
    Stage3OpticsConfig,
    Stage4GrainConfig,
    Stage5CompressionConfig,
)
from .stage2_motion import MotionProvider, register_motion_provider

__all__ = [
    "BUILTIN_PRESETS",
    "CloudAIConfig",
    "MotionProvider",
    "OpticalColorConfig",
    "PipelineConfig",
    "PipelineResult",
    "Stage1FocusConfig",
    "Stage2MotionConfig",
    "Stage3OpticsConfig",
    "Stage4GrainConfig",
    "Stage5CompressionConfig",
    "UserBrushMask",
    "decode_rgba_brush_mask",
    "get_builtin_preset",
    "list_builtin_presets",
    "list_custom_presets",
    "load_preset_json",
    "process_image",
    "register_motion_provider",
    "run_pipeline",
    "save_preset_json",
]
