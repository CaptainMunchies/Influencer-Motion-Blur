"""Influencer Blur Lab — Headless Optical & Smartphone Imperfection Engine.

Zero UI / FastAPI dependencies. Embed directly into any Python web backend or SaaS:

    from PIL import Image
    from engine import process_image, get_builtin_preset

    img = Image.open("photo.jpg")
    preset = get_builtin_preset("flash_party_blur")
    result_rgb = process_image(img, preset)
"""

from .ai_director import (
    AIDirectorResult,
    SUGGESTED_MODELS,
    apply_director_guidance_to_config,
    direct_photo_with_ai,
    list_available_vision_models,
)
from .auto_scene import AutoSceneAnalysis, analyze_and_build_auto_config
from .masks import UserBrushMask, decode_rgba_brush_mask
from .pipeline import PipelineResult, auto_process_image, process_image, run_pipeline
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
    VisionDirectorConfig,
)
from .stage2_motion import MotionProvider, register_motion_provider

__all__ = [
    "AIDirectorResult",
    "AutoSceneAnalysis",
    "BUILTIN_PRESETS",
    "CloudAIConfig",
    "MotionProvider",
    "OpticalColorConfig",
    "PipelineConfig",
    "PipelineResult",
    "SUGGESTED_MODELS",
    "Stage1FocusConfig",
    "Stage2MotionConfig",
    "Stage3OpticsConfig",
    "Stage4GrainConfig",
    "Stage5CompressionConfig",
    "UserBrushMask",
    "VisionDirectorConfig",
    "analyze_and_build_auto_config",
    "apply_director_guidance_to_config",
    "auto_process_image",
    "decode_rgba_brush_mask",
    "direct_photo_with_ai",
    "get_builtin_preset",
    "list_available_vision_models",
    "list_builtin_presets",
    "list_custom_presets",
    "load_preset_json",
    "process_image",
    "register_motion_provider",
    "run_pipeline",
    "save_preset_json",
]
