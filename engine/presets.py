"""Built-in presets calibrated to the 3 target Instagram smartphone looks + JSON I/O."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from .schema import (
    OpticalColorConfig,
    PipelineConfig,
    Stage1FocusConfig,
    Stage2MotionConfig,
    Stage3OpticsConfig,
    Stage4GrainConfig,
    Stage5CompressionConfig,
)

BUILTIN_PRESETS: Dict[str, PipelineConfig] = {
    "flash_party_blur": PipelineConfig(
        name="Flash Party Blur",
        description=(
            "Slow-sync party flash look: face stays crisp, subtle background double-image "
            "ghosting, moving arms & edges trail naturally, smudged flash glow, and gritty night grain."
        ),
        color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.84, highlight_boost=1.05),
        stage1_focus=Stage1FocusConfig(
            enabled=True,
            radius=1.35,
            falloff_mode="face_center",
            falloff_strength=0.72,
            focus_center_x=0.50,
            focus_center_y=0.30,
            aperture_rim_softness=0.20,
        ),
        stage2_motion=Stage2MotionConfig(
            enabled=True,
            method="flash_drag",
            shake_amount=14.0,
            ghost_strength=0.48,
            curvature=0.68,
            direction_angle=18.0,
            flash_sharpness=0.68,
            outer_smear=0.74,
            ambient_warmth=0.18,
            swing_angle=8.5,
            swing_intensity=0.80,
            seed=42,
        ),
        stage3_optics=Stage3OpticsConfig(
            enabled=True,
            glow_amount=0.40,
            haze_lift=0.30,
            warmth=0.18,
            edge_fringing=0.22,
        ),
        stage4_grain=Stage4GrainConfig(
            enabled=True,
            grain_amount=0.46,
            clumpiness=0.58,
            isp_smoothing=0.34,
            shadow_weight=0.78,
            seed=101,
        ),
        stage5_compression=Stage5CompressionConfig(
            enabled=True,
            insta_softness=0.22,
            jpeg_quality=82,
        ),
    ),
    "warm_night_missed_focus": PipelineConfig(
        name="Warm Night Missed Focus",
        description=(
            "Warm low-light missed autofocus: true circular lens softness with soft foreground & "
            "edge falloff, subtle diagonal handheld shake, golden highlight bloom, and soft phone grain."
        ),
        color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.83, highlight_boost=1.15),
        stage1_focus=Stage1FocusConfig(
            enabled=True,
            radius=6.8,
            falloff_mode="face_center",
            falloff_strength=0.62,
            focus_center_x=0.48,
            focus_center_y=0.26,
            aperture_rim_softness=0.22,
        ),
        stage2_motion=Stage2MotionConfig(
            enabled=True,
            method="handheld_shake",
            shake_amount=8.5,
            ghost_strength=0.26,
            curvature=0.58,
            direction_angle=38.0,
            flash_sharpness=0.72,
            outer_smear=0.45,
            ambient_warmth=0.24,
            swing_angle=6.0,
            swing_intensity=0.70,
            seed=77,
        ),
        stage3_optics=Stage3OpticsConfig(
            enabled=True,
            glow_amount=0.46,
            haze_lift=0.18,
            warmth=0.38,
            edge_fringing=0.20,
        ),
        stage4_grain=Stage4GrainConfig(
            enabled=True,
            grain_amount=0.32,
            clumpiness=0.66,
            isp_smoothing=0.48,
            shadow_weight=0.70,
            seed=202,
        ),
        stage5_compression=Stage5CompressionConfig(
            enabled=True,
            insta_softness=0.28,
            jpeg_quality=80,
        ),
    ),
    "quick_gym_mirror_snap": PipelineConfig(
        name="Quick Gym Mirror Snap",
        description=(
            "Whole-frame slight curved thumb-press shutter shake, mild soft focus, light "
            "dirty-mirror veiling haze, and realistic Instagram upload compression."
        ),
        color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.86, highlight_boost=0.80),
        stage1_focus=Stage1FocusConfig(
            enabled=True,
            radius=3.2,
            falloff_mode="uniform",
            falloff_strength=0.0,
            focus_center_x=0.50,
            focus_center_y=0.32,
            aperture_rim_softness=0.18,
        ),
        stage2_motion=Stage2MotionConfig(
            enabled=True,
            method="handheld_shake",
            shake_amount=10.5,
            ghost_strength=0.38,
            curvature=0.62,
            direction_angle=-22.0,
            flash_sharpness=0.72,
            outer_smear=0.40,
            ambient_warmth=0.08,
            swing_angle=6.5,
            swing_intensity=0.70,
            seed=303,
        ),
        stage3_optics=Stage3OpticsConfig(
            enabled=True,
            glow_amount=0.30,
            haze_lift=0.24,
            warmth=0.06,
            edge_fringing=0.16,
        ),
        stage4_grain=Stage4GrainConfig(
            enabled=True,
            grain_amount=0.24,
            clumpiness=0.50,
            isp_smoothing=0.32,
            shadow_weight=0.65,
            seed=309,
        ),
        stage5_compression=Stage5CompressionConfig(
            enabled=True,
            insta_softness=0.36,
            jpeg_quality=78,
        ),
    ),
}


def list_builtin_presets() -> List[Dict]:
    """Return all built-in presets with stable IDs and serialized configs."""
    out = []
    for preset_id, cfg in BUILTIN_PRESETS.items():
        out.append({
            "id": preset_id,
            "builtin": True,
            "name": cfg.name,
            "description": cfg.description,
            "config": cfg.to_dict(),
        })
    return out


def get_builtin_preset(preset_id: str) -> PipelineConfig:
    """Return a validated copy of a built-in preset by ID."""
    if preset_id not in BUILTIN_PRESETS:
        raise KeyError(f"Unknown built-in preset: {preset_id}")
    return PipelineConfig.from_dict(BUILTIN_PRESETS[preset_id].to_dict())


def slugify_preset_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower()).strip("_")
    return slug or "custom_preset"


def save_preset_json(config: PipelineConfig, directory: Path | str, preset_id: str | None = None) -> Path:
    """Save a PipelineConfig as clean JSON inside `directory`."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    pid = slugify_preset_name(preset_id or config.name)
    path = folder / f"{pid}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
    return path


def load_preset_json(path: Path | str) -> PipelineConfig:
    """Load a PipelineConfig from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return PipelineConfig.from_dict(data)


def list_custom_presets(directory: Path | str) -> List[Dict]:
    """List user-saved custom JSON presets in `directory`."""
    folder = Path(directory)
    if not folder.exists():
        return []
    items = []
    for path in sorted(folder.glob("*.json")):
        try:
            cfg = load_preset_json(path)
            items.append({
                "id": path.stem,
                "builtin": False,
                "name": cfg.name,
                "description": cfg.description,
                "config": cfg.to_dict(),
            })
        except Exception:
            continue
    return items
