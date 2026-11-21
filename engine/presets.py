"""Built-in presets calibrated through visual QA across diverse real candid IG photos + JSON I/O."""

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
            "Slow-sync party flash look: face stays crisp & flattering, subtle double-image "
            "ambient ghosting, trailing outer arms/edges, smudged flash glow, and organic night grain."
        ),
        color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.85, highlight_boost=0.85),
        stage1_focus=Stage1FocusConfig(
            enabled=True,
            radius=1.15,
            falloff_mode="face_center",
            falloff_strength=0.78,
            focus_center_x=0.50,
            focus_center_y=0.31,
            aperture_rim_softness=0.20,
        ),
        stage2_motion=Stage2MotionConfig(
            enabled=True,
            method="flash_drag",
            shake_amount=10.5,
            ghost_strength=0.42,
            curvature=0.66,
            direction_angle=18.0,
            flash_sharpness=0.74,
            outer_smear=0.64,
            ambient_warmth=0.15,
            swing_angle=8.0,
            swing_intensity=0.78,
            seed=42,
        ),
        stage3_optics=Stage3OpticsConfig(
            enabled=True,
            glow_amount=0.36,
            haze_lift=0.28,
            warmth=0.18,
            edge_fringing=0.20,
        ),
        stage4_grain=Stage4GrainConfig(
            enabled=True,
            grain_amount=0.38,
            clumpiness=0.58,
            isp_smoothing=0.34,
            shadow_weight=0.80,
            seed=101,
        ),
        stage5_compression=Stage5CompressionConfig(
            enabled=True,
            insta_softness=0.18,
            jpeg_quality=83,
        ),
    ),
    "warm_night_missed_focus": PipelineConfig(
        name="Warm Night Missed Focus",
        description=(
            "Warm low-light missed autofocus: dreamy circular lens softness with foreground & "
            "edge falloff, subtle curved hand shake, golden highlight bloom, and velvety night grain."
        ),
        color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.84, highlight_boost=1.00),
        stage1_focus=Stage1FocusConfig(
            enabled=True,
            radius=4.2,
            falloff_mode="face_center",
            falloff_strength=0.68,
            focus_center_x=0.50,
            focus_center_y=0.28,
            aperture_rim_softness=0.22,
        ),
        stage2_motion=Stage2MotionConfig(
            enabled=True,
            method="handheld_shake",
            shake_amount=5.8,
            ghost_strength=0.22,
            curvature=0.58,
            direction_angle=32.0,
            flash_sharpness=0.74,
            outer_smear=0.40,
            ambient_warmth=0.22,
            swing_angle=6.0,
            swing_intensity=0.70,
            seed=77,
        ),
        stage3_optics=Stage3OpticsConfig(
            enabled=True,
            glow_amount=0.40,
            haze_lift=0.14,
            warmth=0.38,
            edge_fringing=0.18,
        ),
        stage4_grain=Stage4GrainConfig(
            enabled=True,
            grain_amount=0.28,
            clumpiness=0.66,
            isp_smoothing=0.44,
            shadow_weight=0.76,
            seed=202,
        ),
        stage5_compression=Stage5CompressionConfig(
            enabled=True,
            insta_softness=0.22,
            jpeg_quality=82,
        ),
    ),
    "quick_gym_mirror_snap": PipelineConfig(
        name="Quick Gym Mirror Snap",
        description=(
            "Casual mirror selfie feel: subtle whole-frame thumb-press curved shutter shake, "
            "gentle softness, light dirty-mirror sheen with rich blacks, and natural Instagram polish."
        ),
        color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.87, highlight_boost=0.68),
        stage1_focus=Stage1FocusConfig(
            enabled=True,
            radius=2.1,
            falloff_mode="face_center",
            falloff_strength=0.35,
            focus_center_x=0.50,
            focus_center_y=0.30,
            aperture_rim_softness=0.18,
        ),
        stage2_motion=Stage2MotionConfig(
            enabled=True,
            method="handheld_shake",
            shake_amount=6.8,
            ghost_strength=0.32,
            curvature=0.60,
            direction_angle=-20.0,
            flash_sharpness=0.74,
            outer_smear=0.36,
            ambient_warmth=0.06,
            swing_angle=6.5,
            swing_intensity=0.70,
            seed=303,
        ),
        stage3_optics=Stage3OpticsConfig(
            enabled=True,
            glow_amount=0.24,
            haze_lift=0.10,
            warmth=0.06,
            edge_fringing=0.14,
        ),
        stage4_grain=Stage4GrainConfig(
            enabled=True,
            grain_amount=0.20,
            clumpiness=0.52,
            isp_smoothing=0.30,
            shadow_weight=0.72,
            seed=309,
        ),
        stage5_compression=Stage5CompressionConfig(
            enabled=True,
            insta_softness=0.24,
            jpeg_quality=81,
        ),
    ),
    "daylight_candid_snap": PipelineConfig(
        name="Daylight Street Candid",
        description=(
            "Crisp daytime/outdoor candid snap: preserves rich contrast and clean bright skin, "
            "adds subtle spontaneous handheld movement, faint sunlit lens sheen, and authentic IG texture."
        ),
        color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.88, highlight_boost=0.52),
        stage1_focus=Stage1FocusConfig(
            enabled=True,
            radius=1.35,
            falloff_mode="face_center",
            falloff_strength=0.78,
            focus_center_x=0.50,
            focus_center_y=0.32,
            aperture_rim_softness=0.18,
        ),
        stage2_motion=Stage2MotionConfig(
            enabled=True,
            method="flash_drag",
            shake_amount=6.5,
            ghost_strength=0.24,
            curvature=0.58,
            direction_angle=-16.0,
            flash_sharpness=0.78,
            outer_smear=0.42,
            ambient_warmth=0.05,
            swing_angle=6.0,
            swing_intensity=0.70,
            seed=404,
        ),
        stage3_optics=Stage3OpticsConfig(
            enabled=True,
            glow_amount=0.22,
            haze_lift=0.03,
            warmth=0.08,
            edge_fringing=0.14,
        ),
        stage4_grain=Stage4GrainConfig(
            enabled=True,
            grain_amount=0.14,
            clumpiness=0.46,
            isp_smoothing=0.22,
            shadow_weight=0.70,
            seed=411,
        ),
        stage5_compression=Stage5CompressionConfig(
            enabled=True,
            insta_softness=0.18,
            jpeg_quality=85,
        ),
    ),
    "moving_hands_party": PipelineConfig(
        name="Moving Hands Party",
        description=(
            "Crisp face & background while swinging arms, bracelets, and gesturing hands arc "
            "naturally in motion — great for candid poses in front of walls or crowds."
        ),
        color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.85, highlight_boost=0.80),
        stage1_focus=Stage1FocusConfig(
            enabled=True,
            radius=0.85,
            falloff_mode="face_center",
            falloff_strength=0.82,
            focus_center_x=0.50,
            focus_center_y=0.30,
            aperture_rim_softness=0.18,
        ),
        stage2_motion=Stage2MotionConfig(
            enabled=True,
            method="pivot_swing",
            shake_amount=4.5,
            ghost_strength=0.34,
            curvature=0.62,
            direction_angle=15.0,
            flash_sharpness=0.76,
            outer_smear=0.65,
            ambient_warmth=0.14,
            swing_angle=8.5,
            swing_intensity=0.82,
            seed=505,
        ),
        stage3_optics=Stage3OpticsConfig(
            enabled=True,
            glow_amount=0.30,
            haze_lift=0.12,
            warmth=0.20,
            edge_fringing=0.16,
        ),
        stage4_grain=Stage4GrainConfig(
            enabled=True,
            grain_amount=0.32,
            clumpiness=0.54,
            isp_smoothing=0.28,
            shadow_weight=0.76,
            seed=512,
        ),
        stage5_compression=Stage5CompressionConfig(
            enabled=True,
            insta_softness=0.16,
            jpeg_quality=84,
        ),
    ),
    "club_whip_swirl": PipelineConfig(
        name="Club Whip-Pan Swirl",
        description=(
            "Viral iPhone wrist-flick photo-dump look: subject stays recognizable at center while "
            "the room, club lights, and outer edges sweep in a curved whip-pan shutter swirl."
        ),
        color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.84, highlight_boost=0.95),
        stage1_focus=Stage1FocusConfig(
            enabled=True,
            radius=1.25,
            falloff_mode="face_center",
            falloff_strength=0.76,
            focus_center_x=0.50,
            focus_center_y=0.32,
            aperture_rim_softness=0.20,
        ),
        stage2_motion=Stage2MotionConfig(
            enabled=True,
            method="camera_swirl",
            shake_amount=12.0,
            ghost_strength=0.40,
            curvature=0.68,
            direction_angle=22.0,
            flash_sharpness=0.66,
            outer_smear=0.60,
            ambient_warmth=0.18,
            swing_angle=6.8,
            swing_intensity=0.80,
            seed=606,
        ),
        stage3_optics=Stage3OpticsConfig(
            enabled=True,
            glow_amount=0.38,
            haze_lift=0.18,
            warmth=0.24,
            edge_fringing=0.22,
        ),
        stage4_grain=Stage4GrainConfig(
            enabled=True,
            grain_amount=0.34,
            clumpiness=0.58,
            isp_smoothing=0.34,
            shadow_weight=0.80,
            seed=613,
        ),
        stage5_compression=Stage5CompressionConfig(
            enabled=True,
            insta_softness=0.18,
            jpeg_quality=83,
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
