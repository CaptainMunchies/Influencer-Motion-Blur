"""JSON-serializable configuration schema for Influencer Blur Lab's headless engine.

Every stage has an independent `enabled` flag and clean numeric parameters so the
engine can be embedded directly in a SaaS API or driven by the local QA web UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

FocusFalloffMode = Literal["uniform", "face_center", "edges"]
MotionMethod = Literal["handheld_shake", "flash_drag", "pivot_swing", "cloud_ai"]
CloudProviderName = Literal["fal", "replicate"]
VisionProviderName = Literal["kie", "gemini", "openai", "openrouter"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


@dataclass
class OpticalColorConfig:
    """Automatic pre-step optical light conversion settings."""

    gamma: float = 2.2
    highlight_threshold: float = 0.85
    highlight_boost: float = 0.85  # Headroom energy multiplier for pixels > 0.85

    def validated(self) -> "OpticalColorConfig":
        return OpticalColorConfig(
            gamma=_clamp(self.gamma, 1.8, 2.6),
            highlight_threshold=_clamp(self.highlight_threshold, 0.70, 0.98),
            highlight_boost=_clamp(self.highlight_boost, 0.0, 2.5),
        )


@dataclass
class Stage1FocusConfig:
    """Stage 1: True circular aperture disc defocus + depth/focus falloff."""

    enabled: bool = True
    radius: float = 2.2  # Aperture disc radius in pixels (at 1080p reference)
    falloff_mode: FocusFalloffMode = "face_center"
    falloff_strength: float = 0.55  # 0.0 = uniform, 1.0 = strong sharp center / soft edges
    focus_center_x: float = 0.50  # Normalized 0..1
    focus_center_y: float = 0.34  # Upper-middle portrait face default
    aperture_rim_softness: float = 0.18  # Subtle plastic smartphone lens rim softness

    def validated(self) -> "Stage1FocusConfig":
        mode = self.falloff_mode if self.falloff_mode in ("uniform", "face_center", "edges") else "face_center"
        return Stage1FocusConfig(
            enabled=bool(self.enabled),
            radius=_clamp(self.radius, 0.0, 18.0),
            falloff_mode=mode,
            falloff_strength=_clamp(self.falloff_strength, 0.0, 1.0),
            focus_center_x=_clamp(self.focus_center_x, 0.0, 1.0),
            focus_center_y=_clamp(self.focus_center_y, 0.0, 1.0),
            aperture_rim_softness=_clamp(self.aperture_rim_softness, 0.0, 0.6),
        )


@dataclass
class CloudAIConfig:
    """Optional Cloud AI motion provider settings (unused unless method == 'cloud_ai')."""

    provider: CloudProviderName = "fal"
    api_key: str = ""
    model: str = ""
    prompt: str = "subtle authentic handheld iPhone night camera motion blur, natural skin"
    strength: float = 0.35

    def validated(self) -> "CloudAIConfig":
        provider = self.provider if self.provider in ("fal", "replicate") else "fal"
        return CloudAIConfig(
            provider=provider,
            api_key=str(self.api_key or "").strip(),
            model=str(self.model or "").strip(),
            prompt=str(self.prompt or "").strip(),
            strength=_clamp(self.strength, 0.05, 0.95),
        )


@dataclass
class VisionDirectorConfig:
    """Settings for the AI Art Director (guides masks, focus center & pivots; never regenerates pixels)."""

    provider: VisionProviderName = "kie"
    api_key: str = ""
    model: str = "gemini-4-neon"

    def validated(self) -> "VisionDirectorConfig":
        provider = self.provider if self.provider in ("kie", "gemini", "openai", "openrouter") else "kie"
        default_models = {
            "kie": "gemini-4-neon",
            "gemini": "gemini-4-hydrogen",
            "openai": "gpt-6.2-sol",
            "openrouter": "google/gemini-4-hydrogen",
        }
        model = str(self.model or "").strip() or default_models[provider]
        return VisionDirectorConfig(
            provider=provider,
            api_key=str(self.api_key or "").strip(),
            model=model,
        )


@dataclass
class Stage2MotionConfig:
    """Stage 2: Realistic curved motion blur, slow-sync flash drag, or arm-swing arc."""

    enabled: bool = True
    method: MotionMethod = "handheld_shake"
    shake_amount: float = 10.0  # Trajectory span in pixels (at 1080p reference)
    ghost_strength: float = 0.35  # Dwell-time spike at endpoints -> subtle double-image ghosting
    curvature: float = 0.60  # Non-linear curve bend (never a straight line)
    direction_angle: float = 25.0  # Dominant drift angle in degrees (-180..180)
    # Slow-Sync Flash Drag specifics
    flash_sharpness: float = 0.70  # Weight of sharp flash burst (0.35..0.92)
    outer_smear: float = 0.65  # Extra curved motion smear on outer limbs/edges while keeping face sharp
    ambient_warmth: float = 0.14  # Subtle warm tint on the ambient shutter-drag trail
    # Pivot / Arm Swing specifics
    swing_angle: float = 7.5  # Rotational arc angle in degrees for moving arms/hands
    swing_intensity: float = 0.80  # Blend weight on moving limb zones
    # Optional AI-detected shoulder/elbow pivot points: [{"x": 0..1, "y": 0..1, "angle_deg": float, "polygon": [[x,y],...]}]
    ai_pivots: List[Dict[str, Any]] = field(default_factory=list)
    # Deterministic seed for reproducible hand-shake curves
    seed: int = 42

    def validated(self) -> "Stage2MotionConfig":
        method = (
            self.method
            if self.method in ("handheld_shake", "flash_drag", "pivot_swing", "cloud_ai")
            else "handheld_shake"
        )
        clean_pivots: List[Dict[str, Any]] = []
        if isinstance(self.ai_pivots, list):
            for p in self.ai_pivots[:6]:
                if isinstance(p, dict) and "x" in p and "y" in p:
                    item: Dict[str, Any] = {
                        "x": _clamp(float(p["x"]), 0.0, 1.0),
                        "y": _clamp(float(p["y"]), 0.0, 1.0),
                        "angle_deg": _clamp(float(p.get("angle_deg", self.swing_angle)), -24.0, 24.0),
                    }
                    if isinstance(p.get("polygon"), list) and len(p["polygon"]) >= 3:
                        item["polygon"] = [
                            [_clamp(float(pt[0]), 0.0, 1.0), _clamp(float(pt[1]), 0.0, 1.0)]
                            for pt in p["polygon"]
                            if isinstance(pt, (list, tuple)) and len(pt) >= 2
                        ]
                    clean_pivots.append(item)
        return Stage2MotionConfig(
            enabled=bool(self.enabled),
            method=method,
            shake_amount=_clamp(self.shake_amount, 0.0, 42.0),
            ghost_strength=_clamp(self.ghost_strength, 0.0, 1.0),
            curvature=_clamp(self.curvature, 0.1, 1.0),
            direction_angle=_clamp(self.direction_angle, -180.0, 180.0),
            flash_sharpness=_clamp(self.flash_sharpness, 0.30, 0.95),
            outer_smear=_clamp(self.outer_smear, 0.0, 1.0),
            ambient_warmth=_clamp(self.ambient_warmth, 0.0, 0.5),
            swing_angle=_clamp(self.swing_angle, 1.0, 22.0),
            swing_intensity=_clamp(self.swing_intensity, 0.0, 1.0),
            ai_pivots=clean_pivots,
            seed=int(self.seed),
        )


@dataclass
class Stage3OpticsConfig:
    """Stage 3: Dirty smartphone lens veiling glare, milky haze & subtle edge fringing."""

    enabled: bool = True
    glow_amount: float = 0.35  # Linear-light highlight bloom & dirty-lens scatter (0..1)
    haze_lift: float = 0.22  # Gentle lifted milky blacks / atmospheric flash veil (0..1)
    warmth: float = 0.18  # Warm tint in highlight bloom (0..1)
    edge_fringing: float = 0.20  # Subtle radial chromatic aberration near edges (0..1)

    def validated(self) -> "Stage3OpticsConfig":
        return Stage3OpticsConfig(
            enabled=bool(self.enabled),
            glow_amount=_clamp(self.glow_amount, 0.0, 1.0),
            haze_lift=_clamp(self.haze_lift, 0.0, 1.0),
            warmth=_clamp(self.warmth, 0.0, 1.0),
            edge_fringing=_clamp(self.edge_fringing, 0.0, 1.0),
        )


@dataclass
class Stage4GrainConfig:
    """Stage 4: Nighttime smartphone high-ISO grain & clumpy ISP look (applied AFTER blur)."""

    enabled: bool = True
    grain_amount: float = 0.35  # Overall low-light sensor grain strength (0..1)
    clumpiness: float = 0.55  # Clumpy iPhone Night Mode feel vs fine static (0..1)
    isp_smoothing: float = 0.35  # Light edge-preserving smartphone noise reduction (0..1)
    shadow_weight: float = 0.72  # Extra noise weight in dark shadows vs bright skin (0..1)
    seed: int = 101

    def validated(self) -> "Stage4GrainConfig":
        return Stage4GrainConfig(
            enabled=bool(self.enabled),
            grain_amount=_clamp(self.grain_amount, 0.0, 1.0),
            clumpiness=_clamp(self.clumpiness, 0.0, 1.0),
            isp_smoothing=_clamp(self.isp_smoothing, 0.0, 1.0),
            shadow_weight=_clamp(self.shadow_weight, 0.0, 1.0),
            seed=int(self.seed),
        )


@dataclass
class Stage5CompressionConfig:
    """Stage 5: Final Instagram upload softness & authentic 4:2:0 JPEG compression."""

    enabled: bool = True
    insta_softness: float = 0.25  # Gentle downscale-then-upscale softness (0..1)
    jpeg_quality: int = 82  # Authentic 4:2:0 JPEG quality (72..95)

    def validated(self) -> "Stage5CompressionConfig":
        return Stage5CompressionConfig(
            enabled=bool(self.enabled),
            insta_softness=_clamp(self.insta_softness, 0.0, 1.0),
            jpeg_quality=int(_clamp(self.jpeg_quality, 60, 98)),
        )


@dataclass
class PipelineConfig:
    """Top-level JSON preset schema for the 5-stage Influencer Blur Lab engine."""

    name: str = "Custom Look"
    description: str = ""
    color: OpticalColorConfig = field(default_factory=OpticalColorConfig)
    stage1_focus: Stage1FocusConfig = field(default_factory=Stage1FocusConfig)
    stage2_motion: Stage2MotionConfig = field(default_factory=Stage2MotionConfig)
    stage3_optics: Stage3OpticsConfig = field(default_factory=Stage3OpticsConfig)
    stage4_grain: Stage4GrainConfig = field(default_factory=Stage4GrainConfig)
    stage5_compression: Stage5CompressionConfig = field(default_factory=Stage5CompressionConfig)

    def validated(self) -> "PipelineConfig":
        return PipelineConfig(
            name=str(self.name or "Custom Look"),
            description=str(self.description or ""),
            color=self.color.validated(),
            stage1_focus=self.stage1_focus.validated(),
            stage2_motion=self.stage2_motion.validated(),
            stage3_optics=self.stage3_optics.validated(),
            stage4_grain=self.stage4_grain.validated(),
            stage5_compression=self.stage5_compression.validated(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.validated())

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PipelineConfig":
        if not data:
            return cls().validated()

        def _pick(dc_cls, sub: Optional[Dict[str, Any]]):
            if not isinstance(sub, dict):
                return dc_cls()
            valid_keys = {f.name for f in dc_cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in sub.items() if k in valid_keys}
            return dc_cls(**filtered)

        return cls(
            name=data.get("name", "Custom Look"),
            description=data.get("description", ""),
            color=_pick(OpticalColorConfig, data.get("color")),
            stage1_focus=_pick(Stage1FocusConfig, data.get("stage1_focus")),
            stage2_motion=_pick(Stage2MotionConfig, data.get("stage2_motion")),
            stage3_optics=_pick(Stage3OpticsConfig, data.get("stage3_optics")),
            stage4_grain=_pick(Stage4GrainConfig, data.get("stage4_grain")),
            stage5_compression=_pick(Stage5CompressionConfig, data.get("stage5_compression")),
        ).validated()
