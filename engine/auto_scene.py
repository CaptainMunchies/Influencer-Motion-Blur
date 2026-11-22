"""Automatic Scene & Subject Analyzer (Layer 1 — Zero-Effort Tasteful Auto-Match).

Inspects any uploaded Instagram / social-media portrait in ~25ms on CPU (zero GPU,
zero external models, zero API key required) to determine:
  1. Exact Subject / Face Focus Center `(fx, fy)` using skin-tone + local detail saliency
     (so off-center subjects in street, cafe, car, or window shots stay cleanly protected!).
  2. Existing Photo Sharpness & Existing Glow Guard (never double-blurs or over-blooms
     a photo that is already soft, low-res, or heavily backlit/reflective!).
  3. Scene Lighting Archetype (5 authentic social-media photography profiles):
     - `flash_party`      : Direct flash / high-contrast indoor night party
     - `dark_night_moody` : Very dark ambient night / club / hallway low-light
     - `warm_night`       : Warm restaurant / bar / evening lights / golden sunset
     - `indoor_candid`    : Bright indoor / gym / mirror / cafe casual snapshot
     - `daylight_street`  : Bright outdoor / daylight / high-key candid
  4. Generates a deterministic, tasteful `PipelineConfig` with an optional single
     `strength` multiplier (0.0..1.6, default 1.0) so non-technical users can just
     drop ANY photo and get a believable, flattering accidental-iPhone look immediately.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .schema import (
    OpticalColorConfig,
    PipelineConfig,
    Stage1FocusConfig,
    Stage2MotionConfig,
    Stage3OpticsConfig,
    Stage4GrainConfig,
    Stage5CompressionConfig,
)


@dataclass
class AutoSceneAnalysis:
    """Result of automatic local scene inspection + generated optimal PipelineConfig."""

    archetype: str
    archetype_label: str
    plain_summary: str
    subject_center: Tuple[float, float]  # normalized (x, y)
    existing_sharpness: float
    sharpness_scale: float  # 0.45..1.05 automatic guard against double-blurring
    metrics: Dict[str, float]
    config: PipelineConfig
    scene_role: str = "portrait"  # "portrait" | "non_face_object"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "archetype": self.archetype,
            "archetype_label": self.archetype_label,
            "plain_summary": self.plain_summary,
            "scene_role": self.scene_role,
            "subject_center": [round(self.subject_center[0], 3), round(self.subject_center[1], 3)],
            "existing_sharpness": round(self.existing_sharpness, 1),
            "sharpness_scale": round(self.sharpness_scale, 2),
            "metrics": {k: round(v, 2) for k, v in self.metrics.items()},
            "config": self.config.to_dict(),
        }


def _detect_subject_center_and_role(rgb_u8: np.ndarray) -> Tuple[float, float, str]:
    """Estimate focal center (0..1, 0..1) AND whether the image is a portrait or non-face object shot."""
    h, w = rgb_u8.shape[:2]
    scale = 240.0 / max(h, w)
    sw, sh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
    small = cv2.resize(rgb_u8, (sw, sh), interpolation=cv2.INTER_AREA)

    ycrcb = cv2.cvtColor(small, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y_c, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    skin_bin = (
        (y_c > 35.0)
        & (cr >= 131.0) & (cr <= 178.0)
        & (cb >= 74.0) & (cb <= 130.0)
    ).astype(np.float32)
    skin = cv2.GaussianBlur(skin_bin, (0, 0), sigmaX=max(2.0, sw * 0.04))

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_norm = np.clip(grad / (np.percentile(grad, 95) + 1e-4), 0.0, 1.0)
    grad_smooth = cv2.GaussianBlur(grad_norm, (0, 0), sigmaX=max(3.0, sw * 0.05))

    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    xn = xx / max(1, sw - 1)
    yn = yy / max(1, sh - 1)
    border_fade = (
        np.clip(xn / 0.14, 0.0, 1.0)
        * np.clip((1.0 - xn) / 0.14, 0.0, 1.0)
        * np.clip(yn / 0.08, 0.0, 1.0)
        * np.clip((1.0 - yn) / 0.20, 0.0, 1.0)
    )
    upper_bias = np.exp(-(((yn - 0.31) / 0.25) ** 2)) * np.exp(-(((xn - 0.50) / 0.44) ** 2))
    leg_suppress = np.where(yn > 0.62, np.clip(1.0 - (yn - 0.62) * 2.2, 0.15, 1.0), 1.0)

    # Role detection: distinguish real portraits (including tank tops, bikinis & full-body shots!)
    # from true non-face object/tabletop shots (wooden tables, drinks, food, shoes, streets).
    hsv_small = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
    h_s, s_s, v_s = hsv_small[..., 0], hsv_small[..., 1], hsv_small[..., 2]
    role_skin_bin = (
        (y_c > 40.0)
        & (cr >= 135.0) & (cr <= 172.0)
        & (cb >= 82.0) & (cb <= 126.0)
    ).astype(np.float32)
    role_skin = cv2.GaussianBlur(role_skin_bin, (0, 0), sigmaX=max(2.0, sw * 0.04))
    upper_skin = role_skin[int(sh * 0.06) : int(sh * 0.58), int(sw * 0.14) : int(sw * 0.86)]
    upper_skin_peak = float(np.max(upper_skin)) if upper_skin.size else 0.0
    upper_skin_mean = float(np.mean(upper_skin)) if upper_skin.size else 0.0

    # High-saturation amber/orange wood or food flooding the lower half (human skin has much lower saturation/amber hue)
    amber_wood_bin = (
        (h_s >= 12.0) & (h_s <= 26.0) & (s_s >= 105.0) & (v_s >= 55.0)
    ).astype(np.float32)
    lower_amber = float(np.mean(amber_wood_bin[int(sh * 0.45) :, :]))
    upper_amber = float(np.mean(amber_wood_bin[: int(sh * 0.45), :]))

    is_table_or_object = (
        (upper_skin_peak < 0.22 or upper_skin_mean < 0.012)
        or (lower_amber > 0.34 and lower_amber > upper_amber * 1.7)
    )
    is_portrait = not is_table_or_object
    role = "portrait" if is_portrait else "non_face_object"

    if is_portrait:
        score = (skin * (0.45 + 0.55 * grad_smooth)) * border_fade * (0.25 + 0.75 * upper_bias) * leg_suppress
    else:
        # Non-face filler shot (table, drinks, food, pet, shoes): anchor on sharpest central-lower foreground detail
        obj_prior = np.exp(-(((yn - 0.52) / 0.34) ** 2)) * np.exp(-(((xn - 0.50) / 0.42) ** 2))
        score = grad_smooth * border_fade * (0.35 + 0.65 * obj_prior)

    score = cv2.GaussianBlur(score, (0, 0), sigmaX=max(4.0, min(sw, sh) * 0.07))
    max_val = float(np.max(score))
    if max_val < 1e-4:
        return (0.50, 0.34 if is_portrait else 0.52, role)

    top = np.where(score >= max_val * 0.80, score, 0.0)
    total = float(np.sum(top))
    if total < 1e-6:
        return (0.50, 0.34 if is_portrait else 0.52, role)

    cx = float(np.sum(top * xn) / total)
    cy = float(np.sum(top * yn) / total)
    return (float(np.clip(cx, 0.22, 0.78)), float(np.clip(cy, 0.18, 0.74)), role)


def _detect_subject_center_fast(rgb_u8: np.ndarray) -> Tuple[float, float]:
    cx, cy, _ = _detect_subject_center_and_role(rgb_u8)
    return (cx, cy)


def _deterministic_seed_from_image(rgb_u8: np.ndarray) -> int:
    """Derive a stable per-photo seed so each photo gets its own natural shake curve deterministically."""
    thumb = cv2.resize(rgb_u8, (16, 16), interpolation=cv2.INTER_AREA).tobytes()
    digest = hashlib.blake2b(thumb, digest_size=4).digest()
    return int.from_bytes(digest, "little") % 90000 + 100


def analyze_and_build_auto_config(
    rgb_u8: np.ndarray,
    strength: float = 1.0,
    auto_tilt: bool = False,
    auto_flash_cast: bool = True,
) -> AutoSceneAnalysis:
    """Analyze any Instagram/social photo (portrait OR non-face filler) and return a calibrated PipelineConfig."""
    strength = float(np.clip(strength, 0.25, 1.65))
    h, w = rgb_u8.shape[:2]

    # Measure on normalized ~720px working copy so sharpness metrics are resolution-invariant
    scale_m = 720.0 / max(h, w)
    mw, mh = max(64, int(round(w * scale_m))), max(64, int(round(h * scale_m)))
    work = cv2.resize(rgb_u8, (mw, mh), interpolation=cv2.INTER_AREA) if scale_m < 0.98 else rgb_u8
    mh, mw = work.shape[:2]

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV).astype(np.float32)

    p5, p25, p50, p85, p95, p99 = [float(v) for v in np.percentile(gray, [5, 25, 50, 85, 95, 99])]
    dark_ratio = float(np.mean(gray < 48.0))
    bright_ratio = float(np.mean(gray > 215.0))
    clipped_white_ratio = float(np.mean(gray > 242.0))
    mean_luma = float(np.mean(gray))
    warmth = float(np.mean(work[..., 0].astype(np.float32) - work[..., 2].astype(np.float32)))
    mean_sat = float(np.mean(hsv[..., 1]))

    # Detect subject/object center & whether this shot has a portrait face or is a non-face object shot
    fx, fy, scene_role = _detect_subject_center_and_role(rgb_u8)
    sx, sy = int(fx * mw), int(fy * mh)
    rx, ry = max(24, int(mw * 0.18)), max(24, int(mh * 0.18))
    subj_roi = gray[max(0, sy - ry) : min(mh, sy + ry), max(0, sx - rx) : min(mw, sx + rx)]
    subj_lap = float(cv2.Laplacian(subj_roi, cv2.CV_32F).var()) if subj_roi.size > 0 else 150.0
    global_lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    eff_sharp = max(subj_lap, global_lap * 0.75)

    # Center vs border brightness contrast (direct on-camera flash makes foreground brighter than dark background)
    subj_luma = float(np.mean(subj_roi)) if subj_roi.size > 0 else mean_luma
    border_mask = np.ones((mh, mw), dtype=bool)
    border_mask[int(mh * 0.18) : int(mh * 0.82), int(mw * 0.18) : int(mw * 0.82)] = False
    border_luma = float(np.mean(gray[border_mask]))
    flash_contrast = subj_luma - border_luma

    # Existing-blur guard: if the original photo is ALREADY soft/blurry (eff_sharp < 95),
    # dial down blur smoothly so Auto-Match NEVER turns a soft candid into unreadable mush!
    if eff_sharp < 45.0:
        sharpness_scale = 0.42
    elif eff_sharp < 110.0:
        sharpness_scale = 0.42 + 0.58 * ((eff_sharp - 45.0) / 65.0)
    elif eff_sharp > 600.0:
        sharpness_scale = 1.06
    else:
        sharpness_scale = 1.0

    # Existing-bloom guard: if image already has massive blown white glare (>10% clipped white on dark scene),
    # restrain Stage 3 glow so it doesn't blow out further (e.g. reflective flash jackets!).
    already_bloomy = clipped_white_ratio > 0.08 and dark_ratio > 0.35
    glow_guard = 0.35 if already_bloomy else 1.0

    # Natural deterministic seed & slight diagonal angle tailored per photo
    seed = _deterministic_seed_from_image(rgb_u8)
    angle_choices = [16.0, -22.0, 28.0, -14.0, 34.0, -28.0]
    nat_angle = angle_choices[seed % len(angle_choices)]

    s = strength * sharpness_scale
    metrics = {
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "warmth": warmth,
        "mean_sat": mean_sat,
        "flash_contrast": flash_contrast,
        "subj_lap": subj_lap,
        "global_lap": global_lap,
    }

    # ---------- Classify into 1 of 5 Real Social-Media Photo Archetypes ----------
    # Bright sunlit outdoor / high-key check FIRST so sunny photos with warm skin/concrete NEVER get labeled Warm Night!
    is_bright_outdoor = (
        p50 >= 122.0
        or (bright_ratio > 0.20 and dark_ratio < 0.22)
        or (p95 > 208.0 and p50 > 82.0 and dark_ratio < 0.32 and bright_ratio > 0.08)
    )

    if dark_ratio > 0.45 and p50 < 62.0 and not (p85 > 195.0 and bright_ratio > 0.14):
        if flash_contrast > 18.0 or p95 > 175.0:
            # 1. Direct Flash Night Party (dark room, bright subject / speculars)
            archetype = "flash_party"
            label = "Night Flash Party"
            summary = "Dark room with bright flash highlights — keeping face sharp with subtle slow-sync ghosting, outer arm trail, soft flash glow & night grain."
            cfg = PipelineConfig(
                name="Auto — Night Flash Party",
                description=summary,
                color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.85, highlight_boost=0.85 * glow_guard),
                stage1_focus=Stage1FocusConfig(
                    enabled=True,
                    radius=1.10 * s,
                    falloff_mode="face_center",
                    falloff_strength=0.80,
                    focus_center_x=fx,
                    focus_center_y=fy,
                ),
                stage2_motion=Stage2MotionConfig(
                    enabled=True,
                    method="flash_drag",
                    shake_amount=9.0 * s,
                    ghost_strength=0.38,
                    curvature=0.66,
                    direction_angle=nat_angle,
                    flash_sharpness=0.78,
                    outer_smear=0.56 * min(1.15, strength),
                    ambient_warmth=0.14,
                    seed=seed,
                ),
                stage3_optics=Stage3OpticsConfig(
                    enabled=True,
                    glow_amount=0.30 * strength * glow_guard,
                    haze_lift=0.18 * strength * glow_guard,
                    warmth=0.18 if warmth > 12 else 0.08,
                    edge_fringing=0.16 * strength,
                ),
                stage4_grain=Stage4GrainConfig(
                    enabled=True,
                    grain_amount=0.32 * strength,
                    clumpiness=0.58,
                    isp_smoothing=0.34,
                    shadow_weight=0.80,
                    seed=seed + 17,
                ),
                stage5_compression=Stage5CompressionConfig(
                    enabled=True,
                    insta_softness=0.16 * min(1.2, strength),
                    jpeg_quality=84,
                ),
            )
        else:
            # 2. Dark Moody Ambient Night (dim low-light bedroom / hallway / club without harsh flash)
            archetype = "dark_night_moody"
            label = "Moody Night Swirl"
            summary = "Dim ambient night scene — crisp eyes anchored with soft low-light focus rolloff, subtle camera swirl & Night Mode grain."
            cfg = PipelineConfig(
                name="Auto — Moody Night Swirl",
                description=summary,
                color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.84, highlight_boost=0.90 * glow_guard),
                stage1_focus=Stage1FocusConfig(
                    enabled=True,
                    radius=1.55 * s,
                    falloff_mode="face_center",
                    falloff_strength=0.80,
                    focus_center_x=fx,
                    focus_center_y=fy,
                ),
                stage2_motion=Stage2MotionConfig(
                    enabled=True,
                    method="camera_swirl",
                    shake_amount=7.2 * s,
                    ghost_strength=0.32,
                    curvature=0.64,
                    direction_angle=nat_angle,
                    flash_sharpness=0.74,
                    outer_smear=0.46,
                    ambient_warmth=0.14,
                    swing_angle=4.0 * min(1.25, strength),
                    seed=seed,
                ),
                stage3_optics=Stage3OpticsConfig(
                    enabled=True,
                    glow_amount=0.28 * strength * glow_guard,
                    haze_lift=0.12 * strength * glow_guard,
                    warmth=0.18 if warmth > 10 else 0.08,
                    edge_fringing=0.16 * strength,
                ),
                stage4_grain=Stage4GrainConfig(
                    enabled=True,
                    grain_amount=0.34 * strength,
                    clumpiness=0.64,
                    isp_smoothing=0.38,
                    shadow_weight=0.82,
                    seed=seed + 17,
                ),
                stage5_compression=Stage5CompressionConfig(
                    enabled=True,
                    insta_softness=0.16 * min(1.2, strength),
                    jpeg_quality=84,
                ),
            )
    elif not is_bright_outdoor and warmth > 22.0 and p50 < 118.0 and p85 < 188.0:
        # 3. Warm Night / Bar / Sunset / Golden Backlit Candid
        archetype = "warm_night"
        label = "Warm Evening Focus"
        summary = "Warm evening or golden lighting — dreamy circular missed-focus falloff locked on the subject, subtle curved hand tremor, golden highlight bloom & velvety texture."
        cfg = PipelineConfig(
            name="Auto — Warm Evening Focus",
            description=summary,
            color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.84, highlight_boost=1.00 * glow_guard),
            stage1_focus=Stage1FocusConfig(
                enabled=True,
                radius=3.2 * s,
                falloff_mode="face_center",
                falloff_strength=0.74,
                focus_center_x=fx,
                focus_center_y=fy,
            ),
            stage2_motion=Stage2MotionConfig(
                enabled=True,
                method="handheld_shake",
                shake_amount=4.8 * s,
                ghost_strength=0.26,
                curvature=0.60,
                direction_angle=nat_angle,
                flash_sharpness=0.75,
                outer_smear=0.38,
                ambient_warmth=0.22,
                seed=seed,
            ),
            stage3_optics=Stage3OpticsConfig(
                enabled=True,
                glow_amount=0.36 * strength * glow_guard,
                haze_lift=0.10 * strength,
                warmth=0.36,
                edge_fringing=0.16 * strength,
            ),
            stage4_grain=Stage4GrainConfig(
                enabled=True,
                grain_amount=0.24 * strength,
                clumpiness=0.62,
                isp_smoothing=0.36,
                shadow_weight=0.76,
                seed=seed + 17,
            ),
            stage5_compression=Stage5CompressionConfig(
                enabled=True,
                insta_softness=0.18 * min(1.2, strength),
                jpeg_quality=83,
            ),
        )
    elif is_bright_outdoor:
        # 4. Bright Daylight / High-Key Outdoor / Modern Transit / Flash Daylight
        archetype = "daylight_street"
        label = "Daylight Candid"
        summary = "Bright daylight or high-key lighting — rich blacks preserved (zero muddy haze), spontaneous outer candid motion keeping the face crisp & subtle sunlit lens sheen."
        cfg = PipelineConfig(
            name="Auto — Daylight Candid",
            description=summary,
            color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.88, highlight_boost=0.52),
            stage1_focus=Stage1FocusConfig(
                enabled=True,
                radius=1.30 * s,
                falloff_mode="face_center",
                falloff_strength=0.78,
                focus_center_x=fx,
                focus_center_y=fy,
            ),
            stage2_motion=Stage2MotionConfig(
                enabled=True,
                method="flash_drag",
                shake_amount=6.0 * s,
                ghost_strength=0.24,
                curvature=0.58,
                direction_angle=nat_angle,
                flash_sharpness=0.80,
                outer_smear=0.44 * min(1.15, strength),
                ambient_warmth=0.05,
                seed=seed,
            ),
            stage3_optics=Stage3OpticsConfig(
                enabled=True,
                glow_amount=0.20 * strength,
                haze_lift=0.02 * strength,
                warmth=0.10 if warmth > 12 else 0.04,
                edge_fringing=0.14 * strength,
            ),
            stage4_grain=Stage4GrainConfig(
                enabled=True,
                grain_amount=0.13 * strength,
                clumpiness=0.45,
                isp_smoothing=0.20,
                shadow_weight=0.70,
                seed=seed + 17,
            ),
            stage5_compression=Stage5CompressionConfig(
                enabled=True,
                insta_softness=0.16 * min(1.2, strength),
                jpeg_quality=85,
            ),
        )
    else:
        # 5. Everyday Indoor / Gym / Cafe / Mirror Casual Snapshot
        archetype = "indoor_candid"
        label = "Mirror / Indoor Snap"
        summary = "Everyday indoor or mirror selfie lighting — authentic curved thumb-press handheld micro-shake, subtle optical softness, clean mirror sheen & realistic IG polish."
        cfg = PipelineConfig(
            name="Auto — Mirror / Indoor Snap",
            description=summary,
            color=OpticalColorConfig(gamma=2.2, highlight_threshold=0.86, highlight_boost=0.68),
            stage1_focus=Stage1FocusConfig(
                enabled=True,
                radius=1.45 * s,
                falloff_mode="face_center",
                falloff_strength=0.62,
                focus_center_x=fx,
                focus_center_y=fy,
            ),
            stage2_motion=Stage2MotionConfig(
                enabled=True,
                method="handheld_shake",
                shake_amount=5.6 * s,
                ghost_strength=0.34,
                curvature=0.62,
                direction_angle=nat_angle,
                flash_sharpness=0.76,
                outer_smear=0.40,
                ambient_warmth=0.07,
                seed=seed,
            ),
            stage3_optics=Stage3OpticsConfig(
                enabled=True,
                glow_amount=0.24 * strength,
                haze_lift=0.06 * strength,
                warmth=0.14 if warmth > 14 else 0.06,
                edge_fringing=0.15 * strength,
            ),
            stage4_grain=Stage4GrainConfig(
                enabled=True,
                grain_amount=0.20 * strength,
                clumpiness=0.52,
                isp_smoothing=0.28,
                shadow_weight=0.72,
                seed=seed + 17,
            ),
            stage5_compression=Stage5CompressionConfig(
                enabled=True,
                insta_softness=0.20 * min(1.2, strength),
                jpeg_quality=83,
            ),
        )

    # Apply whisper-subtle Flash Falloff + Dirty Bar Shadow Cast ONLY on dark/night scenes (never on daylight/mirror)
    if auto_flash_cast:
        if archetype == "flash_party":
            cfg.stage3_optics.flash_falloff = round(0.22 * min(1.25, strength) * glow_guard, 2)
            cfg.stage3_optics.bar_shadow_cast = round(0.18 * min(1.25, strength), 2)
        elif archetype == "dark_night_moody":
            cfg.stage3_optics.flash_falloff = round(0.12 * min(1.25, strength) * glow_guard, 2)
            cfg.stage3_optics.bar_shadow_cast = round(0.15 * min(1.25, strength), 2)
        elif archetype == "warm_night":
            cfg.stage3_optics.flash_falloff = round(0.10 * min(1.25, strength) * glow_guard, 2)
            cfg.stage3_optics.bar_shadow_cast = round(0.08 * min(1.25, strength), 2)

    # Apply very subtle accidental phone tilt & off-center crop when enabled
    if auto_tilt:
        tilt_choices = [-1.2, 1.0, -0.9, 1.3, -1.1, 0.8]
        cfg.stage5_compression.tilt_angle = tilt_choices[seed % len(tilt_choices)]
        cfg.stage5_compression.crop_offset = 0.14 if archetype != "indoor_candid" else 0.10
        cfg.stage5_compression.framing_seed = seed

    # Non-face filler photos (table, drinks, food, shoes, street) get cleaner object-centered focus
    if scene_role == "non_face_object":
        cfg.stage1_focus.falloff_mode = "edges"
        cfg.stage1_focus.falloff_strength = min(cfg.stage1_focus.falloff_strength, 0.58)
        summary = f"No-face scene ({label}) — anchored focus on foreground hero objects with natural phone flash, grain & framing."

    return AutoSceneAnalysis(
        archetype=archetype,
        archetype_label=label,
        plain_summary=summary,
        subject_center=(fx, fy),
        existing_sharpness=eff_sharp,
        sharpness_scale=sharpness_scale,
        metrics=metrics,
        config=cfg.validated(),
        scene_role=scene_role,
    )


def _smooth_1d_hist(vals: np.ndarray, bins: int, rng: Tuple[float, float]) -> np.ndarray:
    h, _ = np.histogram(vals, bins=bins, range=rng)
    h = cv2.GaussianBlur(h.astype(np.float32).reshape(1, -1), (5, 1), 1.0).flatten()
    return h / (float(np.linalg.norm(h)) + 1e-6)


def _scene_fingerprint(rgb_u8: np.ndarray) -> np.ndarray:
    """Shift-invariant lighting + 2D chroma + coarse Lab layout fingerprint to cluster same-photoshoot slides."""
    small = cv2.resize(rgb_u8, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Outer border ring captures background room/pillow/hallway invariant to subject head/pose shift
    mask = np.ones((128, 128), dtype=bool)
    mask[26:102, 26:102] = False
    b_lab = lab[mask]

    # 2D blurred Lab chroma histogram (a, b)
    ab_full = cv2.calcHist([lab.astype(np.uint8)], [1, 2], None, [10, 10], [88, 168, 88, 168]).astype(np.float32)
    ab_full = cv2.GaussianBlur(ab_full, (3, 3), 0.9).flatten()
    ab_full /= (float(np.linalg.norm(ab_full)) + 1e-6)

    l_full = _smooth_1d_hist(lab[..., 0], 14, (0.0, 256.0))
    l_bord = _smooth_1d_hist(b_lab[:, 0], 14, (0.0, 256.0))
    s_full = _smooth_1d_hist(hsv[..., 1], 12, (0.0, 256.0))

    # Zero-centered coarse 4x4 Lab layout signature (cleanly separates dark bed vs hallway vs wood table)
    grid = cv2.resize(lab, (4, 4), interpolation=cv2.INTER_AREA)
    grid[..., 0] = (grid[..., 0] - 128.0) / 128.0
    grid[..., 1] = (grid[..., 1] - 128.0) / 48.0
    grid[..., 2] = (grid[..., 2] - 128.0) / 48.0
    gvec = grid.flatten().astype(np.float32)
    gvec /= (float(np.linalg.norm(gvec)) + 1e-6)

    vec = np.concatenate([
        ab_full * 0.42,
        l_full * 0.28,
        l_bord * 0.34,
        s_full * 0.22,
        gvec * 0.48,
    ]).astype(np.float32)
    vec /= (float(np.linalg.norm(vec)) + 1e-6)
    return vec


def cluster_carousel_scenes(images_rgb: List[np.ndarray], similarity_threshold: float = 0.88) -> List[int]:
    """Group slides from the same location/photoshoot/lighting into matching scene_group indices (0, 1, 2...).

    Slides in the same scene_group MUST share the same camera look, lighting, flash falloff & shadow cast,
    while slides from genuinely different settings/moments can naturally shift looks.
    """
    n = len(images_rgb)
    if n == 0:
        return []
    fps = [_scene_fingerprint(img) for img in images_rgb]
    group_ids: List[int] = [-1] * n
    group_reps: List[np.ndarray] = []

    for i, fp in enumerate(fps):
        best_g = -1
        best_sim = -1.0
        for g_idx, rep in enumerate(group_reps):
            sim = float(np.dot(fp, rep))
            if sim > best_sim:
                best_sim = sim
                best_g = g_idx
        if best_g != -1 and best_sim >= similarity_threshold:
            group_ids[i] = best_g
            # Update running centroid smoothly
            merged = group_reps[best_g] + fp
            group_reps[best_g] = merged / (float(np.linalg.norm(merged)) + 1e-6)
        else:
            group_ids[i] = len(group_reps)
            group_reps.append(fp.copy())

    return group_ids


def build_carousel_plan_local(
    images_rgb: List[np.ndarray],
    carousel_vibe: str = "one_night_out",
    auto_tilt: bool = True,
    auto_flash_cast: bool = True,
    vary_per_slide: bool = True,
    seed_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Build a cohesive per-slide recipe list where same-photoshoot slides share one consistent look."""
    from .presets import get_builtin_preset

    group_ids = cluster_carousel_scenes(images_rgb)
    analyses = [
        analyze_and_build_auto_config(
            img,
            strength=1.0,
            auto_tilt=auto_tilt,
            auto_flash_cast=auto_flash_cast,
        )
        for img in images_rgb
    ]

    # Pick ONE canonical camera/lighting recipe per scene_group so matching photos look shot on the same camera!
    group_base_cfg: Dict[int, PipelineConfig] = {}
    group_label: Dict[int, str] = {}
    group_arch_label: Dict[int, str] = {}

    for idx, (gid, an) in enumerate(zip(group_ids, analyses)):
        if gid in group_base_cfg:
            continue

        cfg = PipelineConfig.from_dict(an.config.to_dict())
        arch_lbl = an.archetype_label

        # Gentle carousel-vibe shaping (only when compatible with the actual photo lighting!)
        if carousel_vibe == "3am_dump" and an.archetype in ("flash_party", "dark_night_moody", "warm_night"):
            p = get_builtin_preset("3am_night_zoom")
            cfg = PipelineConfig.from_dict(p.to_dict())
            arch_lbl = "3AM Night Zoom"
            if auto_flash_cast:
                cfg.stage3_optics.flash_falloff = 0.28
                cfg.stage3_optics.bar_shadow_cast = 0.30
        elif carousel_vibe == "daylight_casual" and an.archetype in ("daylight_street", "indoor_candid"):
            cfg.stage3_optics.flash_falloff = 0.0
            cfg.stage3_optics.bar_shadow_cast = 0.0
            cfg.stage3_optics.haze_lift = min(cfg.stage3_optics.haze_lift, 0.05)

        group_base_cfg[gid] = cfg.validated()
        group_arch_label[gid] = arch_lbl
        group_label[gid] = f"Scene {gid + 1} · {arch_lbl}"

    # Assign cohesive configs per slide:
    # - Same scene_group => IDENTICAL base look, flash_falloff, bar_shadow_cast, glow, grain!
    # - Tiny natural per-shot variation ONLY in shake seed & small organic tilt delta (±0.5°) so it looks like
    #   two sequential photos taken 10 seconds apart on the same iPhone — NOT two totally different filters!
    seen_in_group: Dict[int, int] = {}
    plan: List[Dict[str, Any]] = []

    tilt_pool = [-1.2, 1.0, -0.9, 1.3, -1.1, 0.8]
    for idx, (gid, an) in enumerate(zip(group_ids, analyses)):
        occ = seen_in_group.get(gid, 0)
        seen_in_group[gid] = occ + 1

        cfg = PipelineConfig.from_dict(group_base_cfg[gid].to_dict())
        # Keep each individual slide's own detected focal anchor & framing
        cfg.stage1_focus.focus_center_x = an.subject_center[0]
        cfg.stage1_focus.focus_center_y = an.subject_center[1]
        if an.scene_role == "non_face_object":
            cfg.stage1_focus.falloff_mode = "edges"

        slide_seed = int(an.config.stage2_motion.seed + seed_offset + idx * 71)
        cfg.stage2_motion.seed = slide_seed
        cfg.stage4_grain.seed = slide_seed + 19
        cfg.stage5_compression.framing_seed = slide_seed + 37

        if auto_tilt:
            base_tilt = tilt_pool[(gid + seed_offset) % len(tilt_pool)]
            # Sequential shots in the same photoshoot get a tiny natural wrist shift (±0.25°), never crooked
            jitter = ((occ % 3) - 1) * 0.25 if vary_per_slide else 0.0
            cfg.stage5_compression.tilt_angle = float(np.clip(base_tilt + jitter, -1.8, 1.8))
            cfg.stage5_compression.crop_offset = 0.14 if an.archetype != "indoor_candid" else 0.10
        else:
            cfg.stage5_compression.tilt_angle = 0.0
            cfg.stage5_compression.crop_offset = 0.0

        if not auto_flash_cast:
            cfg.stage3_optics.flash_falloff = 0.0
            cfg.stage3_optics.bar_shadow_cast = 0.0

        plan.append({
            "index": idx,
            "scene_group": f"scene_{gid + 1}",
            "scene_group_label": group_label[gid],
            "scene_role": an.scene_role,
            "archetype_label": group_arch_label[gid],
            "summary": an.plain_summary,
            "config": cfg.validated(),
        })

    return plan
