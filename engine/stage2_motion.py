"""Stage 2: Realistic Motion Blur, Slow-Sync Flash Drag & Pluggable Cloud AI Hook.

Provides a clean `MotionProvider` interface with 3 fast CPU local methods plus an
optional Cloud AI provider (fal.ai / Replicate) that stays dormant unless an API key
is explicitly configured.

Methods:
  - Method A (`handheld_shake`): Non-linear 2D curved hand-shake random-walk PSF with
    subtle dwell-time spikes at the trajectory ends -> authentic tiny double-image ghosting.
  - Method B (`flash_drag`): Physically blends a sharp "flash burst" exposure (~65-75%)
    with a curved ambient shutter-drag exposure (~25-35%) + localized outer-limb / edge
    motion smear keeping the face/center frozen.
  - Method C (`pivot_swing`): Localized rotational / arc swing blur around moving arms &
    hands via vectorized remap while keeping the face sharp.
  - Optional Cloud (`cloud_ai`): Clean hook for fal.ai / Replicate REST APIs.
"""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .color import linear_to_srgb_u8, srgb_u8_to_linear
from .masks import UserBrushMask, build_outer_limb_mask
from .schema import CloudAIConfig, OpticalColorConfig, Stage2MotionConfig
from .stage1_focus import convolve_linear_rgb


@dataclass
class MotionResult:
    """Output of Stage 2 motion provider, including optional status note."""

    linear_rgb: np.ndarray
    provider_used: str
    note: str = ""


def generate_curved_shake_psf(
    span_px: float,
    curvature: float = 0.60,
    ghost_strength: float = 0.35,
    direction_angle_deg: float = 25.0,
    seed: int = 42,
    num_samples: int = 240,
) -> np.ndarray:
    """Generate a realistic non-linear 2D curved hand-shake trajectory PSF with dwell spikes.

    Real smartphone shutter blur is never a straight Photoshop line: the thumb press
    causes a curved wrist arc with brief dwell pauses near the start and end of the
    exposure, creating subtle organic double-image ghosting on high-contrast edges.
    """
    span = max(1.0, float(span_px))
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)

    # Base non-linear trajectory: smooth low-frequency random walk + wrist arc bend
    # Start near (0,0), curve smoothly along primary axis with orthogonal S/C bend
    bend_sign = 1.0 if rng.random() > 0.5 else -1.0
    c_bend = curvature * (0.28 + 0.32 * rng.random()) * bend_sign
    s_bend = curvature * (0.14 + 0.20 * rng.random()) * (1.0 if rng.random() > 0.5 else -1.0)

    # Smooth low-frequency micro-jitter harmonics (3..6 cycles) for organic hand tremor
    phases = rng.uniform(0.0, 2.0 * np.pi, size=4).astype(np.float32)
    amps = (rng.uniform(0.02, 0.07, size=4) * curvature).astype(np.float32)

    # Smooth eased longitudinal progression (slower at ends -> natural dwell!)
    ease = 0.5 - 0.5 * np.cos(np.pi * t)
    u = (1.0 - 0.55 * ghost_strength) * t + (0.55 * ghost_strength) * ease

    v = (
        c_bend * np.sin(np.pi * t)
        + s_bend * np.sin(2.0 * np.pi * t)
        + amps[0] * np.sin(3.0 * np.pi * t + phases[0])
        + amps[1] * np.cos(4.0 * np.pi * t + phases[1])
    )
    u = u + amps[2] * np.sin(3.0 * np.pi * t + phases[2]) + amps[3] * np.cos(5.0 * np.pi * t + phases[3])

    # Normalize total trajectory span to exact requested pixel length
    u -= u.mean()
    v -= v.mean()
    pts = np.stack([u, v], axis=1)
    max_ext = float(np.max(np.linalg.norm(pts - pts[0], axis=1)))
    if max_ext < 1e-5:
        max_ext = 1.0
    pts = pts * (span / max_ext)

    # Rotate to requested dominant drift angle (with slight natural jitter)
    rad = np.deg2rad(direction_angle_deg + float(rng.uniform(-6.0, 6.0)))
    ca, sa = float(np.cos(rad)), float(np.sin(rad))
    xs = pts[:, 0] * ca - pts[:, 1] * sa
    ys = pts[:, 0] * sa + pts[:, 1] * ca

    # Center coordinates around kernel origin
    xs -= xs.mean()
    ys -= ys.mean()

    # Dwell-time weighting along the curve: higher weight at start/end endpoints
    # creates authentic subtle double-image ghosting on bright edges/highlights.
    end_spike = np.exp(-((t / 0.09) ** 2)) + 0.85 * np.exp(-(((1.0 - t) / 0.11) ** 2))
    weights = 1.0 + (ghost_strength * 4.5) * end_spike

    # Allocate compact odd-sized PSF kernel
    half_x = int(np.ceil(np.max(np.abs(xs)))) + 2
    half_y = int(np.ceil(np.max(np.abs(ys)))) + 2
    half = max(half_x, half_y, 2)
    ksize = 2 * half + 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)

    # Sub-pixel bilinear splatting onto kernel grid
    gx = xs + half
    gy = ys + half
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)
    fx = gx - ix
    fy = gy - iy

    np.add.at(kernel, (iy, ix), weights * (1.0 - fx) * (1.0 - fy))
    np.add.at(kernel, (iy, ix + 1), weights * fx * (1.0 - fy))
    np.add.at(kernel, (iy + 1, ix), weights * (1.0 - fx) * fy)
    np.add.at(kernel, (iy + 1, ix + 1), weights * fx * fy)

    # Tiny optical anti-alias smoothing so PSF has no harsh single-pixel stairsteps
    kernel = cv2.GaussianBlur(kernel, (3, 3), 0.55)
    s = float(kernel.sum())
    if s > 1e-8:
        kernel /= s
    else:
        kernel[half, half] = 1.0
    return kernel


class MotionProvider(ABC):
    """Pluggable Stage 2 motion interface (local CPU methods + optional Cloud AI)."""

    name: str = "base"

    @abstractmethod
    def apply(
        self,
        linear_rgb: np.ndarray,
        cfg: Stage2MotionConfig,
        brush: Optional[UserBrushMask] = None,
        color_cfg: Optional[OpticalColorConfig] = None,
        cloud_cfg: Optional[CloudAIConfig] = None,
    ) -> MotionResult:
        raise NotImplementedError


class LocalHandheldShakeProvider(MotionProvider):
    """Method A: Handheld Micro-Shake (curved 2D trajectory + endpoint dwell ghosting)."""

    name = "handheld_shake"

    def apply(
        self,
        linear_rgb: np.ndarray,
        cfg: Stage2MotionConfig,
        brush: Optional[UserBrushMask] = None,
        color_cfg: Optional[OpticalColorConfig] = None,
        cloud_cfg: Optional[CloudAIConfig] = None,
    ) -> MotionResult:
        if cfg.shake_amount <= 0.25:
            return MotionResult(linear_rgb=linear_rgb.copy(), provider_used=self.name)

        h, w = linear_rgb.shape[:2]
        scale = min(h, w) / 1080.0
        span_px = max(1.2, cfg.shake_amount * scale)

        psf = generate_curved_shake_psf(
            span_px=span_px,
            curvature=cfg.curvature,
            ghost_strength=cfg.ghost_strength,
            direction_angle_deg=cfg.direction_angle,
            seed=cfg.seed,
        )
        blurred = convolve_linear_rgb(linear_rgb, psf)

        # Respect user Keep Sharp / Make Blurry brush strokes if present
        if brush is not None and (brush.has_sharp or brush.has_blur):
            weight = np.ones((h, w), dtype=np.float32)
            if brush.has_sharp:
                weight = weight * (1.0 - brush.keep_sharp)
            if brush.has_blur:
                # Extra stronger curved shake specifically inside painted blur zone
                psf_extra = generate_curved_shake_psf(
                    span_px=span_px * 1.75,
                    curvature=min(1.0, cfg.curvature + 0.15),
                    ghost_strength=cfg.ghost_strength,
                    direction_angle_deg=cfg.direction_angle + 12.0,
                    seed=cfg.seed + 7,
                )
                extra_img = convolve_linear_rgb(linear_rgb, psf_extra)
                eb = brush.extra_blur[..., np.newaxis]
                blurred = blurred * (1.0 - eb) + extra_img * eb
            wm = weight[..., np.newaxis]
            out = linear_rgb * (1.0 - wm) + blurred * wm
            return MotionResult(linear_rgb=out.astype(np.float32), provider_used=self.name)

        return MotionResult(linear_rgb=blurred.astype(np.float32), provider_used=self.name)


class LocalFlashDragProvider(MotionProvider):
    """Method B: Slow-Sync Flash Drag.

    Physically blends a crisp flash-burst exposure (~65-75%) with a curved ambient
    shutter-drag exposure (~25-35%) in Linear RGB, plus optional localized outer-limb /
    edge motion smear keeping the face/center frozen while hands and edges trail.
    """

    name = "flash_drag"

    def apply(
        self,
        linear_rgb: np.ndarray,
        cfg: Stage2MotionConfig,
        brush: Optional[UserBrushMask] = None,
        color_cfg: Optional[OpticalColorConfig] = None,
        cloud_cfg: Optional[CloudAIConfig] = None,
    ) -> MotionResult:
        if cfg.shake_amount <= 0.25 and cfg.outer_smear <= 0.02:
            return MotionResult(linear_rgb=linear_rgb.copy(), provider_used=self.name)

        h, w = linear_rgb.shape[:2]
        scale = min(h, w) / 1080.0
        span_px = max(2.0, cfg.shake_amount * scale)

        # 1. Flash burst exposure: mostly crisp, with microscopic 0.8px optical softness
        flash_exposure = linear_rgb

        # 2. Curved ambient shutter-drag exposure (trailing behind the flash burst)
        psf_ambient = generate_curved_shake_psf(
            span_px=span_px * 1.35,
            curvature=max(0.45, cfg.curvature),
            ghost_strength=min(1.0, cfg.ghost_strength + 0.20),
            direction_angle_deg=cfg.direction_angle,
            seed=cfg.seed,
        )
        # Shift ambient drag slightly off-center like rear-curtain flash trail
        ambient_exposure = convolve_linear_rgb(linear_rgb, psf_ambient)
        rad = np.deg2rad(cfg.direction_angle)
        dx = float(np.cos(rad) * span_px * 0.22)
        dy = float(np.sin(rad) * span_px * 0.22)
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        ambient_exposure = cv2.warpAffine(
            ambient_exposure, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

        # Subtle warm tungsten/club ambient color shift on the dragged ambient exposure
        if cfg.ambient_warmth > 1e-3:
            w_t = cfg.ambient_warmth
            tint = np.array([1.0 + 0.28 * w_t, 1.0 + 0.05 * w_t, 1.0 - 0.22 * w_t], dtype=np.float32)
            ambient_exposure = ambient_exposure * tint[np.newaxis, np.newaxis, :]

        # Physical linear-light blend: flash_sharpness * I_flash + (1 - flash_sharpness) * I_ambient
        fw = float(cfg.flash_sharpness)
        blended = fw * flash_exposure + (1.0 - fw) * ambient_exposure

        # 3. Localized outer-limb / edge motion smear (keeps face frozen, arms/hands trail!)
        if cfg.outer_smear > 0.02:
            outer_mask = build_outer_limb_mask((h, w), brush=brush)
            psf_limb = generate_curved_shake_psf(
                span_px=max(4.0, span_px * (1.6 + 1.1 * cfg.outer_smear)),
                curvature=min(1.0, cfg.curvature + 0.20),
                ghost_strength=min(1.0, cfg.ghost_strength * 0.75),
                direction_angle_deg=cfg.direction_angle + 22.0,
                seed=cfg.seed + 19,
            )
            limb_smeared = convolve_linear_rgb(linear_rgb, psf_limb)
            # Blend flash-drag base with heavier outer-limb smear where arms/edges live
            limb_w = (outer_mask * cfg.outer_smear * 0.82)[..., np.newaxis]
            blended = blended * (1.0 - limb_w) + limb_smeared * limb_w

        # Respect explicit "Keep Sharp" brush strokes on top
        if brush is not None and brush.has_sharp:
            ks = brush.keep_sharp[..., np.newaxis]
            # Preserve a tiny 10% ambient glow so the transition never looks pasted-on
            blended = blended * (1.0 - ks * 0.90) + flash_exposure * (ks * 0.90)

        return MotionResult(linear_rgb=np.maximum(blended, 0.0).astype(np.float32), provider_used=self.name)


class LocalPivotSwingProvider(MotionProvider):
    """Method C: Radial / Pivot Arm Swing.

    Creates authentic rotational/arc motion trails around moving arms and hands
    (pivoting around shoulder/elbow centers or user-painted blur zones) while keeping
    the face and center completely sharp.
    """

    name = "pivot_swing"

    @staticmethod
    def _rotational_arc_blur(
        linear_rgb: np.ndarray,
        pivot_xy: Tuple[float, float],
        max_angle_deg: float,
        steps: int = 15,
        ghost_strength: float = 0.30,
    ) -> np.ndarray:
        """Fast multi-sample rotational arc accumulation in Linear RGB."""
        h, w = linear_rgb.shape[:2]
        px, py = pivot_xy

        angles = np.linspace(-max_angle_deg * 0.35, max_angle_deg * 0.65, steps, dtype=np.float32)
        t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        # Stronger anchor at flash/start instant + smooth fading arc trail
        weights = (1.0 - 0.55 * t) + ghost_strength * (
            1.4 * np.exp(-((t / 0.14) ** 2)) + 0.6 * np.exp(-(((1.0 - t) / 0.18) ** 2))
        )
        weights = (weights / weights.sum()).astype(np.float32)

        acc = np.zeros_like(linear_rgb, dtype=np.float32)
        for ang, wt in zip(angles, weights):
            if abs(float(ang)) < 0.05:
                acc += linear_rgb * wt
                continue
            M = cv2.getRotationMatrix2D((px, py), float(ang), 1.0)
            warped = cv2.warpAffine(
                linear_rgb,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            acc += warped * wt
        return acc

    def apply(
        self,
        linear_rgb: np.ndarray,
        cfg: Stage2MotionConfig,
        brush: Optional[UserBrushMask] = None,
        color_cfg: Optional[OpticalColorConfig] = None,
        cloud_cfg: Optional[CloudAIConfig] = None,
    ) -> MotionResult:
        if cfg.swing_intensity <= 0.02 or cfg.swing_angle <= 0.3:
            return MotionResult(linear_rgb=linear_rgb.copy(), provider_used=self.name)

        h, w = linear_rgb.shape[:2]

        # Priority 1: Exact AI-detected shoulder/elbow pivot points + per-limb polygons!
        if cfg.ai_pivots:
            out = linear_rgb.copy()
            sigma = max(4.0, min(h, w) * 0.025)
            ksize = max(5, min(int(sigma * 4) | 1, 151))
            for piv in cfg.ai_pivots:
                px = float(piv["x"]) * (w - 1)
                py = float(piv["y"]) * (h - 1)
                ang = float(piv.get("angle_deg", cfg.swing_angle))
                # Scale AI angle proportionally to user's Arm Swing Arc slider
                scale_ang = (cfg.swing_angle / 8.0)
                arc_img = self._rotational_arc_blur(
                    linear_rgb,
                    pivot_xy=(px, py),
                    max_angle_deg=ang * scale_ang,
                    steps=15,
                    ghost_strength=cfg.ghost_strength,
                )
                poly_pts = piv.get("polygon")
                if poly_pts and len(poly_pts) >= 3:
                    limb_mask = np.zeros((h, w), dtype=np.float32)
                    poly = np.array(
                        [[int(round(pt[0] * (w - 1))), int(round(pt[1] * (h - 1)))] for pt in poly_pts],
                        dtype=np.int32,
                    )
                    cv2.fillPoly(limb_mask, [poly], 1.0)
                    limb_mask = cv2.GaussianBlur(limb_mask, (ksize, ksize), sigma)
                    limb_mask = np.clip(limb_mask * 1.15, 0.0, 1.0)
                elif brush is not None and brush.has_blur:
                    limb_mask = brush.extra_blur.copy()
                else:
                    limb_mask = build_outer_limb_mask((h, w), brush=brush)

                if brush is not None and brush.has_sharp:
                    limb_mask = limb_mask * (1.0 - brush.keep_sharp)

                wm = (limb_mask * cfg.swing_intensity)[..., np.newaxis]
                out = out * (1.0 - wm) + arc_img * wm

            return MotionResult(linear_rgb=np.maximum(out, 0.0).astype(np.float32), provider_used=f"{self.name}:ai_pivots")

        # Left & right shoulder/elbow pivot points for natural outward arm-swing arcs
        left_pivot = (w * 0.36, h * 0.44)
        right_pivot = (w * 0.64, h * 0.44)

        # If user painted an explicit blur region, anchor pivot slightly inward toward body center
        if brush is not None and brush.has_blur:
            ys, xs = np.where(brush.extra_blur > 0.25)
            if len(xs) > 20:
                bx = float(np.mean(xs))
                by = float(np.mean(ys))
                # Pull pivot 35% toward body center (shoulder/elbow joint)
                piv_x = bx * 0.60 + (w * 0.50) * 0.40
                piv_y = by * 0.65 + (h * 0.42) * 0.35
                sign = 1.0 if bx < w * 0.5 else -1.0
                arc_img = self._rotational_arc_blur(
                    linear_rgb,
                    pivot_xy=(piv_x, piv_y),
                    max_angle_deg=cfg.swing_angle * sign,
                    steps=15,
                    ghost_strength=cfg.ghost_strength,
                )
                mask = brush.extra_blur.copy()
                if brush.has_sharp:
                    mask = mask * (1.0 - brush.keep_sharp)
                wm = (mask * cfg.swing_intensity)[..., np.newaxis]
                out = linear_rgb * (1.0 - wm) + arc_img * wm
                return MotionResult(linear_rgb=out.astype(np.float32), provider_used=self.name)

        # Automatic dual-arm arc blur (left arm swings counter-clockwise, right arm clockwise)
        arc_left = self._rotational_arc_blur(
            linear_rgb, left_pivot, max_angle_deg=cfg.swing_angle, steps=13, ghost_strength=cfg.ghost_strength
        )
        arc_right = self._rotational_arc_blur(
            linear_rgb, right_pivot, max_angle_deg=-cfg.swing_angle, steps=13, ghost_strength=cfg.ghost_strength
        )

        # Smooth left/right split so left side uses left-shoulder pivot, right side uses right-shoulder pivot
        x_ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :]
        right_w = np.clip((x_ramp - 0.38) / 0.24, 0.0, 1.0)
        right_w = (right_w * right_w * (3.0 - 2.0 * right_w))[..., np.newaxis]
        arc_combined = arc_left * (1.0 - right_w) + arc_right * right_w

        # Mask protects face & torso, applies arm-swing arc smoothly to outer arms/hands
        outer_mask = build_outer_limb_mask((h, w), brush=brush)
        wm = (outer_mask * cfg.swing_intensity)[..., np.newaxis]
        out = linear_rgb * (1.0 - wm) + arc_combined * wm

        # Add subtle whole-frame flash softness (tiny 20% hand micro-shake) so the transition feels organic
        if cfg.shake_amount > 1.0:
            scale = min(h, w) / 1080.0
            tiny_psf = generate_curved_shake_psf(
                span_px=max(1.2, cfg.shake_amount * 0.22 * scale),
                curvature=cfg.curvature,
                ghost_strength=cfg.ghost_strength * 0.4,
                direction_angle_deg=cfg.direction_angle,
                seed=cfg.seed,
            )
            micro = convolve_linear_rgb(out, tiny_psf)
            micro_w = 0.22
            if brush is not None and brush.has_sharp:
                ks = brush.keep_sharp[..., np.newaxis]
                out = out * (1.0 - micro_w * (1.0 - ks)) + micro * (micro_w * (1.0 - ks))
            else:
                out = out * (1.0 - micro_w) + micro * micro_w

        return MotionResult(linear_rgb=np.maximum(out, 0.0).astype(np.float32), provider_used=self.name)


class CloudAIMotionProvider(MotionProvider):
    """Optional Cloud AI motion provider (fal.ai / Replicate REST interface).

    Completely optional: if no API key is set or network is unavailable, it gracefully
    falls back to local Slow-Sync Flash Drag and attaches a plain-English note.
    """

    name = "cloud_ai"

    def apply(
        self,
        linear_rgb: np.ndarray,
        cfg: Stage2MotionConfig,
        brush: Optional[UserBrushMask] = None,
        color_cfg: Optional[OpticalColorConfig] = None,
        cloud_cfg: Optional[CloudAIConfig] = None,
    ) -> MotionResult:
        cc = (cloud_cfg or CloudAIConfig()).validated()
        if not cc.api_key:
            fallback = LocalFlashDragProvider().apply(linear_rgb, cfg, brush, color_cfg)
            fallback.note = (
                "No Cloud AI key set in Settings — ran fast local Flash Drag instead."
            )
            return fallback

        try:
            import httpx

            srgb = linear_to_srgb_u8(linear_rgb, color_cfg)
            buf = io.BytesIO()
            Image.fromarray(srgb).save(buf, format="JPEG", quality=94)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            data_uri = f"data:image/jpeg;base64,{b64}"

            if cc.provider == "fal":
                model_path = cc.model or "fal-ai/flux/dev/image-to-image"
                url = f"https://fal.run/{model_path}"
                headers = {
                    "Authorization": f"Key {cc.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "image_url": data_uri,
                    "prompt": cc.prompt,
                    "strength": cc.strength,
                }
                with httpx.Client(timeout=45.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    images = data.get("images") or []
                    if not images or not images[0].get("url"):
                        raise ValueError("Cloud AI response did not include an image URL.")
                    img_resp = client.get(images[0]["url"])
                    img_resp.raise_for_status()
                    out_pil = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            else:
                # Replicate predictions API
                model_ver = cc.model or "black-forest-labs/flux-dev"
                url = f"https://api.replicate.com/v1/models/{model_ver}/predictions"
                headers = {
                    "Authorization": f"Bearer {cc.api_key}",
                    "Content-Type": "application/json",
                    "Prefer": "wait=40",
                }
                payload = {
                    "input": {
                        "image": data_uri,
                        "prompt": cc.prompt,
                        "prompt_strength": cc.strength,
                    }
                }
                with httpx.Client(timeout=50.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    output = data.get("output")
                    out_url = output[0] if isinstance(output, list) and output else output
                    if not isinstance(out_url, str):
                        raise ValueError("Replicate did not return a finished image URL.")
                    img_resp = client.get(out_url)
                    img_resp.raise_for_status()
                    out_pil = Image.open(io.BytesIO(img_resp.content)).convert("RGB")

            h, w = linear_rgb.shape[:2]
            if out_pil.size != (w, h):
                out_pil = out_pil.resize((w, h), Image.Resampling.LANCZOS)
            out_lin = srgb_u8_to_linear(np.array(out_pil, dtype=np.uint8), color_cfg)
            return MotionResult(
                linear_rgb=out_lin,
                provider_used=f"cloud_ai:{cc.provider}",
                note=f"Generated with Cloud AI ({cc.provider}).",
            )
        except Exception as exc:
            fallback = LocalFlashDragProvider().apply(linear_rgb, cfg, brush, color_cfg)
            fallback.note = f"Cloud AI unavailable ({exc}); used local Flash Drag fallback."
            return fallback


# Pluggable provider registry — external SaaS apps can register custom providers easily.
MOTION_PROVIDERS: Dict[str, MotionProvider] = {
    "handheld_shake": LocalHandheldShakeProvider(),
    "flash_drag": LocalFlashDragProvider(),
    "pivot_swing": LocalPivotSwingProvider(),
    "cloud_ai": CloudAIMotionProvider(),
}


def register_motion_provider(name: str, provider: MotionProvider) -> None:
    """Register a custom Stage 2 MotionProvider (for SaaS extensions or cloud models)."""
    MOTION_PROVIDERS[name] = provider


def apply_stage2_motion(
    linear_rgb: np.ndarray,
    cfg: Stage2MotionConfig,
    brush: Optional[UserBrushMask] = None,
    color_cfg: Optional[OpticalColorConfig] = None,
    cloud_cfg: Optional[CloudAIConfig] = None,
) -> MotionResult:
    """Dispatch Stage 2 motion processing to the selected pluggable provider."""
    cfg = cfg.validated()
    if not cfg.enabled:
        return MotionResult(linear_rgb=linear_rgb.copy(), provider_used="disabled")

    provider = MOTION_PROVIDERS.get(cfg.method, MOTION_PROVIDERS["handheld_shake"])
    return provider.apply(
        linear_rgb=linear_rgb,
        cfg=cfg,
        brush=brush,
        color_cfg=color_cfg,
        cloud_cfg=cloud_cfg,
    )
