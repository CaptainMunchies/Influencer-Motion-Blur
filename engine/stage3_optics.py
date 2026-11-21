"""Stage 3: Dirty Smartphone Lens Glow, Atmospheric Haze & Edge Fringing.

Calibrated via visual QA against real iPhone night/flash/daylight candid photos:
- Highlight bloom targets true specular/light sources (high linear luminance) rather
  than washing out large matte white shirts or bright daytime skies.
- Black floor lift uses a gentle perceptual curve so indoor/daylight blacks stay rich
  while dark flash-party scenes get an authentic subtle smudged-lens atmospheric veil.
- Subtle radial chromatic aberration stays strictly zero at center/face and gently
  fringes high-contrast outer edges.
"""

from __future__ import annotations

import cv2
import numpy as np

from .schema import Stage3OpticsConfig


def _fast_wide_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Fast multi-scale wide Gaussian blur via downsample -> blur -> upsample (~3ms on CPU)."""
    h, w = img.shape[:2]
    if sigma_px <= 1.0:
        return img.copy()

    factor = max(1, min(8, int(sigma_px / 6.0)))
    if factor > 1:
        sw = max(8, w // factor)
        sh = max(8, h // factor)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(
            small, (0, 0), sigmaX=sigma_px / factor, borderType=cv2.BORDER_REFLECT_101
        )
        return cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_px, borderType=cv2.BORDER_REFLECT_101)


def _apply_radial_chromatic_aberration(linear_rgb: np.ndarray, strength: float) -> np.ndarray:
    """Very subtle radial R/B edge fringing (zero at center/face, gentle near corners)."""
    if strength <= 1e-3:
        return linear_rgb

    h, w = linear_rgb.shape[:2]
    max_shift_ratio = 0.0020 * strength

    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy

    rn2 = (dx / max(1.0, cx)) ** 2 + (dy / max(1.0, cy)) ** 2
    radial_factor = np.clip((rn2 - 0.15) * 0.72, 0.0, 1.35)

    map_r_x = (xx - dx * max_shift_ratio * radial_factor).astype(np.float32)
    map_r_y = (yy - dy * max_shift_ratio * radial_factor).astype(np.float32)
    map_b_x = (xx + dx * max_shift_ratio * 0.85 * radial_factor).astype(np.float32)
    map_b_y = (yy + dy * max_shift_ratio * 0.85 * radial_factor).astype(np.float32)

    out = linear_rgb.copy()
    out[..., 0] = cv2.remap(
        linear_rgb[..., 0], map_r_x, map_r_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    out[..., 2] = cv2.remap(
        linear_rgb[..., 2], map_b_x, map_b_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    return out


def apply_stage3_optics(
    linear_rgb: np.ndarray,
    cfg: Stage3OpticsConfig,
    focus_center: tuple[float, float] = (0.50, 0.42),
) -> np.ndarray:
    """Apply dirty smartphone lens veiling glare, flash falloff, bar shadow cast & edge fringing in Linear RGB."""
    cfg = cfg.validated()
    if not cfg.enabled:
        return linear_rgb.copy()

    out = linear_rgb.copy()
    h, w = out.shape[:2]
    short_edge = float(min(h, w))

    # 0A. Direct On-Camera Phone Flash Falloff (inverse-square foreground cone + specular sheen)
    if cfg.flash_falloff > 0.01:
        s = cfg.flash_falloff
        fcx = float(np.clip(focus_center[0], 0.15, 0.85))
        fcy = float(np.clip(focus_center[1] + 0.06, 0.20, 0.80))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = (xx / max(1.0, float(w)) - fcx) * (w / max(float(w), float(h))) * 1.35
        dy = (yy / max(1.0, float(h)) - fcy) * (h / max(float(w), float(h))) * 1.10
        dist2 = dx * dx + dy * dy

        flash_cone = 1.0 / (1.0 + 2.8 * dist2)
        gain = 1.0 + ((0.52 + 0.78 * flash_cone) - 1.0) * s
        flash_mult = np.stack(
            [
                gain * (1.0 - 0.015 * s * (1.0 - flash_cone)),
                gain * (1.0 + 0.008 * s * (1.0 - flash_cone)),
                gain * (1.0 + 0.020 * s * flash_cone),
            ],
            axis=-1,
        )
        out = out * flash_mult
        lum_f = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        spec = (np.clip((lum_f - 0.48) / 0.52, 0.0, 1.0) ** 2.0) * flash_cone
        out = out + spec[..., np.newaxis] * (0.11 * s)
        out = out / (1.0 + 0.16 * np.maximum(out - 0.88, 0.0))

    # 0B. Dirty Bar / Club Shadow Cast (calibrated dark charcoal-olive shadows + warm highlight split)
    if cfg.bar_shadow_cast > 0.01:
        b = cfg.bar_shadow_cast
        lum_b = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        sh = np.clip(1.0 - lum_b * 3.4, 0.0, 1.0)[..., np.newaxis]
        hi = np.clip((lum_b - 0.30) * 1.6, 0.0, 1.0)[..., np.newaxis]

        # Multiplicative dark-shadow shift + subtle calibrated linear veil (avoids cyan/teal fog in blacks!)
        sh_mult = np.array([1.0 - 0.14 * b, 1.0 + 0.05 * b, 1.0 + 0.03 * b], dtype=np.float32)
        hi_mult = np.array([1.0 + 0.08 * b, 1.0 + 0.02 * b, 1.0 - 0.09 * b], dtype=np.float32)
        shadow_veil = np.array([-0.0018, 0.0034, 0.0022], dtype=np.float32) * b

        xx_lin = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :, np.newaxis]
        side_spill = (np.clip((xx_lin - 0.58) / 0.42, 0.0, 1.0) ** 2.0)
        neon_cast = np.array([0.0020, 0.0038, 0.0095], dtype=np.float32) * b

        color_mult = (
            1.0
            + sh * (sh_mult[np.newaxis, np.newaxis, :] - 1.0)
            + hi * (hi_mult[np.newaxis, np.newaxis, :] - 1.0)
        )
        out = (
            out * color_mult
            + sh * shadow_veil[np.newaxis, np.newaxis, :]
            + side_spill * (0.35 + 0.65 * sh) * neon_cast[np.newaxis, np.newaxis, :]
        )
        out = np.maximum(out, 0.0)

    # 1. Subtle radial edge chromatic aberration
    if cfg.edge_fringing > 0.01:
        out = _apply_radial_chromatic_aberration(out, cfg.edge_fringing)

    # 2. Linear-light dirty-lens veiling glare (true specular highlight halation + restrained scatter)
    if cfg.glow_amount > 0.01:
        lum = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]

        # Extract true bright specular highlights (linear > 0.55 ≈ sRGB > 0.76) so matte skin
        # and midtone white clothing don't turn into milky fog, while point lights & flash shine glow.
        hl_mask = np.clip((lum - 0.52) / 0.52, 0.0, 1.0)
        hl_mask = hl_mask * hl_mask * (3.0 - 2.0 * hl_mask)

        # Auto-dampen glow slightly if a massive fraction of the frame is already bright white
        # (e.g. bright daylight sky or oversized white background) to prevent foggy washout.
        bright_area_ratio = float(np.mean(hl_mask > 0.5))
        area_guard = float(np.clip(1.0 - max(0.0, bright_area_ratio - 0.14) * 1.65, 0.35, 1.0))

        highlights = out * hl_mask[..., np.newaxis]

        # Tight specular halo + wide smudged smartphone glass scatter
        tight_bloom = _fast_wide_blur(highlights, sigma_px=short_edge * 0.014)
        mid_bloom = _fast_wide_blur(highlights, sigma_px=short_edge * 0.042)
        wide_veil = _fast_wide_blur(out, sigma_px=short_edge * 0.085)

        wt = cfg.warmth
        bloom_tint = np.array([1.0 + 0.22 * wt, 1.0 + 0.05 * wt, 1.0 - 0.18 * wt], dtype=np.float32)

        g = cfg.glow_amount * area_guard
        veil_opacity = 0.075 * g
        tight_strength = 0.28 * g
        mid_strength = 0.14 * g

        out = (
            out * (1.0 - veil_opacity)
            + wide_veil * veil_opacity * bloom_tint[np.newaxis, np.newaxis, :]
            + (tight_bloom * tight_strength + mid_bloom * mid_strength) * bloom_tint[np.newaxis, np.newaxis, :]
        )

    # 3. Gentle lifted milky blacks (perceptually calibrated so 0.25 = subtle +7 sRGB, 0.60 = classic flash haze)
    if cfg.haze_lift > 0.01:
        # Target sRGB black lift ~ 0..24/255 mapped cleanly into linear space (gamma 2.2)
        target_srgb_floor = 0.095 * (cfg.haze_lift ** 1.25)
        lift_floor = float(target_srgb_floor ** 2.2)
        wt = cfg.warmth
        lift_rgb = np.array(
            [lift_floor * (1.0 + 0.10 * wt), lift_floor * (1.0 + 0.02 * wt), lift_floor * (1.0 - 0.06 * wt)],
            dtype=np.float32,
        )
        # Apply lift preferentially in shadows (smooth fade-out above midtones so brights keep crisp punch)
        lum2 = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        shadow_only = np.clip(1.0 - lum2 * 2.2, 0.0, 1.0)[..., np.newaxis]
        out = out + shadow_only * lift_rgb[np.newaxis, np.newaxis, :]

    return np.maximum(out, 0.0).astype(np.float32)
