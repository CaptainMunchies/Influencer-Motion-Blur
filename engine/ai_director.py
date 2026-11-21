"""Headless AI Art Director (Vision-Guided Masking & Effect Control).

NEVER regenerates or alters pixels with generative image models. Instead, a high-capability
vision model (Gemini 4 / Gemini 3.8, OpenAI GPT-6.2, or OpenRouter) inspects the photo and
returns precise polygon masks, face focus center, arm-swing joint pivots, and optional
optical slider recommendations.

The returned polygons are rasterized directly into the same Green/Red RGBA brush format
used by the interactive Focus Brush, so both the local QA UI and future SaaS pipelines
can drive our fast CPU optical engine with zero identity drift.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np
from PIL import Image

from .schema import PipelineConfig, VisionDirectorConfig

DIRECTOR_SYSTEM_PROMPT = """You are an expert smartphone photography art director guiding an optical post-processing engine (authentic Instagram flash blur, handheld micro-shake, arm-swing trails, and missed focus).

You NEVER regenerate the image — instead, you paint precise guidance masks and identify exact focal & motion landmarks so our physics engine blurs only what should move while keeping the subject's face and identity crisp.

Return ONLY valid JSON matching this exact schema (all coordinates normalized 0 to 1000, where x=0 is left, x=1000 is right, y=0 is top, y=1000 is bottom):
{
  "scene_summary": "One short plain-English sentence describing what you saw and what you painted (e.g., 'Kept her face crisp and painted both raised arms/hands for soft party motion trails.').",
  "focus_center": [x, y],
  "keep_sharp_polygons": [
    {
      "label": "face",
      "points": [[x1, y1], [x2, y2], ... 8 to 14 contour points tracing the face/eyes/head closely]
    }
  ],
  "blur_polygons": [
    {
      "label": "left_arm_hand",
      "points": [[x1, y1], [x2, y2], ... 6 to 14 contour points tracing a moving arm, hand, gesture, foreground edge, or hair movement],
      "pivot": [px, py],
      "swing_angle_deg": 8.0
    }
  ],
  "recommended_settings": {
    "motion_method": "flash_drag" | "handheld_shake" | "pivot_swing",
    "direction_angle": 20,
    "shake_amount": 12.0,
    "flash_sharpness": 0.70,
    "outer_smear": 0.70,
    "focus_radius": 2.0,
    "glow_amount": 0.38,
    "grain_amount": 0.36
  }
}

Rules for high accuracy:
1. ALWAYS trace the subject's face (and eyes/mouth) tightly in `keep_sharp_polygons` using 8–14 polygon points following the real contour — NEVER leave the face out.
2. In `blur_polygons`, trace the actual outer limbs/arms/hands/gestures/props that naturally move in candid night or mirror shots (or foreground edges/railings for stairs/depth shots). Do NOT overlap the face.
3. For each arm/hand in `blur_polygons`, place `pivot` at the real shoulder or elbow joint `[px, py]` that the limb swings around, and pick a natural `swing_angle_deg` between -14 and +14 (positive = counter-clockwise, negative = clockwise).
4. Pick `recommended_settings.motion_method`:
   - "flash_drag" for night/indoor flash or party shots where face stays sharp and background/arms trail.
   - "pivot_swing" when one or both hands/arms are clearly raised, waving, fixing hair, or gesturing against a mostly still background.
   - "handheld_shake" for casual mirror selfies or dim-light snapshots where the whole camera gently moved.
5. Output raw JSON only — no markdown fences, no commentary."""


SUGGESTED_MODELS: Dict[str, List[Dict[str, str]]] = {
    "gemini": [
        {"id": "gemini-4-hydrogen", "label": "Gemini 4 Hydrogen (Fast Frontier Vision)"},
        {"id": "gemini-4-neon", "label": "Gemini 4 Neon (High-Precision Vision)"},
        {"id": "gemini-4-carbon", "label": "Gemini 4 Carbon (Max Reasoning Vision)"},
        {"id": "gemini-4-argon", "label": "Gemini 4 Argon (Ultra-Fast)"},
        {"id": "gemini-3.8-flash", "label": "Gemini 3.8 Flash (Balanced & Fast)"},
    ],
    "openai": [
        {"id": "gpt-6.2-sol", "label": "GPT-6.2 Sol (Balanced Flagship Vision)"},
        {"id": "gpt-6.2-astra", "label": "GPT-6.2 Astra (Top Flagship Vision)"},
        {"id": "gpt-6.1-terra", "label": "GPT-6.1 Terra (Spatial & Vision Specialist)"},
    ],
    "openrouter": [
        {"id": "google/gemini-4-hydrogen", "label": "Google: Gemini 4 Hydrogen"},
        {"id": "google/gemini-4-neon", "label": "Google: Gemini 4 Neon"},
        {"id": "openai/gpt-6.2-sol", "label": "OpenAI: GPT-6.2 Sol"},
        {"id": "openai/gpt-6.2-astra", "label": "OpenAI: GPT-6.2 Astra"},
    ],
}


@dataclass
class AIDirectorResult:
    """Structured guidance produced by the Vision AI Art Director."""

    scene_summary: str
    focus_center: Tuple[float, float]  # normalized (0..1, 0..1)
    keep_sharp_polygons: List[Dict[str, Any]]
    blur_polygons: List[Dict[str, Any]]
    pivots: List[Dict[str, Any]]
    recommended_settings: Dict[str, Any]
    brush_rgba: np.ndarray  # HxWx4 uint8 RGBA mask (Green=sharp, Red=blur)
    model_used: str
    provider_used: str
    raw_notes: List[str] = field(default_factory=list)


def _encode_image_for_vision(rgb_u8: np.ndarray, max_edge: int = 1024) -> Tuple[str, str]:
    """Resize softly to <=1024px JPEG base64 for fast, crisp vision reasoning."""
    pil = Image.fromarray(rgb_u8, mode="RGB")
    w, h = pil.size
    if max(w, h) > max_edge:
        scale = max_edge / float(max(w, h))
        pil = pil.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, "image/jpeg"


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Robustly parse JSON from model output even if wrapped in prose or fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("The AI model did not return valid JSON guidance.")


def _norm_coord(val: Any) -> float:
    """Accept either 0..1000 or 0..1 coordinates seamlessly."""
    v = float(val)
    if v > 1.5:
        v = v / 1000.0
    return float(max(0.0, min(1.0, v)))


def _normalize_points(raw_pts: Any) -> List[List[float]]:
    if not isinstance(raw_pts, list):
        return []
    pts: List[List[float]] = []
    for item in raw_pts:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pts.append([_norm_coord(item[0]), _norm_coord(item[1])])
    return pts


def rasterize_guidance_to_brush_rgba(
    hw: Tuple[int, int],
    keep_sharp_polygons: List[Dict[str, Any]],
    blur_polygons: List[Dict[str, Any]],
) -> np.ndarray:
    """Paint the AI's detected polygons into an RGBA brush layer (Green=Keep Sharp, Red=Make Blurry)."""
    h, w = hw
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # First paint Red ("Make Blurry") regions
    blur_mask = np.zeros((h, w), dtype=np.uint8)
    for region in blur_polygons:
        pts = region.get("points") or []
        if len(pts) < 3:
            continue
        poly = np.array([[int(round(p[0] * (w - 1))), int(round(p[1] * (h - 1)))] for p in pts], dtype=np.int32)
        cv2.fillPoly(blur_mask, [poly], 255)

    # Slightly expand blur strokes with a soft brush-like dilation so edges of limbs are well-covered
    if np.any(blur_mask):
        rad = max(3, int(min(h, w) * 0.012))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1, rad * 2 + 1))
        blur_mask = cv2.dilate(blur_mask, k)

    # Paint Green ("Keep Sharp") regions on top so face ALWAYS wins over any adjacent arm mask!
    sharp_mask = np.zeros((h, w), dtype=np.uint8)
    for region in keep_sharp_polygons:
        pts = region.get("points") or []
        if len(pts) < 3:
            continue
        poly = np.array([[int(round(p[0] * (w - 1))), int(round(p[1] * (h - 1)))] for p in pts], dtype=np.int32)
        cv2.fillPoly(sharp_mask, [poly], 255)

    if np.any(sharp_mask):
        rad = max(3, int(min(h, w) * 0.010))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1, rad * 2 + 1))
        sharp_mask = cv2.dilate(sharp_mask, k)
        # Clear red where green protects the face
        blur_mask = cv2.bitwise_and(blur_mask, cv2.bitwise_not(sharp_mask))

    # Encode into the exact RGB+Alpha colors used by our Focus Brush canvas
    red_px = blur_mask > 0
    green_px = sharp_mask > 0
    rgba[red_px] = [240, 58, 58, 235]
    rgba[green_px] = [34, 220, 94, 235]
    return rgba


def _call_gemini_vision(b64: str, mime: str, cfg: VisionDirectorConfig, user_goal: str) -> str:
    model = cfg.model.removeprefix("models/")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = DIRECTOR_SYSTEM_PROMPT
    if user_goal:
        prompt += f"\n\nUser's target look / instruction: {user_goal}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.15,
            "responseMimeType": "application/json",
        },
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, params={"key": cfg.api_key}, json=payload)
        if resp.status_code >= 400:
            detail = resp.text[:400]
            try:
                detail = resp.json().get("error", {}).get("message", detail)
            except Exception:
                pass
            raise RuntimeError(f"Gemini API error ({resp.status_code}): {detail}")
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini returned no candidates for this image.")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not text.strip():
            raise RuntimeError("Gemini returned an empty response.")
        return text


def _call_openai_compat_vision(b64: str, mime: str, cfg: VisionDirectorConfig, user_goal: str) -> str:
    if cfg.provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1:8765",
            "X-Title": "Influencer Blur Lab",
        }
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }

    prompt = DIRECTOR_SYSTEM_PROMPT
    if user_goal:
        prompt += f"\n\nUser's target look / instruction: {user_goal}"

    payload: Dict[str, Any] = {
        "model": cfg.model,
        "temperature": 0.15,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}},
                ],
            }
        ],
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            detail = resp.text[:400]
            try:
                detail = resp.json().get("error", {}).get("message", detail)
            except Exception:
                pass
            raise RuntimeError(f"{cfg.provider.title()} API error ({resp.status_code}): {detail}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"{cfg.provider.title()} returned no choices.")
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not str(content).strip():
            raise RuntimeError(f"{cfg.provider.title()} returned an empty response.")
        return str(content)


def direct_photo_with_ai(
    rgb_u8: np.ndarray,
    director_cfg: VisionDirectorConfig,
    user_goal: str = "",
) -> AIDirectorResult:
    """Ask the configured Vision AI model to inspect the photo and paint guidance masks."""
    cfg = director_cfg.validated()
    if not cfg.api_key:
        raise ValueError(
            "No Vision AI API key is saved yet. Click the gear icon (⚙) in the top right to paste your Gemini, OpenAI, or OpenRouter key."
        )

    b64, mime = _encode_image_for_vision(rgb_u8)
    if cfg.provider == "gemini":
        raw_text = _call_gemini_vision(b64, mime, cfg, user_goal)
    else:
        raw_text = _call_openai_compat_vision(b64, mime, cfg, user_goal)

    parsed = _extract_json_object(raw_text)

    # Normalize focus_center
    raw_fc = parsed.get("focus_center")
    if isinstance(raw_fc, (list, tuple)) and len(raw_fc) >= 2:
        focus_center = (_norm_coord(raw_fc[0]), _norm_coord(raw_fc[1]))
    else:
        focus_center = (0.50, 0.32)

    # Normalize sharp & blur polygons + arm-swing pivots
    sharp_polys: List[Dict[str, Any]] = []
    for r in parsed.get("keep_sharp_polygons") or []:
        if not isinstance(r, dict):
            continue
        pts = _normalize_points(r.get("points"))
        if len(pts) >= 3:
            sharp_polys.append({"label": str(r.get("label", "face")), "points": pts})

    blur_polys: List[Dict[str, Any]] = []
    pivots: List[Dict[str, Any]] = []
    for r in parsed.get("blur_polygons") or []:
        if not isinstance(r, dict):
            continue
        pts = _normalize_points(r.get("points"))
        if len(pts) >= 3:
            entry: Dict[str, Any] = {"label": str(r.get("label", "limb")), "points": pts}
            raw_piv = r.get("pivot")
            if isinstance(raw_piv, (list, tuple)) and len(raw_piv) >= 2:
                px, py = _norm_coord(raw_piv[0]), _norm_coord(raw_piv[1])
                ang = float(max(-20.0, min(20.0, float(r.get("swing_angle_deg", 8.0)))))
                entry["pivot"] = [px, py]
                entry["swing_angle_deg"] = ang
                pivots.append({"x": px, "y": py, "angle_deg": ang, "polygon": pts})
            blur_polys.append(entry)

    rec = parsed.get("recommended_settings") if isinstance(parsed.get("recommended_settings"), dict) else {}
    summary = str(parsed.get("scene_summary") or "AI painted sharp face protection and moving limb blur regions.").strip()

    h, w = rgb_u8.shape[:2]
    brush_rgba = rasterize_guidance_to_brush_rgba((h, w), sharp_polys, blur_polys)

    return AIDirectorResult(
        scene_summary=summary,
        focus_center=focus_center,
        keep_sharp_polygons=sharp_polys,
        blur_polygons=blur_polys,
        pivots=pivots,
        recommended_settings=rec,
        brush_rgba=brush_rgba,
        model_used=cfg.model,
        provider_used=cfg.provider,
    )


def apply_director_guidance_to_config(
    config: PipelineConfig,
    result: AIDirectorResult,
    apply_slider_recommendations: bool = False,
) -> PipelineConfig:
    """Apply AI-detected face focus center, arm pivots, and optional slider recommendations to a config."""
    cfg = PipelineConfig.from_dict(config.to_dict())
    cfg.stage1_focus.focus_center_x = result.focus_center[0]
    cfg.stage1_focus.focus_center_y = result.focus_center[1]
    cfg.stage2_motion.ai_pivots = result.pivots

    if apply_slider_recommendations and result.recommended_settings:
        rec = result.recommended_settings
        method = rec.get("motion_method")
        if method in ("flash_drag", "handheld_shake", "pivot_swing"):
            cfg.stage2_motion.method = method
        if "direction_angle" in rec:
            cfg.stage2_motion.direction_angle = float(rec["direction_angle"])
        if "shake_amount" in rec:
            cfg.stage2_motion.shake_amount = float(rec["shake_amount"])
        if "flash_sharpness" in rec:
            cfg.stage2_motion.flash_sharpness = float(rec["flash_sharpness"])
        if "outer_smear" in rec:
            cfg.stage2_motion.outer_smear = float(rec["outer_smear"])
        if "focus_radius" in rec:
            cfg.stage1_focus.radius = float(rec["focus_radius"])
        if "glow_amount" in rec:
            cfg.stage3_optics.glow_amount = float(rec["glow_amount"])
        if "grain_amount" in rec:
            cfg.stage4_grain.grain_amount = float(rec["grain_amount"])

    return cfg.validated()


def list_available_vision_models(provider: str, api_key: str) -> List[Dict[str, str]]:
    """Query the live provider API to fetch the newest vision-capable model IDs for this key."""
    key = (api_key or "").strip()
    fallback = SUGGESTED_MODELS.get(provider, SUGGESTED_MODELS["gemini"])
    if not key:
        return fallback

    try:
        with httpx.Client(timeout=12.0) as client:
            if provider == "gemini":
                resp = client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": key, "pageSize": 100},
                )
                resp.raise_for_status()
                models = []
                for m in resp.json().get("models") or []:
                    methods = m.get("supportedGenerationMethods") or []
                    name = str(m.get("name", "")).removeprefix("models/")
                    if "generateContent" not in methods:
                        continue
                    if not name.startswith("gemini-"):
                        continue
                    # Skip legacy / embedding / audio-only models
                    lower = name.lower()
                    if any(skip in lower for skip in ("embedding", "aqa", "tts", "imagen", "veo", "1.0", "1.5", "2.0")):
                        continue
                    disp = m.get("displayName") or name
                    models.append({"id": name, "label": f"{disp} ({name})"})
                if models:
                    # Sort newest generation numbers first (e.g. gemini-4 > gemini-3 > gemini-2.5)
                    models.sort(key=lambda x: x["id"], reverse=True)
                    return models[:25]
            elif provider == "openai":
                resp = client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                )
                resp.raise_for_status()
                models = []
                for m in resp.json().get("data") or []:
                    mid = str(m.get("id", ""))
                    low = mid.lower()
                    if not (low.startswith("gpt-") or low.startswith("o")):
                        continue
                    if any(skip in low for skip in ("audio", "realtime", "tts", "transcribe", "search", "instruct", "3.5")):
                        continue
                    models.append({"id": mid, "label": mid})
                if models:
                    models.sort(key=lambda x: x["id"], reverse=True)
                    return models[:25]
            elif provider == "openrouter":
                resp = client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                )
                resp.raise_for_status()
                models = []
                for m in resp.json().get("data") or []:
                    mid = str(m.get("id", ""))
                    arch = m.get("architecture") or {}
                    inputs = arch.get("input_modalities") or []
                    if "image" not in inputs:
                        continue
                    if any(mid.startswith(pref) for pref in ("google/gemini-", "openai/gpt-", "anthropic/claude-")):
                        models.append({"id": mid, "label": m.get("name") or mid})
                if models:
                    models.sort(key=lambda x: x["id"], reverse=True)
                    return models[:30]
    except Exception:
        return fallback

    return fallback
