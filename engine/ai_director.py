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

DIRECTOR_SYSTEM_PROMPT = """You are an expert smartphone photography art director guiding an optical post-processing engine (authentic Instagram flash blur, handheld micro-shake, arm-swing trails, missed focus, direct phone flash falloff, dirty bar shadow cast, and casual accidental phone framing).

You NEVER regenerate the image — instead, you classify the photo (whether it has people OR NO faces at all, like a drinks/table/food/pet/outfit filler photo), paint precise guidance masks, and recommend realistic optical settings.

Return ONLY valid JSON matching this exact schema (all coordinates normalized 0 to 1000, where x=0 is left, x=1000 is right, y=0 is top, y=1000 is bottom):
{
  "scene_role": "portrait" | "non_face_object",
  "scene_summary": "One short plain-English sentence describing what you saw and chose (e.g., 'Night hallway portrait — kept face sharp, painted raised hands, added flash falloff and subtle dirty bar shadow cast.' OR 'Non-face bar table filler — kept foreground drinks and hand sharp with gentle flash and casual tilt.').",
  "focus_center": [x, y],
  "keep_sharp_polygons": [
    {
      "label": "face_or_hero_object",
      "points": [[x1, y1], [x2, y2], ... 6 to 14 contour points tracing every clearly visible face OR the main foreground hero object if there is no face]
    }
  ],
  "blur_polygons": [
    {
      "label": "moving_arm_or_soft_edge",
      "points": [[x1, y1], [x2, y2], ... 6 to 14 contour points tracing moving arms/hands/hair OR dark outer background edges on non-face shots],
      "pivot": [px, py],
      "swing_angle_deg": 8.0
    }
  ],
  "recommended_settings": {
    "motion_method": "flash_drag" | "handheld_shake" | "pivot_swing" | "camera_swirl",
    "direction_angle": 20,
    "shake_amount": 9.0,
    "flash_sharpness": 0.74,
    "outer_smear": 0.62,
    "focus_radius": 1.8,
    "glow_amount": 0.30,
    "grain_amount": 0.28,
    "flash_falloff": 0.22,
    "bar_shadow_cast": 0.18,
    "tilt_angle": -1.0,
    "crop_offset": 0.12
  }
}

Rules for maximum realism:
1. Works on ANY photo — including non-face filler shots (drinks, restaurant table, food, pets, shoes, handbags, street scenes):
   - If one or more people/faces are clearly visible: set `scene_role` to `"portrait"` and trace ALL clearly visible faces in `keep_sharp_polygons` (do NOT intentionally blur out friends' faces in a group photo).
   - If NO face is visible: set `scene_role` to `"non_face_object"`, set `focus_center` on the main foreground hero object (e.g. cocktails, plate, reaching hand, pet, phone), trace that hero object in `keep_sharp_polygons`, and optionally trace empty dark background corners in `blur_polygons` (omit `pivot` for non-limb edges).
2. In `blur_polygons` on portraits, trace moving outer limbs/arms/hands/gestures/hair without overlapping any face. Place `pivot` at the real shoulder or elbow joint `[px, py]` with natural `swing_angle_deg` (-14..+14).
3. Be VERY restrained and whisper-subtle with lighting & framing effects (less is ALWAYS more realistic):
   - `flash_falloff` (0.0..0.32): Use 0.12–0.28 ONLY on genuinely dark night/flash scenes (dim hallways, bars, clubs, dark tables). Set `0.0` on bright daylight, sunlit beaches/cars, or bright indoor/mirror shots!
   - `bar_shadow_cast` (0.0..0.28): Use 0.10–0.24 ONLY on dark bars, clubs, or dim hallways. Set `0.0` on daylight, outdoor, or normal clean bedrooms/mirrors!
   - `tilt_angle` (-1.5..+1.5): Use a barely-noticeable casual tilt (±0.6° to ±1.3°); keep ±0.0° to ±0.6° on mirror selfies or straight-line walls.
   - `crop_offset` (0.0..0.20): Keep subtle (0.08–0.16) so you never crop off heads, hands, or edges.
4. Pick `recommended_settings.motion_method`:
   - "flash_drag" for night/indoor flash, bars, parties, or flash table shots.
   - "pivot_swing" when hands/arms are clearly raised, waving, fixing hair, or gesturing.
   - "camera_swirl" for club, nightlife, neon street, or high-energy candid moments.
   - "handheld_shake" for casual mirror selfies, daylight snaps, or subtle handheld shots.
5. Output raw JSON only — no markdown fences, no commentary."""


CAROUSEL_DIRECTOR_PROMPT = """You are an expert Instagram photography director planning a cohesive, ultra-realistic multi-slide Photo Dump carousel.

You are looking at a numbered contact sheet of slides (SLIDE 1, SLIDE 2, ...). Your job is to decide:
1. Which slides were shot in the **SAME photoshoot / room / lighting moment** (`scene_group`), vs which slides are from **DIFFERENT moments/locations**.
2. What optical recipe each `scene_group` should get so the entire carousel looks 100% real.

CRITICAL REALISM RULES:
- **Same-scene cohesion**: If two or more slides are clearly from the SAME photoshoot, room, outfit, or lighting (for example, Slide 1 and Slide 4 both showing the same person in the same dark hallway), they **MUST share the EXACT SAME `scene_group` ID** and therefore receive the **EXACT SAME camera look, `preset_id`, `flash_falloff`, and `bar_shadow_cast`**! Never make two photos from the same hallway look like totally different cameras/filters.
- **Different-scene variety**: Slides from genuinely different moments (e.g., a mirror selfie at home vs a dim bar table with cocktails vs a dark hallway) SHOULD have different `scene_group` IDs so each moment gets authentic camera physics matching its real lighting (e.g., `flash_falloff: 0.0` and `bar_shadow_cast: 0.0` on daylight/mirror, stronger flash/cast on dark bar/hallway).
- **Non-face shots supported**: If a slide has NO face (e.g., drinks, food, table, shoes, street, pet), set `scene_role` to `"non_face_object"`. Otherwise set `scene_role` to `"portrait"`. Do NOT apply group-shot main-face-only blur separation.

Allowed `preset_id` values:
- `"flash_party_blur"` (night direct flash party / hallway / flash table shot)
- `"club_whip_swirl"` (moody dim club / nightlife ambient swirl)
- `"warm_night_missed_focus"` (warm restaurant / golden evening soft focus)
- `"quick_gym_mirror_snap"` (bright indoor / mirror selfie / clean casual)
- `"daylight_candid_snap"` (bright outdoor / daylight candid)
- `"moving_hands_party"` (raised moving hands / gesture blur)
- `"3am_night_zoom"` (very dark late-night pinch-zoom grainy dump)
- `"lofi_streetlight_blur"` (moody outdoor 3AM streetlight blur)

Return ONLY valid JSON matching this exact schema:
{
  "carousel_summary": "One short plain-English sentence explaining how you grouped matching scenes and balanced the carousel.",
  "scene_groups": {
    "hallway_night": {
      "label": "Night Hallway",
      "preset_id": "flash_party_blur",
      "flash_falloff": 0.22,
      "bar_shadow_cast": 0.18,
      "base_tilt": -1.0
    }
  },
  "slides": [
    {
      "slide_number": 1,
      "scene_group": "hallway_night",
      "scene_role": "portrait" | "non_face_object",
      "tilt_offset": -0.2,
      "crop_offset": 0.12,
      "note": "Short 5-8 word plain-English tag"
    }
  ]
}
Output raw JSON only — no markdown fences, no commentary."""


SUGGESTED_MODELS: Dict[str, List[Dict[str, str]]] = {
    "kie": [
        {"id": "gemini-4-neon", "label": "Gemini 4 Neon (Recommended — High-Precision Vision)"},
        {"id": "gemini-4-hydrogen", "label": "Gemini 4 Hydrogen (Fast Frontier Vision)"},
        {"id": "gemini-4-carbon", "label": "Gemini 4 Carbon (Max Reasoning Vision)"},
        {"id": "gemini-3.8-flash", "label": "Gemini 3.8 Flash (Fast & Light)"},
        {"id": "gemini-3.6-flash", "label": "Gemini 3.6 Flash"},
        {"id": "gemini-3.1-pro", "label": "Gemini 3.1 Pro"},
    ],
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
    scene_role: str = "portrait"  # "portrait" | "non_face_object"
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


def _call_gemini_vision(
    b64: str,
    mime: str,
    cfg: VisionDirectorConfig,
    user_goal: str,
    system_prompt: str = DIRECTOR_SYSTEM_PROMPT,
) -> str:
    model = cfg.model.removeprefix("models/")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = system_prompt
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


def _call_openai_compat_vision(
    b64: str,
    mime: str,
    cfg: VisionDirectorConfig,
    user_goal: str,
    system_prompt: str = DIRECTOR_SYSTEM_PROMPT,
) -> str:
    if cfg.provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1:8785",
            "X-Title": "Influencer Blur Lab",
        }
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }

    prompt = system_prompt
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


def build_kie_chat_url(model: str) -> str:
    """Build Kie.ai's OpenAI-compatible chat completions URL for any model ID.

    Kie.ai routes Gemini models via `https://api.kie.ai/{slug}-openai/v1/chat/completions`,
    converting dots to hyphens (e.g. `gemini-3.6-flash` -> `gemini-3-6-flash-openai`).
    If the user enters a full URL or already includes `-openai`, preserve it cleanly.
    """
    m = model.strip()
    if m.startswith("http://") or m.startswith("https://"):
        return m
    slug = m.lower().replace(".", "-").strip("/")
    if not slug.endswith("-openai"):
        slug = f"{slug}-openai"
    return f"https://api.kie.ai/{slug}/v1/chat/completions"


def _upload_base64_to_kie(b64: str, mime: str, api_key: str) -> Optional[str]:
    """Upload temporary JPEG to Kie.ai's official Base64 File Upload utility (auto-expires in 24h)."""
    url = "https://kieai.redpandaai.co/api/file-base64-upload"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "base64Data": f"data:{mime};base64,{b64}",
        "uploadPath": "influencer-blur-lab/temp",
        "fileName": "director_input.jpg",
    }
    try:
        with httpx.Client(timeout=25.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                return None
            body = resp.json()
            data = body.get("data") if isinstance(body.get("data"), dict) else body
            for field_name in ("downloadUrl", "fileUrl", "url"):
                val = data.get(field_name) if isinstance(data, dict) else None
                if isinstance(val, str) and val.startswith("http"):
                    return val
    except Exception:
        return None
    return None


def _call_kie_vision(
    b64: str,
    mime: str,
    cfg: VisionDirectorConfig,
    user_goal: str,
    system_prompt: str = DIRECTOR_SYSTEM_PROMPT,
) -> str:
    """Call Kie.ai's OpenAI-compatible multimodal endpoint (e.g. Gemini 4 Neon / Hydrogen / Carbon)."""
    url = build_kie_chat_url(cfg.model)
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    # Kie.ai multimodal routes reliably accept hosted URLs via their temp upload utility;
    # fall back seamlessly to inline data URI if upload utility is unreachable.
    hosted_url = _upload_base64_to_kie(b64, mime, cfg.api_key)
    image_ref = hosted_url or f"data:{mime};base64,{b64}"

    prompt = system_prompt
    if user_goal:
        prompt += f"\n\nUser's target look / instruction: {user_goal}"

    base_payload: Dict[str, Any] = {
        "model": cfg.model,
        "stream": False,
        "temperature": 0.15,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_ref}},
                ],
            }
        ],
    }

    with httpx.Client(timeout=75.0) as client:
        # Try with JSON response_format first; if a Kie.ai proxy route rejects response_format, retry cleanly without it.
        payload = dict(base_payload, response_format={"type": "json_object"})
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code == 400 and "response_format" in resp.text.lower():
            resp = client.post(url, headers=headers, json=base_payload)

        if resp.status_code >= 400:
            detail = resp.text[:400]
            try:
                j = resp.json()
                detail = j.get("msg") or j.get("message") or j.get("error", {}).get("message", detail)
            except Exception:
                pass
            raise RuntimeError(f"Kie.ai API error ({resp.status_code}): {detail}")

        data = resp.json()
        # Handle Kie.ai envelope errors that return HTTP 200 with non-200 code
        if isinstance(data.get("code"), int) and data["code"] not in (0, 200) and not data.get("choices"):
            raise RuntimeError(f"Kie.ai error ({data['code']}): {data.get('msg') or data.get('message') or 'Request failed'}")

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Kie.ai returned no choices for this image.")
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not str(content).strip():
            raise RuntimeError("Kie.ai returned an empty response.")
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
            "No Vision AI API key is saved yet. Click the gear icon (⚙) in the top right to paste your Kie.ai, Gemini, OpenAI, or OpenRouter key."
        )

    b64, mime = _encode_image_for_vision(rgb_u8)
    if cfg.provider == "kie":
        raw_text = _call_kie_vision(b64, mime, cfg, user_goal)
    elif cfg.provider == "gemini":
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
    summary = str(parsed.get("scene_summary") or "AI painted sharp subject protection and motion blur regions.").strip()
    raw_role = str(parsed.get("scene_role") or "portrait").strip().lower()
    scene_role = "non_face_object" if raw_role in ("non_face_object", "object", "no_face", "filler") else "portrait"

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
        scene_role=scene_role,
    )


def apply_director_guidance_to_config(
    config: PipelineConfig,
    result: AIDirectorResult,
    apply_slider_recommendations: bool = False,
    apply_realism_recommendations: bool = True,
    apply_framing_recommendations: bool = False,
) -> PipelineConfig:
    """Apply AI-detected focus center, arm pivots, restrained realism extras, and optional sliders."""
    cfg = PipelineConfig.from_dict(config.to_dict())
    cfg.stage1_focus.focus_center_x = result.focus_center[0]
    cfg.stage1_focus.focus_center_y = result.focus_center[1]
    if result.scene_role == "non_face_object":
        cfg.stage1_focus.falloff_mode = "edges"
    cfg.stage2_motion.ai_pivots = result.pivots

    rec = result.recommended_settings or {}
    if apply_realism_recommendations and rec:
        if "flash_falloff" in rec:
            cfg.stage3_optics.flash_falloff = float(max(0.0, min(0.32, float(rec["flash_falloff"]))))
        if "bar_shadow_cast" in rec:
            cfg.stage3_optics.bar_shadow_cast = float(max(0.0, min(0.28, float(rec["bar_shadow_cast"]))))
    if apply_framing_recommendations and rec:
        if "tilt_angle" in rec:
            cfg.stage5_compression.tilt_angle = float(max(-1.6, min(1.6, float(rec["tilt_angle"]))))
        if "crop_offset" in rec:
            cfg.stage5_compression.crop_offset = float(max(0.0, min(0.20, float(rec["crop_offset"]))))

    if apply_slider_recommendations and rec:
        method = rec.get("motion_method")
        if method in ("flash_drag", "handheld_shake", "pivot_swing", "camera_swirl"):
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


def _build_numbered_contact_sheet(images_rgb: List[np.ndarray]) -> np.ndarray:
    """Build a crisp numbered contact sheet (SLIDE 1..N) so the Vision LLM inspects the whole carousel at once."""
    from PIL import ImageDraw

    n = len(images_rgb)
    cols = min(4, max(1, n))
    rows = (n + cols - 1) // cols
    cell_w, cell_h = 340, 420
    pad = 14
    header_h = 30

    canvas = Image.new(
        "RGB",
        (cols * cell_w + (cols + 1) * pad, rows * (cell_h + header_h) + (rows + 1) * pad),
        (16, 18, 22),
    )
    draw = ImageDraw.Draw(canvas)

    for idx, rgb in enumerate(images_rgb):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + header_h + pad)

        draw.rectangle([x0, y0, x0 + cell_w, y0 + header_h], fill=(35, 38, 45))
        draw.text((x0 + 10, y0 + 8), f"SLIDE {idx + 1}", fill=(245, 166, 35))

        pil = Image.fromarray(rgb, mode="RGB")
        pil.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        ox = x0 + (cell_w - pil.width) // 2
        oy = y0 + header_h + (cell_h - pil.height) // 2
        canvas.paste(pil, (ox, oy))

    return np.array(canvas, dtype=np.uint8)


def direct_carousel_with_ai(
    images_rgb: List[np.ndarray],
    director_cfg: VisionDirectorConfig,
    carousel_vibe: str = "one_night_out",
    auto_tilt: bool = True,
    auto_flash_cast: bool = True,
    vary_per_slide: bool = True,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """One-shot Vision LLM Carousel Director: groups same-photoshoot slides together & assigns cohesive recipes."""
    from .auto_scene import analyze_and_build_auto_config
    from .presets import get_builtin_preset

    cfg = director_cfg.validated()
    if not cfg.api_key:
        raise ValueError("No Vision AI API key is configured.")

    sheet = _build_numbered_contact_sheet(images_rgb)
    b64, mime = _encode_image_for_vision(sheet, max_edge=1440)
    goal = (
        f"Target Carousel Vibe: {carousel_vibe}. "
        f"There are {len(images_rgb)} slides numbered SLIDE 1 through SLIDE {len(images_rgb)}. "
        "Remember: slides taken in the same room/photoshoot/outfit MUST share the EXACT same scene_group and camera look!"
    )

    if cfg.provider == "kie":
        raw_text = _call_kie_vision(b64, mime, cfg, goal, system_prompt=CAROUSEL_DIRECTOR_PROMPT)
    elif cfg.provider == "gemini":
        raw_text = _call_gemini_vision(b64, mime, cfg, goal, system_prompt=CAROUSEL_DIRECTOR_PROMPT)
    else:
        raw_text = _call_openai_compat_vision(b64, mime, cfg, goal, system_prompt=CAROUSEL_DIRECTOR_PROMPT)

    parsed = _extract_json_object(raw_text)
    raw_groups = parsed.get("scene_groups") if isinstance(parsed.get("scene_groups"), dict) else {}
    raw_slides = parsed.get("slides") if isinstance(parsed.get("slides"), list) else []
    carousel_summary = str(parsed.get("carousel_summary") or "Vision AI grouped matching scenes for consistent camera realism.").strip()

    # Map slide_number -> LLM decision
    slide_by_num: Dict[int, Dict[str, Any]] = {}
    for s in raw_slides:
        if isinstance(s, dict) and "slide_number" in s:
            try:
                slide_by_num[int(s["slide_number"])] = s
            except Exception:
                continue

    # Build canonical PipelineConfig per LLM scene_group so matching photos get IDENTICAL base optics & lighting!
    group_cfgs: Dict[str, PipelineConfig] = {}
    group_labels: Dict[str, str] = {}
    group_base_tilt: Dict[str, float] = {}

    for g_id, g_meta in raw_groups.items():
        if not isinstance(g_meta, dict):
            continue
        pid = str(g_meta.get("preset_id") or "flash_party_blur")
        try:
            base = get_builtin_preset(pid)
        except Exception:
            base = get_builtin_preset("flash_party_blur")

        c = PipelineConfig.from_dict(base.to_dict())
        if auto_flash_cast:
            c.stage3_optics.flash_falloff = float(max(0.0, min(0.32, float(g_meta.get("flash_falloff", 0.0)))))
            c.stage3_optics.bar_shadow_cast = float(max(0.0, min(0.28, float(g_meta.get("bar_shadow_cast", 0.0)))))
        else:
            c.stage3_optics.flash_falloff = 0.0
            c.stage3_optics.bar_shadow_cast = 0.0

        group_cfgs[str(g_id)] = c.validated()
        group_labels[str(g_id)] = str(g_meta.get("label") or str(g_id).replace("_", " ").title())
        group_base_tilt[str(g_id)] = float(max(-1.4, min(1.4, float(g_meta.get("base_tilt", -1.0)))))

    plan: List[Dict[str, Any]] = []
    for idx, img in enumerate(images_rgb):
        local_an = analyze_and_build_auto_config(
            img,
            strength=1.0,
            auto_tilt=auto_tilt,
            auto_flash_cast=auto_flash_cast,
        )
        s_meta = slide_by_num.get(idx + 1, {})
        g_id = str(s_meta.get("scene_group") or f"scene_{idx + 1}")
        raw_role = str(s_meta.get("scene_role") or local_an.scene_role).lower()
        role = "non_face_object" if raw_role in ("non_face_object", "object", "no_face", "filler") else "portrait"

        if g_id in group_cfgs:
            c = PipelineConfig.from_dict(group_cfgs[g_id].to_dict())
            g_label = group_labels[g_id]
            base_tilt = group_base_tilt.get(g_id, -1.6)
        else:
            c = PipelineConfig.from_dict(local_an.config.to_dict())
            g_label = f"Scene {idx + 1} · {local_an.archetype_label}"
            base_tilt = c.stage5_compression.tilt_angle

        c.stage1_focus.focus_center_x = local_an.subject_center[0]
        c.stage1_focus.focus_center_y = local_an.subject_center[1]
        if role == "non_face_object":
            c.stage1_focus.falloff_mode = "edges"

        slide_seed = int(local_an.config.stage2_motion.seed + seed_offset + idx * 71)
        c.stage2_motion.seed = slide_seed
        c.stage4_grain.seed = slide_seed + 19
        c.stage5_compression.framing_seed = slide_seed + 37

        if auto_tilt:
            jitter = float(max(-0.35, min(0.35, float(s_meta.get("tilt_offset", 0.0))))) if vary_per_slide else 0.0
            c.stage5_compression.tilt_angle = float(np.clip(base_tilt + jitter, -1.7, 1.7))
            c.stage5_compression.crop_offset = float(np.clip(float(s_meta.get("crop_offset", 0.13)), 0.0, 0.20))
        else:
            c.stage5_compression.tilt_angle = 0.0
            c.stage5_compression.crop_offset = 0.0

        plan.append({
            "index": idx,
            "scene_group": g_id,
            "scene_group_label": g_label,
            "scene_role": role,
            "archetype_label": c.name,
            "summary": str(s_meta.get("note") or local_an.plain_summary),
            "config": c.validated(),
        })

    return {
        "carousel_summary": carousel_summary,
        "model_used": cfg.model,
        "provider_used": cfg.provider,
        "slides": plan,
    }


def list_available_vision_models(provider: str, api_key: str) -> List[Dict[str, str]]:
    """Query the live provider API to fetch the newest vision-capable model IDs for this key."""
    key = (api_key or "").strip()
    fallback = SUGGESTED_MODELS.get(provider, SUGGESTED_MODELS["kie"])
    if not key or provider == "kie":
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
