"""FastAPI backend for Influencer Blur Lab visual QA studio.

Completely decoupled from `engine/`: this server handles image upload, preview caching,
full-resolution export, custom JSON preset management, optional Cloud AI settings, and
auto-opening the local web UI in the user's browser.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from engine import (
    CloudAIConfig,
    PipelineConfig,
    SUGGESTED_MODELS,
    VisionDirectorConfig,
    analyze_and_build_auto_config,
    apply_director_guidance_to_config,
    direct_photo_with_ai,
    list_available_vision_models,
    list_builtin_presets,
    list_custom_presets,
    run_pipeline,
    save_preset_json,
)
from engine.presets import slugify_preset_name

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
PRESETS_DIR = ROOT_DIR / "presets"
SETTINGS_PATH = ROOT_DIR / "settings.local.json"

PRESETS_DIR.mkdir(parents=True, exist_ok=True)

# Max working preview dimension so interactive slider QA stays 1-2s on CPU;
# "Download Final Result" always renders at the full uploaded resolution.
PREVIEW_MAX_LONG_EDGE = 1280

app = FastAPI(title="Influencer Blur Lab", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class UploadedPhoto:
    def __init__(
        self,
        image_id: str,
        filename: str,
        full_rgb: np.ndarray,
        preview_rgb: np.ndarray,
        subject_center: tuple[float, float] = (0.50, 0.34),
    ):
        self.image_id = image_id
        self.filename = filename
        self.full_rgb = full_rgb
        self.preview_rgb = preview_rgb
        self.subject_center = subject_center
        self.created_at = time.time()


# In-memory session stores (lightweight, single-user desktop studio)
PHOTOS: Dict[str, UploadedPhoto] = {}
RENDER_CACHE: Dict[str, bytes] = {}
MAX_PHOTOS = 8
MAX_RENDER_CACHE = 80


def _prune_caches() -> None:
    if len(PHOTOS) > MAX_PHOTOS:
        oldest = sorted(PHOTOS.items(), key=lambda kv: kv[1].created_at)[:-MAX_PHOTOS]
        for k, _ in oldest:
            PHOTOS.pop(k, None)
    if len(RENDER_CACHE) > MAX_RENDER_CACHE:
        for k in list(RENDER_CACHE.keys())[: len(RENDER_CACHE) - MAX_RENDER_CACHE]:
            RENDER_CACHE.pop(k, None)


def _encode_image_bytes(rgb: np.ndarray, fmt: str = "JPEG", quality: int = 92) -> bytes:
    buf = io.BytesIO()
    pil = Image.fromarray(rgb, mode="RGB")
    if fmt == "PNG":
        pil.save(buf, format="PNG", compress_level=3)
    else:
        pil.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _store_render(rgb: np.ndarray, quality: int = 93) -> str:
    token = uuid.uuid4().hex[:14]
    RENDER_CACHE[token] = _encode_image_bytes(rgb, fmt="JPEG", quality=quality)
    _prune_caches()
    return f"/api/render/{token}.jpg"


def _decode_brush_data_url(data_url: Optional[str]) -> Optional[np.ndarray]:
    if not data_url:
        return None
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url)
        pil = Image.open(io.BytesIO(raw)).convert("RGBA")
        return np.array(pil, dtype=np.uint8)
    except Exception:
        return None


def _load_raw_settings() -> Dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_cloud_settings() -> CloudAIConfig:
    data = _load_raw_settings()
    sub = data.get("cloud_motion") if isinstance(data.get("cloud_motion"), dict) else data
    try:
        return CloudAIConfig(
            provider=sub.get("provider", "fal"),
            api_key=sub.get("api_key", ""),
            model=sub.get("model", ""),
            prompt=sub.get("prompt", CloudAIConfig.prompt),
            strength=float(sub.get("strength", 0.35)),
        ).validated()
    except Exception:
        return CloudAIConfig()


def _load_vision_settings() -> VisionDirectorConfig:
    data = _load_raw_settings()
    sub = data.get("vision_director") if isinstance(data.get("vision_director"), dict) else {}
    try:
        return VisionDirectorConfig(
            provider=sub.get("provider", "kie"),
            api_key=sub.get("api_key", ""),
            model=sub.get("model", ""),
        ).validated()
    except Exception:
        return VisionDirectorConfig()


def _save_all_settings(vision: VisionDirectorConfig, cloud: CloudAIConfig) -> None:
    v = vision.validated()
    c = cloud.validated()
    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "vision_director": {
                    "provider": v.provider,
                    "api_key": v.api_key,
                    "model": v.model,
                },
                "cloud_motion": {
                    "provider": c.provider,
                    "api_key": c.api_key,
                    "model": c.model,
                    "prompt": c.prompt,
                    "strength": c.strength,
                },
            },
            f,
            indent=2,
        )


def _mask_preview(key: str) -> str:
    if not key:
        return ""
    return (key[:4] + "••••" + key[-3:]) if len(key) > 8 else "••••"


def _encode_rgba_png_data_url(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", compress_level=4)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------- Pydantic request schemas ----------

class ProcessRequest(BaseModel):
    image_id: str
    config: Dict[str, Any]
    brush_data_url: Optional[str] = None


class ShowdownOption(BaseModel):
    label: str
    subtitle: str = ""
    config: Dict[str, Any]


class ShowdownRequest(BaseModel):
    image_id: str
    options: List[ShowdownOption] = Field(min_length=1, max_length=4)
    brush_data_url: Optional[str] = None


class SavePresetRequest(BaseModel):
    name: str
    description: str = ""
    config: Dict[str, Any]


class SettingsUpdateRequest(BaseModel):
    # Vision AI Art Director (masks, face center, arm pivots — never regenerates pixels)
    vision_provider: str = "kie"
    vision_api_key: str = "__KEEP__"
    vision_model: str = "gemini-4-neon"
    # Optional Cloud AI Motion (legacy/optional fal.ai or Replicate)
    provider: str = "fal"
    api_key: str = "__KEEP__"
    model: str = ""
    prompt: str = "subtle authentic handheld iPhone night camera motion blur, natural skin"
    strength: float = 0.35


class AIDirectRequest(BaseModel):
    image_id: str
    config: Dict[str, Any]
    apply_sliders: bool = False
    user_goal: str = ""


class AutoProcessRequest(BaseModel):
    image_id: str
    strength: float = 1.0
    seed_offset: int = 0
    use_ai_vision: bool = False
    brush_data_url: Optional[str] = None


class AIModelsRequest(BaseModel):
    provider: str = "gemini"
    api_key: str = "__KEEP__"


# ---------- API Routes ----------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "app": "Influencer Blur Lab"}


@app.get("/api/presets")
def get_presets() -> Dict[str, Any]:
    return {
        "builtin": list_builtin_presets(),
        "custom": list_custom_presets(PRESETS_DIR),
    }


@app.post("/api/presets")
def create_preset(req: SavePresetRequest) -> Dict[str, Any]:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Please enter a name for your preset.")
    cfg = PipelineConfig.from_dict(req.config)
    cfg.name = name
    cfg.description = req.description.strip() or "Custom saved look"
    path = save_preset_json(cfg, PRESETS_DIR)
    return {
        "id": path.stem,
        "builtin": False,
        "name": cfg.name,
        "description": cfg.description,
        "config": cfg.to_dict(),
    }


@app.delete("/api/presets/{preset_id}")
def delete_preset(preset_id: str) -> Dict[str, Any]:
    safe_id = slugify_preset_name(preset_id)
    target = PRESETS_DIR / f"{safe_id}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Preset not found.")
    target.unlink()
    return {"deleted": safe_id}


@app.get("/api/settings")
def get_settings() -> Dict[str, Any]:
    v = _load_vision_settings()
    c = _load_cloud_settings()
    return {
        "vision_provider": v.provider,
        "vision_has_api_key": bool(v.api_key),
        "vision_api_key_preview": _mask_preview(v.api_key),
        "vision_model": v.model,
        "suggested_models": SUGGESTED_MODELS,
        "provider": c.provider,
        "has_api_key": bool(c.api_key),
        "api_key_preview": _mask_preview(c.api_key),
        "model": c.model,
        "prompt": c.prompt,
        "strength": c.strength,
    }


@app.post("/api/settings")
def update_settings(req: SettingsUpdateRequest) -> Dict[str, Any]:
    existing_v = _load_vision_settings()
    existing_c = _load_cloud_settings()
    new_v_key = existing_v.api_key if req.vision_api_key == "__KEEP__" else req.vision_api_key.strip()
    new_c_key = existing_c.api_key if req.api_key == "__KEEP__" else req.api_key.strip()

    v_cfg = VisionDirectorConfig(
        provider=req.vision_provider,  # type: ignore[arg-type]
        api_key=new_v_key,
        model=req.vision_model,
    ).validated()
    c_cfg = CloudAIConfig(
        provider=req.provider,  # type: ignore[arg-type]
        api_key=new_c_key,
        model=req.model,
        prompt=req.prompt,
        strength=req.strength,
    ).validated()
    _save_all_settings(v_cfg, c_cfg)
    return get_settings()


@app.post("/api/ai-models")
def fetch_ai_models(req: AIModelsRequest) -> Dict[str, Any]:
    existing_v = _load_vision_settings()
    key = existing_v.api_key if req.api_key == "__KEEP__" else req.api_key.strip()
    models = list_available_vision_models(req.provider, key)
    return {"provider": req.provider, "models": models}


@app.post("/api/ai-direct")
def ai_direct_photo(req: AIDirectRequest) -> Dict[str, Any]:
    photo = PHOTOS.get(req.image_id)
    if photo is None:
        raise HTTPException(status_code=404, detail="Photo session expired. Please upload your photo again.")

    v_cfg = _load_vision_settings()
    if not v_cfg.api_key:
        raise HTTPException(
            status_code=400,
            detail="Add your Kie.ai, Gemini, OpenAI, or OpenRouter API key in Settings (⚙) first to use AI Auto-Paint.",
        )

    try:
        director_res = direct_photo_with_ai(
            rgb_u8=photo.preview_rgb,
            director_cfg=v_cfg,
            user_goal=req.user_goal,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    base_cfg = PipelineConfig.from_dict(req.config)
    updated_cfg = apply_director_guidance_to_config(
        base_cfg,
        director_res,
        apply_slider_recommendations=req.apply_sliders,
    )

    cloud_cfg = _load_cloud_settings()
    pipe_res = run_pipeline(
        image=photo.preview_rgb,
        config=updated_cfg,
        brush_rgba=director_res.brush_rgba,
        cloud_config=cloud_cfg,
        return_intermediates=True,
    )

    stage_urls = {
        key: _store_render(img, quality=94 if key != "stage5" else 96)
        for key, img in pipe_res.stages.items()
    }
    return {
        "image_id": req.image_id,
        "scene_summary": director_res.scene_summary,
        "model_used": director_res.model_used,
        "provider_used": director_res.provider_used,
        "focus_center": list(director_res.focus_center),
        "pivots_count": len(director_res.pivots),
        "sharp_regions": len(director_res.keep_sharp_polygons),
        "blur_regions": len(director_res.blur_polygons),
        "brush_data_url": _encode_rgba_png_data_url(director_res.brush_rgba),
        "config": updated_cfg.to_dict(),
        "stages": stage_urls,
        "timings_ms": pipe_res.timings_ms,
        "total_ms": pipe_res.total_ms,
        "notes": pipe_res.notes,
    }


@app.post("/api/upload")
async def upload_photo(file: UploadFile = File(...)) -> Dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file was empty.")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read that photo ({exc}). Please try JPG, PNG, or WebP.")

    full_rgb = np.array(pil, dtype=np.uint8)
    fh, fw = full_rgb.shape[:2]

    # Build snappy preview copy for interactive slider tuning
    long_edge = max(fw, fh)
    if long_edge > PREVIEW_MAX_LONG_EDGE:
        scale = PREVIEW_MAX_LONG_EDGE / float(long_edge)
        pw = max(64, int(round(fw * scale)))
        ph = max(64, int(round(fh * scale)))
        preview_rgb = np.array(pil.resize((pw, ph), Image.Resampling.LANCZOS), dtype=np.uint8)
    else:
        preview_rgb = full_rgb.copy()

    image_id = uuid.uuid4().hex[:12]
    initial_analysis = analyze_and_build_auto_config(preview_rgb, strength=1.0)
    PHOTOS[image_id] = UploadedPhoto(
        image_id=image_id,
        filename=file.filename or "photo.jpg",
        full_rgb=full_rgb,
        preview_rgb=preview_rgb,
        subject_center=initial_analysis.subject_center,
    )
    _prune_caches()

    orig_url = _store_render(preview_rgb, quality=94)
    ph, pw = preview_rgb.shape[:2]
    return {
        "image_id": image_id,
        "filename": file.filename or "photo.jpg",
        "full_width": fw,
        "full_height": fh,
        "preview_width": pw,
        "preview_height": ph,
        "subject_center": list(initial_analysis.subject_center),
        "original_url": orig_url,
    }


@app.post("/api/auto-process")
def auto_process_preview(req: AutoProcessRequest) -> Dict[str, Any]:
    """Layer 1 Zero-Effort Endpoint: analyzes the photo & runs the optimal scene-matched recipe."""
    photo = PHOTOS.get(req.image_id)
    if photo is None:
        raise HTTPException(status_code=404, detail="Photo session expired. Please upload your photo again.")

    analysis = analyze_and_build_auto_config(photo.preview_rgb, strength=req.strength)
    cfg = analysis.config
    if req.seed_offset:
        cfg.stage2_motion.seed = int(cfg.stage2_motion.seed + req.seed_offset)
        cfg.stage4_grain.seed = int(cfg.stage4_grain.seed + req.seed_offset)

    brush_rgba = _decode_brush_data_url(req.brush_data_url)
    ai_brush_url: Optional[str] = None
    ai_note: str = ""

    # Optional hands-free Kie.ai / Vision AI upgrade when enabled & API key is saved
    if req.use_ai_vision:
        v_cfg = _load_vision_settings()
        if v_cfg.api_key:
            try:
                dir_res = direct_photo_with_ai(photo.preview_rgb, v_cfg)
                cfg = apply_director_guidance_to_config(cfg, dir_res, apply_slider_recommendations=False)
                brush_rgba = dir_res.brush_rgba
                ai_brush_url = _encode_rgba_png_data_url(dir_res.brush_rgba)
                ai_note = f"AI ({dir_res.model_used}): {dir_res.scene_summary}"
            except Exception as exc:
                ai_note = f"Local auto-match used (Vision AI skipped: {exc})"

    cloud_cfg = _load_cloud_settings()
    res = run_pipeline(
        image=photo.preview_rgb,
        config=cfg,
        brush_rgba=brush_rgba,
        cloud_config=cloud_cfg,
        return_intermediates=True,
    )
    stage_urls = {
        key: _store_render(img, quality=94 if key != "stage5" else 96)
        for key, img in res.stages.items()
    }
    notes = list(res.notes)
    if ai_note:
        notes.insert(0, ai_note)

    return {
        "image_id": req.image_id,
        "analysis": {
            "archetype": analysis.archetype,
            "archetype_label": analysis.archetype_label,
            "plain_summary": analysis.plain_summary,
            "subject_center": list(analysis.subject_center),
            "existing_sharpness": round(analysis.existing_sharpness, 1),
            "sharpness_scale": round(analysis.sharpness_scale, 2),
        },
        "brush_data_url": ai_brush_url,
        "stages": stage_urls,
        "timings_ms": res.timings_ms,
        "total_ms": res.total_ms,
        "motion_provider": res.motion_provider,
        "notes": notes,
        "config": cfg.to_dict(),
    }


@app.post("/api/process")
def process_preview(req: ProcessRequest) -> Dict[str, Any]:
    photo = PHOTOS.get(req.image_id)
    if photo is None:
        raise HTTPException(status_code=404, detail="Photo session expired. Please upload your photo again.")

    cfg = PipelineConfig.from_dict(req.config)
    brush_rgba = _decode_brush_data_url(req.brush_data_url)
    cloud_cfg = _load_cloud_settings()

    res = run_pipeline(
        image=photo.preview_rgb,
        config=cfg,
        brush_rgba=brush_rgba,
        cloud_config=cloud_cfg,
        return_intermediates=True,
    )

    stage_urls = {
        key: _store_render(img, quality=94 if key != "stage5" else 96)
        for key, img in res.stages.items()
    }
    return {
        "image_id": req.image_id,
        "stages": stage_urls,
        "timings_ms": res.timings_ms,
        "total_ms": res.total_ms,
        "motion_provider": res.motion_provider,
        "notes": res.notes,
        "config": cfg.to_dict(),
    }


@app.post("/api/showdown")
def run_showdown(req: ShowdownRequest) -> Dict[str, Any]:
    photo = PHOTOS.get(req.image_id)
    if photo is None:
        raise HTTPException(status_code=404, detail="Photo session expired. Please upload your photo again.")

    brush_rgba = _decode_brush_data_url(req.brush_data_url)
    cloud_cfg = _load_cloud_settings()
    original_url = _store_render(photo.preview_rgb, quality=94)

    results = []
    t_all = time.perf_counter()
    for idx, opt in enumerate(req.options):
        cfg = PipelineConfig.from_dict(opt.config)
        res = run_pipeline(
            image=photo.preview_rgb,
            config=cfg,
            brush_rgba=brush_rgba,
            cloud_config=cloud_cfg,
            return_intermediates=False,
        )
        results.append({
            "index": idx,
            "label": opt.label,
            "subtitle": opt.subtitle,
            "image_url": _store_render(res.final, quality=95),
            "total_ms": res.total_ms,
            "motion_provider": res.motion_provider,
            "notes": res.notes,
            "config": cfg.to_dict(),
        })

    return {
        "image_id": req.image_id,
        "original_url": original_url,
        "options": results,
        "total_ms": round((time.perf_counter() - t_all) * 1000.0, 2),
    }


@app.post("/api/download")
def download_full_res(req: ProcessRequest) -> Response:
    photo = PHOTOS.get(req.image_id)
    if photo is None:
        raise HTTPException(status_code=404, detail="Photo session expired. Please upload your photo again.")

    cfg = PipelineConfig.from_dict(req.config)
    brush_rgba = _decode_brush_data_url(req.brush_data_url)
    cloud_cfg = _load_cloud_settings()

    res = run_pipeline(
        image=photo.full_rgb,
        config=cfg,
        brush_rgba=brush_rgba,
        cloud_config=cloud_cfg,
        return_intermediates=False,
    )

    # If Stage 5 is enabled, save with its exact JPEG quality & 4:2:0 subsampling; otherwise high quality
    buf = io.BytesIO()
    pil = Image.fromarray(res.final, mode="RGB")
    if cfg.stage5_compression.enabled:
        pil.save(buf, format="JPEG", quality=int(cfg.stage5_compression.jpeg_quality), subsampling=2)
    else:
        pil.save(buf, format="JPEG", quality=94, subsampling=0)

    stem = Path(photo.filename).stem or "photo"
    preset_slug = slugify_preset_name(cfg.name)
    out_name = f"{stem}_{preset_slug}.jpg"

    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


@app.get("/api/render/{token}.jpg")
def get_rendered_image(token: str) -> Response:
    data = RENDER_CACHE.get(token)
    if data is None:
        raise HTTPException(status_code=404, detail="Preview expired.")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=300"},
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


def _open_browser_later(url: str) -> None:
    time.sleep(1.0)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Influencer Blur Lab local web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"\n  Influencer Blur Lab is running at: {url}")
    print("  Keep this window open while editing. Close it when you are done.\n")

    if not args.no_browser:
        threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
