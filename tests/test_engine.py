"""Automated verification & CPU speed benchmark for Influencer Blur Lab."""

from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.server import app
from engine import (
    AIDirectorResult,
    CloudAIConfig,
    PipelineConfig,
    analyze_and_build_auto_config,
    apply_director_guidance_to_config,
    auto_process_image,
    get_builtin_preset,
    list_builtin_presets,
    load_preset_json,
    process_image,
    run_pipeline,
    save_preset_json,
)
from engine.ai_director import build_kie_chat_url, rasterize_guidance_to_brush_rgba


def _make_synthetic_portrait(w: int = 960, h: int = 1200) -> np.ndarray:
    """Create a realistic high-contrast synthetic portrait test pattern with specular highlights."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Dark warm room gradient
    img[..., 0] = 28 + 30 * (xx / w)
    img[..., 1] = 20 + 18 * (yy / h)
    img[..., 2] = 24 + 15 * (1.0 - yy / h)

    # Bright face ellipse in upper-center
    fx = (xx - w * 0.50) / (w * 0.14)
    fy = (yy - h * 0.30) / (h * 0.12)
    face = (fx * fx + fy * fy) < 1.0
    img[face] = [215, 165, 145]

    # Sharp eye details for sharpness verification
    for ex in (0.455, 0.545):
        eye = ((xx - w * ex) ** 2 + (yy - h * 0.285) ** 2) < (w * 0.015) ** 2
        img[eye] = [25, 22, 22]

    # Moving arm/sleeve bands on left & right
    left_arm = (xx > w * 0.12) & (xx < w * 0.28) & (yy > h * 0.35) & (yy < h * 0.65)
    right_arm = (xx > w * 0.72) & (xx < w * 0.88) & (yy > h * 0.35) & (yy < h * 0.65)
    img[left_arm] = [190, 145, 128]
    img[right_arm] = [190, 145, 128]

    # Bright specular flash highlights (>0.85)
    for hx, hy, hr in [(0.50, 0.26, 18), (0.82, 0.22, 28), (0.18, 0.18, 22)]:
        hl = ((xx - w * hx) ** 2 + (yy - h * hy) ** 2) < hr ** 2
        img[hl] = [252, 248, 240]

    return np.clip(img, 0, 255).astype(np.uint8)


def test_presets_and_determinism() -> None:
    img = _make_synthetic_portrait(720, 900)
    presets = list_builtin_presets()
    assert len(presets) == 8

    for p in presets:
        cfg = get_builtin_preset(p["id"])
        r1 = run_pipeline(img, cfg)
        r2 = run_pipeline(img, cfg)
        assert set(r1.stages.keys()) == {"original", "stage1", "stage2", "stage3", "stage4", "stage5"}
        assert np.array_equal(r1.final, r2.final), f"Non-deterministic output for {p['id']}"
        assert r1.final.shape == img.shape
        assert r1.final.dtype == np.uint8
        print(f"  [OK] Preset '{cfg.name}' deterministic ({r1.total_ms:.0f} ms) timings={r1.timings_ms}")


def test_auto_scene_engine() -> None:
    img = _make_synthetic_portrait(720, 900)
    out, analysis = auto_process_image(img, strength=1.0)
    assert out.shape == img.shape
    assert analysis.archetype in ("flash_party", "dark_night_moody", "warm_night", "indoor_candid", "daylight_street")
    assert 0.2 <= analysis.subject_center[0] <= 0.8
    assert 0.15 <= analysis.subject_center[1] <= 0.72
    print(f"  [OK] Layer-1 Auto-Match verified: {analysis.archetype_label} subj={analysis.subject_center}")


def test_independent_stage_toggles() -> None:
    img = _make_synthetic_portrait(480, 600)
    cfg = get_builtin_preset("flash_party_blur")
    cfg.stage1_focus.enabled = False
    cfg.stage2_motion.enabled = False
    cfg.stage3_optics.enabled = False
    cfg.stage4_grain.enabled = False
    cfg.stage5_compression.enabled = False

    res = run_pipeline(img, cfg)
    # When all stages are disabled, only subtle tone roundtrip occurs; close to original
    diff = float(np.mean(np.abs(res.final.astype(np.float32) - img.astype(np.float32))))
    assert diff < 6.0, f"Unexpectedly large diff with all stages off: {diff}"
    print(f"  [OK] All-stages-off bypass verified (mean diff={diff:.2f}/255)")


def test_motion_methods_and_cloud_fallback() -> None:
    img = _make_synthetic_portrait(600, 750)
    for method in ("handheld_shake", "flash_drag", "pivot_swing", "camera_swirl", "cloud_ai"):
        cfg = get_builtin_preset("flash_party_blur")
        cfg.stage2_motion.method = method  # type: ignore[assignment]
        res = run_pipeline(img, cfg, cloud_config=CloudAIConfig(api_key=""))
        assert res.final.shape == img.shape
        if method == "cloud_ai":
            assert len(res.notes) > 0, "Expected friendly fallback note when Cloud API key is empty"
        print(f"  [OK] Motion method '{method}' -> provider={res.motion_provider} ({res.total_ms:.0f} ms)")


def test_brush_mask_and_preset_json() -> None:
    img = _make_synthetic_portrait(540, 680)
    h, w = img.shape[:2]
    brush = np.zeros((h, w, 4), dtype=np.uint8)
    # Paint green keep-sharp on face, red extra-blur on right arm
    brush[int(h * 0.20):int(h * 0.40), int(w * 0.35):int(w * 0.65)] = [0, 230, 0, 255]
    brush[int(h * 0.35):int(h * 0.65), int(w * 0.70):int(w * 0.90)] = [240, 0, 0, 255]

    cfg = get_builtin_preset("flash_party_blur")
    out = process_image(img, cfg, brush_rgba=brush)
    assert out.shape == img.shape

    with tempfile.TemporaryDirectory() as tmp:
        saved = save_preset_json(cfg, tmp)
        loaded = load_preset_json(saved)
        assert loaded.to_dict() == cfg.to_dict()
    print("  [OK] Brush mask & JSON preset roundtrip verified")


def test_fastapi_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200

    presets = client.get("/api/presets").json()
    assert len(presets["builtin"]) == 8

    # Upload a test JPEG
    img = _make_synthetic_portrait(800, 1000)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=92)
    buf.seek(0)

    up = client.post("/api/upload", files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")})
    assert up.status_code == 200
    image_id = up.json()["image_id"]

    # Layer 1 Auto-Process endpoint
    ap = client.post("/api/auto-process", json={"image_id": image_id, "strength": 1.0})
    assert ap.status_code == 200
    ap_data = ap.json()
    assert "analysis" in ap_data and "stage5" in ap_data["stages"]

    cfg_dict = presets["builtin"][0]["config"]
    proc = client.post("/api/process", json={"image_id": image_id, "config": cfg_dict})
    assert proc.status_code == 200
    pdata = proc.json()
    assert set(pdata["stages"].keys()) == {"original", "stage1", "stage2", "stage3", "stage4", "stage5"}

    # Verify rendered stage URL is retrievable
    r_img = client.get(pdata["stages"]["stage5"])
    assert r_img.status_code == 200 and len(r_img.content) > 1000

    # Showdown endpoint
    opt_b = json.loads(json.dumps(cfg_dict))
    opt_b["stage2_motion"]["method"] = "handheld_shake"
    opt_c = json.loads(json.dumps(cfg_dict))
    opt_c["stage2_motion"]["method"] = "pivot_swing"
    sd = client.post(
        "/api/showdown",
        json={
            "image_id": image_id,
            "options": [
                {"label": "A", "config": cfg_dict},
                {"label": "B", "config": opt_b},
                {"label": "C", "config": opt_c},
            ],
        },
    )
    assert sd.status_code == 200
    assert len(sd.json()["options"]) == 3

    # Full-resolution download endpoint
    dl = client.post("/api/download", json={"image_id": image_id, "config": cfg_dict})
    assert dl.status_code == 200
    dl_img = Image.open(io.BytesIO(dl.content))
    assert dl_img.size == (800, 1000)
    print(f"  [OK] FastAPI endpoints verified (process={pdata['total_ms']:.0f} ms, showdown={sd.json()['total_ms']:.0f} ms)")


def test_cpu_speed_1080x1350() -> None:
    img = _make_synthetic_portrait(1080, 1350)
    for preset_id in ("flash_party_blur", "warm_night_missed_focus", "quick_gym_mirror_snap"):
        cfg = get_builtin_preset(preset_id)
        t0 = time.perf_counter()
        res = run_pipeline(img, cfg)
        elapsed_s = time.perf_counter() - t0
        print(f"  [BENCH 1080x1350] {cfg.name}: {elapsed_s:.2f}s (timings={res.timings_ms})")
        assert elapsed_s < 4.5, f"Too slow on CPU: {elapsed_s:.2f}s"


def test_ai_director_rasterize_and_pivots() -> None:
    img = _make_synthetic_portrait(600, 750)
    h, w = img.shape[:2]
    sharp_polys = [
        {"label": "face", "points": [[0.40, 0.20], [0.60, 0.20], [0.60, 0.40], [0.40, 0.40]]}
    ]
    blur_polys = [
        {
            "label": "left_arm",
            "points": [[0.10, 0.35], [0.28, 0.35], [0.28, 0.65], [0.10, 0.65]],
            "pivot": [0.30, 0.40],
            "swing_angle_deg": 9.0,
        }
    ]
    rgba = rasterize_guidance_to_brush_rgba((h, w), sharp_polys, blur_polys)
    assert rgba.shape == (h, w, 4)
    # Face center should be painted green, arm center red
    assert rgba[int(h * 0.30), int(w * 0.50), 1] > 200
    assert rgba[int(h * 0.50), int(w * 0.18), 0] > 200

    res_dir = AIDirectorResult(
        scene_summary="Kept face sharp, blurred left arm.",
        focus_center=(0.50, 0.30),
        keep_sharp_polygons=sharp_polys,
        blur_polygons=blur_polys,
        pivots=[{"x": 0.30, "y": 0.40, "angle_deg": 9.0, "polygon": blur_polys[0]["points"]}],
        recommended_settings={"motion_method": "pivot_swing", "shake_amount": 11.0},
        brush_rgba=rgba,
        model_used="gemini-4-hydrogen",
        provider_used="gemini",
    )
    cfg = apply_director_guidance_to_config(get_builtin_preset("flash_party_blur"), res_dir, apply_slider_recommendations=True)
    assert cfg.stage2_motion.method == "pivot_swing"
    assert len(cfg.stage2_motion.ai_pivots) == 1
    pipe = run_pipeline(img, cfg, brush_rgba=rgba)
    assert pipe.motion_provider == "pivot_swing:ai_pivots"
    assert build_kie_chat_url("gemini-4-neon") == "https://api.kie.ai/gemini-4-neon-openai/v1/chat/completions"
    assert build_kie_chat_url("gemini-3.6-flash") == "https://api.kie.ai/gemini-3-6-flash-openai/v1/chat/completions"
    print(f"  [OK] AI Art Director polygon rasterize + exact arm-pivot swing + Kie.ai URL routing verified ({pipe.total_ms:.0f} ms)")


if __name__ == "__main__":
    print("Running Influencer Blur Lab test suite...")
    test_presets_and_determinism()
    test_auto_scene_engine()
    test_independent_stage_toggles()
    test_motion_methods_and_cloud_fallback()
    test_brush_mask_and_preset_json()
    test_ai_director_rasterize_and_pivots()
    test_fastapi_endpoints()
    test_cpu_speed_1080x1350()
    print("\nALL TESTS PASSED!")
