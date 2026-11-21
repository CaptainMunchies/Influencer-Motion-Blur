# Influencer Blur Lab

A fast, standalone Windows 10 desktop-launched studio for creating authentic, accidental Instagram smartphone blur, slow-sync party flash drag, dirty-lens night glow, and low-light iPhone grain — running 100% locally on CPU.

## Quick Start (Windows 10)

1. Double-click **`Launch_InfluencerBlurLab.bat`** (or your Desktop shortcut).
2. It automatically:
   - Pulls the latest updates from GitHub,
   - Installs/updates Python packages inside a clean local `.venv`,
   - Opens **Influencer Blur Lab** automatically in your web browser.
3. Drop any photo (JPG, PNG, or WebP), click one of the 3 built-in looks (**Flash Party Blur**, **Warm Night Missed Focus**, **Quick Gym Mirror Snap**), tweak sliders or compare 3 options in **Option Showdown**, and click **Download Final Result**.

## Architecture (Product-Ready Separation)

- `engine/` — **100% headless Python package** (NumPy, OpenCV, SciPy, Pillow only — zero web/UI dependencies). Ready to drop directly into a SaaS backend:
  ```python
  from PIL import Image
  from engine import process_image, get_builtin_preset

  img = Image.open("portrait.jpg")
  preset = get_builtin_preset("flash_party_blur")
  result_rgb = process_image(img, preset)
  ```
- `app/` — Standalone local visual QA web studio (FastAPI backend + dark-mode browser UI).
- `presets/` — Saved custom JSON presets.

## 5-Stage Optical Pipeline

- **Automatic Linear-Light Pre-step**: sRGB → Linear RGB (gamma 2.2) with highlight headroom boost (`> 0.85`) before optical stages, then soft tone rolloff back to sRGB.
- **Stage 1 — Missed Focus**: True circular aperture disc PSF defocus (not Gaussian) with Face/Center focus falloff & optional Sharp/Blur brush mask.
- **Stage 2 — Motion & Flash Blur**: Pluggable `MotionProvider` interface with:
  - *Handheld Micro-Shake* (curved 2D random-walk PSF with endpoint dwell-time ghosting),
  - *Slow-Sync Flash Drag* (linear-light blend of sharp flash burst + curved ambient shutter drag + outer-limb smear),
  - *Moving Hands Swing* (localized rotational/arc blur around moving arms/hands while keeping the face sharp),
  - *Optional Cloud AI Hook* (fal.ai / Replicate via optional Settings drawer; runs 100% locally when no key is set).
- **Stage 3 — Dirty Lens Glow**: Linear-light veiling glare, soft lifted milky blacks, and subtle radial edge color fringing.
- **Stage 4 — Nighttime Grain**: Applied *after* blur — shadow-weighted high-ISO sensor noise with clumpy iPhone Night Mode correlation, chroma softening, and light edge-preserving smoothing.
- **Stage 5 — Instagram Polish**: Gentle upload softness + authentic 4:2:0 JPEG compression pass.
