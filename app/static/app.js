/* Influencer Blur Lab — 3-Layer Studio (Layer 1 Auto-Match -> Layer 2 Presets -> Layer 3 Fine-Tune) */

const STAGE_META = [
  { key: "original", label: "Original", shortLabel: "Original", toggleField: null },
  { key: "stage1", label: "1. Missed Focus", shortLabel: "1. Focus", toggleField: "s1_enabled" },
  { key: "stage2", label: "2. Motion & Flash", shortLabel: "2. Motion", toggleField: "s2_enabled" },
  { key: "stage3", label: "3. Dirty Lens Glow", shortLabel: "3. Glow", toggleField: "s3_enabled" },
  { key: "stage4", label: "4. Nighttime Grain", shortLabel: "4. Grain", toggleField: "s4_enabled" },
  { key: "stage5", label: "5. Final Result", shortLabel: "5. Final", toggleField: "s5_enabled" },
];

const SHORT_PRESET_NAMES = {
  flash_party_blur: "Flash Party",
  warm_night_missed_focus: "Warm Night",
  quick_gym_mirror_snap: "Mirror Selfie",
  club_whip_swirl: "Club Swirl",
  moving_hands_party: "Moving Hands",
  daylight_candid_snap: "Daylight",
  "3am_night_zoom": "3AM Zoom",
  lofi_streetlight_blur: "Lo-Fi Night",
};

const state = {
  activeTab: "qa",
  advancedOpen: false, // Layer 3 collapsed by default -> ultra-clean Layer 1+2 experience!
  autoMode: true,      // Layer 1 Auto-Match active by default!
  autoStrength: 1.0,
  autoSeedOffset: 0,
  autoAnalysis: null,
  // ON by default: automatically uses AI Auto-Paint on every uploaded photo whenever an API key is saved
  autoRunVision: localStorage.getItem("ibl_auto_vision") !== "0",
  imageId: null,
  originalUrl: "",
  filename: "",
  previewWidth: 0,
  previewHeight: 0,
  subjectCenter: [0.50, 0.34],
  presets: { builtin: [], custom: [] },
  activePresetId: "auto",
  config: null,
  stages: {},
  selectedStage: "stage5",
  viewMode: "side",
  wipePct: 50,
  zoomed: false,
  showdownZoomed: false,
  allExpanded: false,
  brushMode: "off",
  brushHasStrokes: false,
  showMaskOverlay: true,
  isPainting: false,
  isProcessing: false,
  debounceTimer: null,
  strengthTimer: null,
  showdownResults: null,
  suggestedModels: {},
  visionHasKey: false,
};

const $ = (id) => document.getElementById(id);

function showToast(msg, duration = 2400) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), duration);
}

function setStatus(text, show = true) {
  const pill = $("statusPill");
  pill.textContent = text;
  pill.classList.toggle("hidden", !show);
}

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

// ---------- Layer 3 Fine-Tune Drawer Toggle ----------
function setAdvancedOpen(open) {
  state.advancedOpen = Boolean(open);
  $("tab-qa").classList.toggle("advanced-open", state.advancedOpen);
  $("controlsPanel").classList.toggle("hidden", !state.advancedOpen);
  $("filmstripBar").classList.toggle("hidden", !state.advancedOpen);
  $("brushToolbarGroup").classList.toggle("hidden", !state.advancedOpen);
  $("toggleAdvancedBtn").classList.toggle("active-ghost", state.advancedOpen);
  $("toggleAdvancedBtn").textContent = state.advancedOpen ? "✕ Close Fine-Tune" : "🎛️ Fine-Tune";
  if (!state.advancedOpen && state.brushMode !== "off") {
    setBrushMode("off");
  }
  requestAnimationFrame(() => fitStageImages());
}

// ---------- Bulletproof Viewport Image Sizing ----------
function fitStageImages() {
  if (!state.previewWidth || !state.previewHeight) return;
  const aspect = state.previewWidth / state.previewHeight;

  const sizeStack = (viewport, stack, pad = 12) => {
    if (!viewport || !stack) return;
    if (state.zoomed) {
      stack.style.width = `${state.previewWidth}px`;
      stack.style.height = `${state.previewHeight}px`;
      return;
    }
    const availW = Math.max(40, viewport.clientWidth - pad);
    const availH = Math.max(40, viewport.clientHeight - pad);
    let fitW = availW;
    let fitH = fitW / aspect;
    if (fitH > availH) {
      fitH = availH;
      fitW = fitH * aspect;
    }
    stack.style.width = `${Math.floor(fitW)}px`;
    stack.style.height = `${Math.floor(fitH)}px`;
  };

  if (state.viewMode === "side") {
    sizeStack($("leftViewport"), $("leftStack"), 12);
    sizeStack($("rightViewport"), $("rightStack"), 12);
  } else {
    sizeStack($("wipeViewport"), $("wipeStack"), 16);
    syncWipeInnerDimensions();
  }
}

function syncWipeInnerDimensions() {
  const stack = $("wipeStack");
  const beforeImg = $("wipeBeforeImg");
  const clip = $("wipeClip");
  const handle = $("wipeHandle");
  if (!stack || !beforeImg || !clip || !handle) return;
  const w = stack.style.width;
  const h = stack.style.height;
  if (w && h) {
    beforeImg.style.width = w;
    beforeImg.style.height = h;
  }
  clip.style.width = `${state.wipePct}%`;
  handle.style.left = `${state.wipePct}%`;
}

// ---------- Accordion Sidebar ----------
function openAccordionStage(stageKey) {
  if (state.allExpanded) return;
  const target = stageKey === "original" ? "stage2" : stageKey;
  document.querySelectorAll(".acc-card").forEach((card) => {
    card.classList.toggle("open", card.dataset.stageCard === target);
  });
}

function setupAccordion() {
  document.querySelectorAll(".acc-head").forEach((head) => {
    head.addEventListener("click", (e) => {
      if (e.target.closest(".switch")) return;
      const card = head.closest(".acc-card");
      if (state.allExpanded) {
        card.classList.toggle("open");
        return;
      }
      const wasOpen = card.classList.contains("open");
      document.querySelectorAll(".acc-card").forEach((c) => c.classList.remove("open"));
      if (!wasOpen) card.classList.add("open");
    });
  });

  $("expandAllBtn").addEventListener("click", () => {
    state.allExpanded = !state.allExpanded;
    $("expandAllBtn").textContent = state.allExpanded ? "Compact" : "Show all";
    document.querySelectorAll(".acc-card").forEach((card) => {
      if (state.allExpanded) {
        card.classList.add("open");
      } else {
        card.classList.toggle("open", card.dataset.stageCard === (state.selectedStage === "original" ? "stage2" : state.selectedStage));
      }
    });
  });
}

// ---------- Presets (Layer 1 Auto-Match + Layer 2 Quick Looks) ----------
async function loadPresets() {
  const resp = await fetch("/api/presets");
  const data = await resp.json();
  state.presets = data;
  renderPresetBar();

  if (!state.config && data.builtin.length > 0) {
    state.config = deepClone(data.builtin[0].config);
    syncControlsFromConfig();
  }
}

function allPresets() {
  return [...state.presets.builtin, ...state.presets.custom];
}

function renderPresetBar() {
  const container = $("presetList");
  container.innerHTML = "";

  // Layer 1 primary button: ✨ Auto-Match
  const autoBtn = document.createElement("button");
  autoBtn.className = "preset-pill auto-pill" + (state.activePresetId === "auto" ? " active" : "");
  autoBtn.title = "Automatically analyzes your photo and applies the best matching candid phone blur";
  autoBtn.textContent = "✨ Auto-Match";
  autoBtn.addEventListener("click", () => {
    state.autoMode = true;
    state.activePresetId = "auto";
    renderPresetBar();
    if (state.imageId) {
      runAutoMatch(false);
    }
  });
  container.appendChild(autoBtn);

  // Layer 2 optional 1-click presets
  for (const p of allPresets()) {
    const btn = document.createElement("button");
    btn.className = "preset-pill" + (state.activePresetId === p.id ? " active" : "");
    btn.title = p.description || p.name;

    const span = document.createElement("span");
    span.textContent = SHORT_PRESET_NAMES[p.id] || p.name;
    btn.appendChild(span);

    if (!p.builtin) {
      const del = document.createElement("span");
      del.className = "preset-del";
      del.textContent = "✕";
      del.title = "Delete saved preset";
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        await fetch(`/api/presets/${encodeURIComponent(p.id)}`, { method: "DELETE" });
        showToast(`Deleted "${p.name}"`);
        await loadPresets();
      });
      btn.appendChild(del);
    }

    btn.addEventListener("click", () => applyPresetToState(p, true));
    container.appendChild(btn);
  }
}

function scaleConfigByStrength(baseCfg, mult) {
  const cfg = deepClone(baseCfg);
  if (Math.abs(mult - 1.0) < 0.02) return cfg;
  cfg.stage1_focus.radius = Math.max(0, Math.min(14, cfg.stage1_focus.radius * mult));
  cfg.stage2_motion.shake_amount = Math.max(0, Math.min(34, cfg.stage2_motion.shake_amount * mult));
  cfg.stage2_motion.outer_smear = Math.max(0, Math.min(1, cfg.stage2_motion.outer_smear * mult));
  cfg.stage2_motion.swing_angle = Math.max(1.5, Math.min(20, cfg.stage2_motion.swing_angle * mult));
  cfg.stage3_optics.glow_amount = Math.max(0, Math.min(1, cfg.stage3_optics.glow_amount * mult));
  cfg.stage3_optics.haze_lift = Math.max(0, Math.min(1, cfg.stage3_optics.haze_lift * mult));
  cfg.stage4_grain.grain_amount = Math.max(0, Math.min(1, cfg.stage4_grain.grain_amount * mult));
  return cfg;
}

function applyPresetToState(preset, triggerProcess = true) {
  state.autoMode = false;
  state.activePresetId = preset.id;
  const scaled = scaleConfigByStrength(preset.config, state.autoStrength);
  // Keep detected subject center so even manual presets protect off-center faces automatically!
  scaled.stage1_focus.focus_center_x = state.subjectCenter[0];
  scaled.stage1_focus.focus_center_y = state.subjectCenter[1];
  state.config = scaled;

  $("activeLookTitle").textContent = preset.name;
  $("autoBadge").textContent = `Look: ${SHORT_PRESET_NAMES[preset.id] || preset.name}`;
  $("noteBanner").classList.add("hidden");
  syncControlsFromConfig();
  renderPresetBar();
  if (triggerProcess && state.imageId) {
    processCurrentPhoto();
  }
}

// ---------- Sync Config <-> Controls ----------
function syncControlsFromConfig() {
  const c = state.config;
  if (!c) return;

  $("s1_enabled").checked = c.stage1_focus.enabled;
  $("s1_radius").value = c.stage1_focus.radius;
  $("s1_falloff_mode").value = c.stage1_focus.falloff_mode;
  $("s1_falloff_strength").value = Math.round(c.stage1_focus.falloff_strength * 100);

  $("s2_enabled").checked = c.stage2_motion.enabled;
  $("s2_method").value = c.stage2_motion.method;
  $("s2_shake_amount").value = c.stage2_motion.shake_amount;
  $("s2_ghost_strength").value = Math.round(c.stage2_motion.ghost_strength * 100);
  $("s2_flash_sharpness").value = Math.round(c.stage2_motion.flash_sharpness * 100);
  $("s2_outer_smear").value = Math.round(c.stage2_motion.outer_smear * 100);
  $("s2_swing_angle").value = c.stage2_motion.swing_angle;
  $("s2_swing_intensity").value = Math.round(c.stage2_motion.swing_intensity * 100);
  $("s2_direction_angle").value = Math.round(c.stage2_motion.direction_angle);

  $("s3_enabled").checked = c.stage3_optics.enabled;
  $("s3_glow_amount").value = Math.round(c.stage3_optics.glow_amount * 100);
  $("s3_haze_lift").value = Math.round(c.stage3_optics.haze_lift * 100);
  $("s3_warmth").value = Math.round(c.stage3_optics.warmth * 100);
  $("s3_edge_fringing").value = Math.round(c.stage3_optics.edge_fringing * 100);

  $("s4_enabled").checked = c.stage4_grain.enabled;
  $("s4_grain_amount").value = Math.round(c.stage4_grain.grain_amount * 100);
  $("s4_clumpiness").value = Math.round(c.stage4_grain.clumpiness * 100);
  $("s4_isp_smoothing").value = Math.round(c.stage4_grain.isp_smoothing * 100);
  $("s4_digital_zoom").value = Math.round((c.stage4_grain.digital_zoom || 0) * 100);
  $("s4_night_cast").value = Math.round((c.stage4_grain.night_cast || 0) * 100);

  $("s5_enabled").checked = c.stage5_compression.enabled;
  $("s5_insta_softness").value = Math.round(c.stage5_compression.insta_softness * 100);
  $("s5_jpeg_quality").value = Math.round(c.stage5_compression.jpeg_quality);

  updateControlReadoutsAndVisibility();
}

function readControlsIntoConfig() {
  const c = state.config;
  if (!c) return;

  c.stage1_focus.enabled = $("s1_enabled").checked;
  c.stage1_focus.radius = parseFloat($("s1_radius").value);
  c.stage1_focus.falloff_mode = $("s1_falloff_mode").value;
  c.stage1_focus.falloff_strength = parseFloat($("s1_falloff_strength").value) / 100.0;

  c.stage2_motion.enabled = $("s2_enabled").checked;
  c.stage2_motion.method = $("s2_method").value;
  c.stage2_motion.shake_amount = parseFloat($("s2_shake_amount").value);
  c.stage2_motion.ghost_strength = parseFloat($("s2_ghost_strength").value) / 100.0;
  c.stage2_motion.flash_sharpness = parseFloat($("s2_flash_sharpness").value) / 100.0;
  c.stage2_motion.outer_smear = parseFloat($("s2_outer_smear").value) / 100.0;
  c.stage2_motion.swing_angle = parseFloat($("s2_swing_angle").value);
  c.stage2_motion.swing_intensity = parseFloat($("s2_swing_intensity").value) / 100.0;
  c.stage2_motion.direction_angle = parseFloat($("s2_direction_angle").value);

  c.stage3_optics.enabled = $("s3_enabled").checked;
  c.stage3_optics.glow_amount = parseFloat($("s3_glow_amount").value) / 100.0;
  c.stage3_optics.haze_lift = parseFloat($("s3_haze_lift").value) / 100.0;
  c.stage3_optics.warmth = parseFloat($("s3_warmth").value) / 100.0;
  c.stage3_optics.edge_fringing = parseFloat($("s3_edge_fringing").value) / 100.0;

  c.stage4_grain.enabled = $("s4_enabled").checked;
  c.stage4_grain.grain_amount = parseFloat($("s4_grain_amount").value) / 100.0;
  c.stage4_grain.clumpiness = parseFloat($("s4_clumpiness").value) / 100.0;
  c.stage4_grain.isp_smoothing = parseFloat($("s4_isp_smoothing").value) / 100.0;
  c.stage4_grain.digital_zoom = parseFloat($("s4_digital_zoom").value) / 100.0;
  c.stage4_grain.night_cast = parseFloat($("s4_night_cast").value) / 100.0;

  c.stage5_compression.enabled = $("s5_enabled").checked;
  c.stage5_compression.insta_softness = parseFloat($("s5_insta_softness").value) / 100.0;
  c.stage5_compression.jpeg_quality = parseInt($("s5_jpeg_quality").value, 10);

  updateControlReadoutsAndVisibility();
}

function updateControlReadoutsAndVisibility() {
  $("val_s1_radius").textContent = parseFloat($("s1_radius").value).toFixed(1);
  $("val_s1_falloff").textContent = `${$("s1_falloff_strength").value}%`;
  $("wrap_s1_falloff").classList.toggle("hidden", $("s1_falloff_mode").value === "uniform");

  const method = $("s2_method").value;
  $("val_s2_shake").textContent = parseFloat($("s2_shake_amount").value).toFixed(1);
  $("val_s2_ghost").textContent = `${$("s2_ghost_strength").value}%`;
  $("val_s2_flash").textContent = `${$("s2_flash_sharpness").value}%`;
  $("val_s2_outer").textContent = `${$("s2_outer_smear").value}%`;
  $("val_s2_swing").textContent = `${parseFloat($("s2_swing_angle").value).toFixed(1)}°`;
  $("val_s2_swing_int").textContent = `${$("s2_swing_intensity").value}%`;
  $("val_s2_angle").textContent = `${$("s2_direction_angle").value}°`;

  $("wrap_s2_flash").classList.toggle("hidden", method !== "flash_drag" && method !== "camera_swirl");
  $("wrap_s2_outer").classList.toggle("hidden", method !== "flash_drag");
  $("wrap_s2_swing").classList.toggle("hidden", method !== "pivot_swing" && method !== "camera_swirl");
  $("wrap_s2_swing_int").classList.toggle("hidden", method !== "pivot_swing");
  $("wrap_s2_shake").classList.toggle("hidden", method === "cloud_ai");
  $("wrap_s2_ghost").classList.toggle("hidden", method === "cloud_ai");
  $("wrap_s2_angle").classList.toggle("hidden", method === "cloud_ai" || method === "pivot_swing");

  $("val_s3_glow").textContent = `${$("s3_glow_amount").value}%`;
  $("val_s3_haze").textContent = `${$("s3_haze_lift").value}%`;
  $("val_s3_warmth").textContent = `${$("s3_warmth").value}%`;
  $("val_s3_fringe").textContent = `${$("s3_edge_fringing").value}%`;

  $("val_s4_grain").textContent = `${$("s4_grain_amount").value}%`;
  $("val_s4_clump").textContent = `${$("s4_clumpiness").value}%`;
  $("val_s4_smooth").textContent = `${$("s4_isp_smoothing").value}%`;
  $("val_s4_zoom").textContent = `${$("s4_digital_zoom").value}%`;
  $("val_s4_cast").textContent = `${$("s4_night_cast").value}%`;

  $("val_s5_soft").textContent = `${$("s5_insta_softness").value}%`;
  $("val_s5_qual").textContent = $("s5_jpeg_quality").value;

  const toggles = [
    ["stage1", "s1_enabled"],
    ["stage2", "s2_enabled"],
    ["stage3", "s3_enabled"],
    ["stage4", "s4_enabled"],
    ["stage5", "s5_enabled"],
  ];
  for (const [cardKey, inputId] of toggles) {
    const card = document.querySelector(`[data-stage-card="${cardKey}"]`);
    if (card) card.classList.toggle("disabled", !$(inputId).checked);
  }
}

function scheduleAutoProcess() {
  state.autoMode = false;
  readControlsIntoConfig();
  if (!$("autoApplyCheck").checked || !state.imageId) return;
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(() => {
    processCurrentPhoto();
  }, 340);
}

// ---------- Upload (Always One-Shot Auto-Match + Auto AI Auto-Paint!) ----------
async function uploadFile(file) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  setStatus("Uploading…");

  try {
    const resp = await fetch("/api/upload", { method: "POST", body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Upload failed");
    }
    const data = await resp.json();
    state.imageId = data.image_id;
    state.originalUrl = data.original_url;
    state.filename = data.filename;
    state.previewWidth = data.preview_width;
    state.previewHeight = data.preview_height;
    state.subjectCenter = data.subject_center || [0.50, 0.34];
    state.autoSeedOffset = 0;

    // Always start every new upload in clean Layer-1 Auto-Match mode
    state.autoMode = true;
    state.activePresetId = "auto";
    state.selectedStage = "stage5";
    renderPresetBar();

    // Immediately show Original photo on screen so user never stares at a blank screen
    state.stages = { original: data.original_url, stage5: data.original_url };
    $("origImg").src = data.original_url;
    $("wipeBeforeImg").src = data.original_url;
    $("stageImg").src = data.original_url;
    $("wipeAfterImg").src = data.original_url;

    initBrushCanvases(data.preview_width, data.preview_height);
    $("emptyState").classList.add("hidden");
    $("downloadBtn").disabled = false;
    switchTab(state.activeTab);
    requestAnimationFrame(() => fitStageImages());

    // Check live whether a Vision AI key (e.g. Kie.ai) is configured so one-shot upload uses it automatically!
    const sResp = await fetch("/api/settings");
    const sData = await sResp.json().catch(() => ({}));
    state.visionHasKey = Boolean(sData.vision_has_api_key);
    const shouldUseVision = state.autoRunVision && state.visionHasKey;

    await runAutoMatch(shouldUseVision);
  } catch (err) {
    setStatus("Upload error");
    showToast(err.message);
  }
}

// ---------- Layer 1: Automatic Scene-Matched Pipeline ----------
async function runAutoMatch(useVisionAI = false) {
  if (!state.imageId) return;
  state.isProcessing = true;
  const loadText = $("loadingOverlay").querySelector("span");
  if (loadText) loadText.textContent = useVisionAI ? "✨ AI tracing face & arms…" : "Auto-matching…";
  $("loadingOverlay").classList.remove("hidden");
  setStatus(useVisionAI ? "AI Matching…" : "Auto-Matching…");

  try {
    const resp = await fetch("/api/auto-process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        strength: state.autoStrength,
        seed_offset: state.autoSeedOffset,
        use_ai_vision: Boolean(useVisionAI),
        brush_data_url: useVisionAI ? null : getBrushDataUrl(),
      }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || "Auto-match failed");
    }

    state.autoAnalysis = data.analysis;
    state.config = data.config;
    state.stages = data.stages;
    state.selectedStage = "stage5";

    if (data.brush_data_url) {
      await paintBrushFromDataUrl(data.brush_data_url);
      // Keep mask overlay hidden by default in Layer 1 so the finished result looks clean!
      setMaskVisibility(false);
    }
    if (data.analysis && data.analysis.subject_center) {
      state.subjectCenter = data.analysis.subject_center;
    }

    syncControlsFromConfig();
    renderFilmstrip();
    updateViewerImages();
    requestAnimationFrame(() => fitStageImages());

    const label = data.analysis?.archetype_label || "Auto-Match";
    $("activeLookTitle").textContent = `Auto: ${label}`;
    $("autoBadge").textContent = `✨ ${label}`;
    $("autoBadge").title = data.analysis?.plain_summary || "";

    const sec = (data.total_ms / 1000).toFixed(2);
    setStatus(`${sec}s`);

    const noteBanner = $("noteBanner");
    const extra = (data.notes || []).join(" ");
    const msg = extra || (data.analysis?.plain_summary ? `✨ Detected ${label}: ${data.analysis.plain_summary}` : "");
    if (msg) {
      noteBanner.textContent = msg;
      noteBanner.classList.remove("hidden");
    } else {
      noteBanner.classList.add("hidden");
    }
  } catch (err) {
    setStatus("Error");
    showToast(err.message, 3500);
  } finally {
    state.isProcessing = false;
    $("loadingOverlay").classList.add("hidden");
  }
}

// ---------- Manual / Preset Pipeline ----------
async function processCurrentPhoto() {
  if (!state.imageId || !state.config) return;
  state.isProcessing = true;
  const loadText = $("loadingOverlay").querySelector("span");
  if (loadText) loadText.textContent = "Updating…";
  $("loadingOverlay").classList.remove("hidden");
  setStatus("Updating…");

  try {
    const resp = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        config: state.config,
        brush_data_url: getBrushDataUrl(),
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Could not process photo");
    }
    const data = await resp.json();
    state.stages = data.stages;
    renderFilmstrip();
    updateViewerImages();
    requestAnimationFrame(() => fitStageImages());

    const sec = (data.total_ms / 1000).toFixed(2);
    setStatus(`${sec}s`);

    const noteBanner = $("noteBanner");
    if (data.notes && data.notes.length > 0) {
      noteBanner.textContent = data.notes.join(" ");
      noteBanner.classList.remove("hidden");
    } else if (!state.autoMode) {
      noteBanner.classList.add("hidden");
    }
  } catch (err) {
    setStatus("Error");
    showToast(err.message);
  } finally {
    state.isProcessing = false;
    $("loadingOverlay").classList.add("hidden");
  }
}

// ---------- Bottom Filmstrip ----------
function renderFilmstrip() {
  const strip = $("filmstrip");
  strip.innerHTML = "";

  for (const meta of STAGE_META) {
    const isEnabled = meta.toggleField ? $(meta.toggleField).checked : true;
    const card = document.createElement("div");
    card.className =
      "film-card" +
      (state.selectedStage === meta.key ? " active" : "") +
      (!isEnabled ? " is-off" : "");
    card.title = meta.label;

    const thumbWrap = document.createElement("div");
    thumbWrap.className = "film-thumb-wrap";
    const img = document.createElement("img");
    img.className = "film-thumb";
    img.src = state.stages[meta.key] || "";
    img.alt = meta.label;
    thumbWrap.appendChild(img);

    const info = document.createElement("div");
    info.className = "film-info";

    const title = document.createElement("span");
    title.className = "film-title";
    title.textContent = meta.shortLabel;
    info.appendChild(title);

    if (meta.toggleField) {
      const tgl = document.createElement("button");
      tgl.className = "film-toggle" + (isEnabled ? "" : " off");
      tgl.textContent = isEnabled ? "ON" : "OFF";
      tgl.title = `Turn ${meta.label} ${isEnabled ? "off" : "on"}`;
      tgl.addEventListener("click", (e) => {
        e.stopPropagation();
        const cb = $(meta.toggleField);
        cb.checked = !cb.checked;
        state.autoMode = false;
        readControlsIntoConfig();
        processCurrentPhoto();
      });
      info.appendChild(tgl);
    }

    card.appendChild(thumbWrap);
    card.appendChild(info);

    card.addEventListener("click", () => {
      state.selectedStage = meta.key;
      renderFilmstrip();
      updateViewerImages();
      if (meta.key !== "original") {
        openAccordionStage(meta.key);
      }
    });
    strip.appendChild(card);
  }
}

function updateViewerImages() {
  const origUrl = state.originalUrl || state.stages.original || "";
  const selUrl = state.stages[state.selectedStage] || state.stages.stage5 || origUrl;
  const meta = STAGE_META.find((m) => m.key === state.selectedStage) || STAGE_META[5];

  const badgeText =
    state.selectedStage === "stage5"
      ? (state.autoMode && state.autoAnalysis ? `✨ ${state.autoAnalysis.archetype_label}` : (state.config?.name || "Final Result"))
      : meta.label;

  $("rightBadge").textContent = badgeText;
  $("wipeRightBadge").textContent = badgeText;

  // Keep Original URL stable so the Before image never flickers or reloads unnecessarily
  if (origUrl && $("origImg").getAttribute("src") !== origUrl) {
    $("origImg").src = origUrl;
  }
  if (origUrl && $("wipeBeforeImg").getAttribute("src") !== origUrl) {
    $("wipeBeforeImg").src = origUrl;
  }

  if (!selUrl) return;

  // Preload & decode new After image cleanly before swapping so Wipe Slider & Side-by-Side
  // repaint IMMEDIATELY on the exact frame the new render is ready (no dragging needed!)
  const loader = new Image();
  loader.onload = () => {
    $("stageImg").src = selUrl;
    $("wipeAfterImg").src = selUrl;
    requestAnimationFrame(() => {
      fitStageImages();
      syncWipeInnerDimensions();
    });
  };
  loader.src = selUrl;
}

// ---------- View Modes & Synced 100% Zoom ----------
function setViewMode(mode) {
  state.viewMode = mode;
  $("modeSideBtn").classList.toggle("active", mode === "side");
  $("modeWipeBtn").classList.toggle("active", mode === "wipe");
  $("sideBySideView").classList.toggle("hidden", mode !== "side");
  $("wipeView").classList.toggle("hidden", mode !== "wipe");
  requestAnimationFrame(() => fitStageImages());
}

function toggleZoom() {
  state.zoomed = !state.zoomed;
  $("zoomBtn").classList.toggle("active", state.zoomed);
  $("leftViewport").classList.toggle("zoomed", state.zoomed);
  $("rightViewport").classList.toggle("zoomed", state.zoomed);
  $("wipeViewport").classList.toggle("zoomed", state.zoomed);
  requestAnimationFrame(() => {
    fitStageImages();
    if (state.zoomed) {
      const [fx, fy] = state.subjectCenter;
      const centerZoom = (vp) => {
        if (!vp) return;
        vp.scrollLeft = Math.max(0, state.previewWidth * fx - vp.clientWidth * 0.5);
        vp.scrollTop = Math.max(0, state.previewHeight * fy - vp.clientHeight * 0.5);
      };
      centerZoom($("leftViewport"));
      centerZoom($("rightViewport"));
      centerZoom($("wipeViewport"));
    }
  });
}

function setupSyncedPanning() {
  const left = $("leftViewport");
  const right = $("rightViewport");
  let syncing = false;

  const bindSync = (src, dst) => {
    src.addEventListener("scroll", () => {
      if (syncing) return;
      syncing = true;
      dst.scrollLeft = src.scrollLeft;
      dst.scrollTop = src.scrollTop;
      syncing = false;
    });

    let dragging = false;
    let startX = 0;
    let startY = 0;
    let origLeft = 0;
    let origTop = 0;

    src.addEventListener("mousedown", (e) => {
      if (!state.zoomed || state.brushMode !== "off") return;
      dragging = true;
      startX = e.clientX;
      startY = e.clientY;
      origLeft = src.scrollLeft;
      origTop = src.scrollTop;
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      src.scrollLeft = origLeft - (e.clientX - startX);
      src.scrollTop = origTop - (e.clientY - startY);
    });
    window.addEventListener("mouseup", () => {
      dragging = false;
    });
  };

  bindSync(left, right);
  bindSync(right, left);
}

function setupWipeSlider() {
  const stack = $("wipeStack");
  const handle = $("wipeHandle");
  let dragging = false;

  const updatePct = (clientX) => {
    const rect = stack.getBoundingClientRect();
    if (rect.width <= 0) return;
    state.wipePct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    syncWipeInnerDimensions();
  };

  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    dragging = true;
  });
  stack.addEventListener("mousedown", (e) => {
    dragging = true;
    updatePct(e.clientX);
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    updatePct(e.clientX);
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
  });
}

// ---------- Focus Brush & AI Auto-Paint ----------
function initBrushCanvases(w, h) {
  const c1 = $("brushCanvas");
  const c2 = $("brushMirrorCanvas");
  c1.width = w;
  c1.height = h;
  c2.width = w;
  c2.height = h;
  c1.getContext("2d").clearRect(0, 0, w, h);
  c2.getContext("2d").clearRect(0, 0, w, h);
  state.brushHasStrokes = false;
}

function getBrushDataUrl() {
  if (!state.brushHasStrokes) return null;
  return $("brushCanvas").toDataURL("image/png");
}

function mirrorBrushCanvas() {
  const src = $("brushCanvas");
  const dst = $("brushMirrorCanvas");
  const ctx = dst.getContext("2d");
  ctx.clearRect(0, 0, dst.width, dst.height);
  ctx.drawImage(src, 0, 0);
}

function setMaskVisibility(show) {
  state.showMaskOverlay = show;
  $("brushCanvas").classList.toggle("mask-hidden", !show);
  $("brushMirrorCanvas").classList.toggle("mask-hidden", !show);
  $("toggleMaskBtn").classList.toggle("active-ghost", show);
}

async function paintBrushFromDataUrl(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c1 = $("brushCanvas");
      const ctx = c1.getContext("2d");
      ctx.clearRect(0, 0, c1.width, c1.height);
      ctx.drawImage(img, 0, 0, c1.width, c1.height);
      state.brushHasStrokes = true;
      mirrorBrushCanvas();
      resolve();
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

async function runAIAutoPaint() {
  if (!state.imageId || !state.config) {
    showToast("Upload a photo first.");
    return;
  }

  const sResp = await fetch("/api/settings");
  const sData = await sResp.json();
  if (!sData.vision_has_api_key) {
    showToast("Paste your Kie.ai, Gemini, OpenAI, or OpenRouter API key first.");
    await openSettingsModal();
    $("visionKeyInput").focus();
    return;
  }

  const btn = $("aiAutoPaintBtn");
  const prevText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "✨ Looking…";
  $("loadingOverlay").classList.remove("hidden");
  setStatus("AI directing…");

  try {
    readControlsIntoConfig();
    const applySliders = $("aiTuneSlidersCheck").checked;
    const resp = await fetch("/api/ai-direct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        config: state.config,
        apply_sliders: applySliders,
      }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || "AI Auto-Paint failed");
    }

    if (data.brush_data_url) {
      await paintBrushFromDataUrl(data.brush_data_url);
      setMaskVisibility(state.advancedOpen);
    }

    state.config = data.config;
    if (data.focus_center) state.subjectCenter = data.focus_center;
    syncControlsFromConfig();
    state.stages = data.stages;
    renderFilmstrip();
    updateViewerImages();
    requestAnimationFrame(() => fitStageImages());

    const sec = (data.total_ms / 1000).toFixed(2);
    setStatus(`AI ready (${sec}s)`);

    const noteBanner = $("noteBanner");
    noteBanner.textContent = `✨ ${data.model_used}: ${data.scene_summary} (${data.sharp_regions} sharp, ${data.blur_regions} blur, ${data.pivots_count} arm pivots)`;
    noteBanner.classList.remove("hidden");
    showToast("AI traced face & moving limbs!");
  } catch (err) {
    setStatus("AI error");
    showToast(err.message, 4200);
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
    $("loadingOverlay").classList.add("hidden");
  }
}

function setBrushMode(mode) {
  state.brushMode = mode;
  document.querySelectorAll(".brush-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.brush === mode);
  });
  const canvas = $("brushCanvas");
  canvas.classList.toggle("painting", mode !== "off");
  $("brushHint").classList.toggle("hidden", mode === "off");
  if (mode !== "off") {
    setMaskVisibility(true);
    if (state.viewMode !== "side") setViewMode("side");
  }
}

function setupBrushEvents() {
  const canvas = $("brushCanvas");
  const ctx = canvas.getContext("2d");
  let lastPt = null;

  const getCoords = (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / Math.max(1, rect.width);
    const scaleY = canvas.height / Math.max(1, rect.height);
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  const paintStroke = (from, to) => {
    const size = parseInt($("brushSizeInput").value, 10) * (canvas.width / 800);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = Math.max(10, size);
    ctx.strokeStyle = state.brushMode === "sharp" ? "rgb(34, 220, 94)" : "rgb(240, 58, 58)";
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    state.brushHasStrokes = true;
    mirrorBrushCanvas();
  };

  canvas.addEventListener("mousedown", (e) => {
    if (state.brushMode === "off") return;
    e.preventDefault();
    state.isPainting = true;
    const pt = getCoords(e);
    lastPt = pt;
    paintStroke(pt, { x: pt.x + 0.1, y: pt.y + 0.1 });
  });

  window.addEventListener("mousemove", (e) => {
    if (!state.isPainting || !lastPt) return;
    const pt = getCoords(e);
    paintStroke(lastPt, pt);
    lastPt = pt;
  });

  window.addEventListener("mouseup", () => {
    if (!state.isPainting) return;
    state.isPainting = false;
    lastPt = null;
    if ($("autoApplyCheck").checked && state.imageId) {
      processCurrentPhoto();
    }
  });
}

// ---------- Tab 2: Option Showdown ----------
function findPresetById(id) {
  return allPresets().find((p) => p.id === id);
}

function buildShowdownOptions() {
  const mode = $("showdownModeSelect").value;
  const withSubject = (cfg) => {
    const c = deepClone(cfg);
    c.stage1_focus.focus_center_x = state.subjectCenter[0];
    c.stage1_focus.focus_center_y = state.subjectCenter[1];
    return c;
  };

  if (mode === "vibes") {
    const p1 = findPresetById("flash_party_blur");
    const p2 = findPresetById("club_whip_swirl");
    const p3 = findPresetById("warm_night_missed_focus");
    return [
      { label: "Option A — Flash Party", subtitle: "Crisp flash face + subtle ghosting & night glow", config: withSubject(p1.config) },
      { label: "Option B — Club Whip Swirl", subtitle: "Face anchored while lights & room swirl in an arc", config: withSubject(p2.config) },
      { label: "Option C — Warm Night Focus", subtitle: "Dreamy circular missed focus + golden glow", config: withSubject(p3.config) },
    ];
  }

  if (mode === "casual") {
    const p1 = findPresetById("quick_gym_mirror_snap");
    const p2 = findPresetById("daylight_candid_snap");
    const p3 = findPresetById("moving_hands_party");
    return [
      { label: "Option A — Mirror Selfie", subtitle: "Subtle curved thumb-press shake + clean glass sheen", config: withSubject(p1.config) },
      { label: "Option B — Daylight Candid", subtitle: "Rich blacks, crisp face, spontaneous outer motion", config: withSubject(p2.config) },
      { label: "Option C — Moving Hands", subtitle: "Crisp face & background, swinging arms & hands", config: withSubject(p3.config) },
    ];
  }

  // 3 Motion styles on current config
  const base = withSubject(state.config);
  const optA = deepClone(base);
  optA.name = "Handheld Micro-Shake";
  optA.stage2_motion.enabled = true;
  optA.stage2_motion.method = "handheld_shake";
  optA.stage2_motion.shake_amount = Math.max(6, base.stage2_motion.shake_amount);

  const optB = deepClone(base);
  optB.name = "Flash Party Drag";
  optB.stage2_motion.enabled = true;
  optB.stage2_motion.method = "flash_drag";
  optB.stage2_motion.shake_amount = Math.max(9.5, base.stage2_motion.shake_amount);

  const optC = deepClone(base);
  optC.name = "Moving Hands Swing";
  optC.stage2_motion.enabled = true;
  optC.stage2_motion.method = "pivot_swing";
  optC.stage2_motion.swing_angle = Math.max(8.0, base.stage2_motion.swing_angle || 8.5);
  optC.stage2_motion.swing_intensity = Math.max(0.78, base.stage2_motion.swing_intensity || 0.8);

  return [
    { label: "Option A — Handheld Shake", subtitle: "Whole-frame curved thumb shake + subtle ghosting", config: optA },
    { label: "Option B — Flash Party Drag", subtitle: "Crisp flash face + trailing background & edges", config: optB },
    { label: "Option C — Moving Hands Swing", subtitle: "Sharp face + swinging arms & hands", config: optC },
  ];
}

async function runShowdown() {
  if (!state.imageId || !state.config) return;
  const grid = $("showdownGrid");
  grid.innerHTML = `<div style="grid-column:1/-1;display:flex;align-items:center;justify-content:center;gap:10px;color:var(--text-muted);"><div class="spinner"></div><span>Building 3 showdown options…</span></div>`;
  setStatus("Showdown…");

  try {
    const resp = await fetch("/api/showdown", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        options: buildShowdownOptions(),
        brush_data_url: getBrushDataUrl(),
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Showdown failed");
    }
    const data = await resp.json();
    state.showdownResults = data;
    renderShowdownGrid(data);
    setStatus(`${(data.total_ms / 1000).toFixed(2)}s`);
  } catch (err) {
    grid.innerHTML = "";
    showToast(err.message);
    setStatus("Error");
  }
}

function renderShowdownGrid(data) {
  const grid = $("showdownGrid");
  grid.innerHTML = "";

  const origCard = document.createElement("div");
  origCard.className = "showdown-card";
  origCard.innerHTML = `
    <div class="showdown-card-head">
      <div class="showdown-card-title">Original</div>
      <div class="showdown-card-sub">Before blur &amp; night optics</div>
    </div>
    <div class="showdown-img-wrap ${state.showdownZoomed ? "zoomed" : ""}">
      <img src="${data.original_url}" alt="Original" />
    </div>
    <div class="showdown-card-foot">
      <span class="showdown-time">Reference</span>
    </div>
  `;
  grid.appendChild(origCard);

  for (const opt of data.options) {
    const card = document.createElement("div");
    card.className = "showdown-card";
    card.innerHTML = `
      <div class="showdown-card-head">
        <div class="showdown-card-title">${opt.label}</div>
        <div class="showdown-card-sub">${opt.subtitle}</div>
      </div>
      <div class="showdown-img-wrap ${state.showdownZoomed ? "zoomed" : ""}">
        <img src="${opt.image_url}" alt="${opt.label}" />
      </div>
      <div class="showdown-card-foot">
        <span class="showdown-time">${(opt.total_ms / 1000).toFixed(2)}s</span>
        <button class="btn btn-primary btn-xs pick-btn">Pick Winner</button>
      </div>
    `;
    card.querySelector(".pick-btn").addEventListener("click", () => {
      state.autoMode = false;
      state.activePresetId = "";
      state.config = deepClone(opt.config);
      $("activeLookTitle").textContent = opt.label;
      $("autoBadge").textContent = opt.label.replace(/^Option [A-C] — /, "Look: ");
      syncControlsFromConfig();
      renderPresetBar();
      switchTab("qa");
      processCurrentPhoto();
      showToast(`Selected "${opt.label}"!`);
    });
    grid.appendChild(card);
  }
}

// ---------- Tabs ----------
function switchTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".tab-btn").forEach((b) => {
    const isAct = b.dataset.tab === tab;
    b.classList.toggle("active", isAct);
    b.setAttribute("aria-selected", String(isAct));
  });

  if (!state.imageId) {
    $("tab-qa").classList.add("hidden");
    $("tab-showdown").classList.add("hidden");
    return;
  }

  $("tab-qa").classList.toggle("hidden", tab !== "qa");
  $("tab-showdown").classList.toggle("hidden", tab !== "showdown");

  if (tab === "qa") {
    requestAnimationFrame(() => fitStageImages());
  } else if (tab === "showdown") {
    runShowdown();
  }
}

// ---------- Download Full-Resolution Result ----------
async function downloadFinalResult() {
  if (!state.imageId || !state.config) return;
  const btn = $("downloadBtn");
  btn.disabled = true;
  btn.textContent = "Saving…";
  try {
    const resp = await fetch("/api/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.imageId,
        config: state.config,
        brush_data_url: getBrushDataUrl(),
      }),
    });
    if (!resp.ok) throw new Error("Download failed");
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const cd = resp.headers.get("Content-Disposition") || "";
    const match = cd.match(/filename="([^"]+)"/);
    a.href = url;
    a.download = match ? match[1] : "influencer_blur_result.jpg";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    showToast("Downloaded full-resolution photo!");
  } catch (err) {
    showToast(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Download";
  }
}

// ---------- Modals ----------
function populateVisionModelsDropdown(models, selectedId) {
  const sel = $("visionModelSelect");
  sel.innerHTML = "";
  let found = false;
  for (const m of models || []) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label || m.id;
    if (m.id === selectedId) found = true;
    sel.appendChild(opt);
  }
  if (found) {
    sel.value = selectedId;
    $("visionCustomModelInput").value = "";
  } else if (selectedId) {
    $("visionCustomModelInput").value = selectedId;
  }
}

async function openSettingsModal() {
  const resp = await fetch("/api/settings");
  const data = await resp.json();
  state.suggestedModels = data.suggested_models || {};
  state.visionHasKey = Boolean(data.vision_has_api_key);

  const vProv = data.vision_provider || "kie";
  $("visionProviderSelect").value = vProv;
  $("visionKeyInput").value = data.vision_has_api_key ? "__KEEP__" : "";
  $("visionKeyStatus").textContent = data.vision_has_api_key
    ? `Saved key: ${data.vision_api_key_preview} (clear box to remove)`
    : "Paste your Kie.ai, Gemini, OpenAI, or OpenRouter key to enable ✨ AI Auto-Paint.";
  populateVisionModelsDropdown(state.suggestedModels[vProv] || [], data.vision_model);
  $("autoRunVisionCheck").checked = state.autoRunVision;

  $("cloudProviderSelect").value = data.provider || "fal";
  $("cloudKeyInput").value = data.has_api_key ? "__KEEP__" : "";
  $("cloudKeyStatus").textContent = data.has_api_key
    ? `Saved key: ${data.api_key_preview} (clear box to remove)`
    : "No Cloud Motion key set — running 100% locally on your PC.";
  $("cloudModelInput").value = data.model || "";
  $("settingsModal").classList.remove("hidden");
}

async function fetchLiveVisionModels() {
  const btn = $("fetchModelsBtn");
  const prev = btn.textContent;
  btn.textContent = "Loading…";
  try {
    const provider = $("visionProviderSelect").value;
    const keyVal = $("visionKeyInput").value;
    const resp = await fetch("/api/ai-models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, api_key: keyVal }),
    });
    const data = await resp.json();
    const cur = $("visionCustomModelInput").value.trim() || $("visionModelSelect").value;
    populateVisionModelsDropdown(data.models || [], cur);
    showToast(`Loaded ${(data.models || []).length} models for ${provider}`);
  } catch (err) {
    showToast("Could not fetch live models — check your API key.");
  } finally {
    btn.textContent = prev;
  }
}

async function saveSettings() {
  const customModel = $("visionCustomModelInput").value.trim();
  const chosenModel = customModel || $("visionModelSelect").value;
  state.autoRunVision = $("autoRunVisionCheck").checked;
  localStorage.setItem("ibl_auto_vision", state.autoRunVision ? "1" : "0");

  const resp = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      vision_provider: $("visionProviderSelect").value,
      vision_api_key: $("visionKeyInput").value,
      vision_model: chosenModel,
      provider: $("cloudProviderSelect").value,
      api_key: $("cloudKeyInput").value,
      model: $("cloudModelInput").value,
    }),
  });
  if (resp.ok) {
    $("settingsModal").classList.add("hidden");
    showToast("AI Settings saved!");
  }
}

async function saveCustomPreset() {
  const name = $("presetNameInput").value.trim();
  if (!name) {
    showToast("Please type a name for your preset.");
    return;
  }
  readControlsIntoConfig();
  const resp = await fetch("/api/presets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      description: $("presetDescInput").value.trim(),
      config: state.config,
    }),
  });
  if (!resp.ok) {
    showToast("Could not save preset.");
    return;
  }
  const saved = await resp.json();
  $("presetModal").classList.add("hidden");
  $("presetNameInput").value = "";
  $("presetDescInput").value = "";
  await loadPresets();
  state.autoMode = false;
  state.activePresetId = saved.id;
  $("activeLookTitle").textContent = saved.name;
  renderPresetBar();
  showToast(`Saved "${saved.name}"!`);
}

// ---------- Bootstrap ----------
function bindEvents() {
  setupAccordion();

  $("toggleAdvancedBtn").addEventListener("click", () => setAdvancedOpen(!state.advancedOpen));

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  $("fileInput").addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0]);
      e.target.value = "";
    }
  });

  const dz = $("dropzone");
  ["dragenter", "dragover"].forEach((ev) => {
    document.body.addEventListener(ev, (e) => {
      e.preventDefault();
      if (dz) dz.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    document.body.addEventListener(ev, (e) => {
      e.preventDefault();
      if (dz) dz.classList.remove("drag-over");
    });
  });
  document.body.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      uploadFile(file);
    }
  });

  // Layer 1 Master Strength Slider & Quick Re-roll
  $("autoStrengthSlider").addEventListener("input", () => {
    const pct = parseInt($("autoStrengthSlider").value, 10);
    $("autoStrengthVal").textContent = `${pct}%`;
    state.autoStrength = pct / 100.0;
    if (!state.imageId) return;
    clearTimeout(state.strengthTimer);
    state.strengthTimer = setTimeout(() => {
      if (state.autoMode) {
        runAutoMatch(false);
      } else {
        const chosen = allPresets().find((p) => p.id === state.activePresetId);
        if (chosen) applyPresetToState(chosen, true);
      }
    }, 280);
  });

  $("quickRerollBtn").addEventListener("click", () => {
    if (!state.imageId) return;
    if (state.autoMode) {
      state.autoSeedOffset += 137;
      runAutoMatch(false);
    } else if (state.config) {
      state.config.stage2_motion.seed = Math.floor(Math.random() * 99999) + 1;
      processCurrentPhoto();
    }
  });

  $("modeSideBtn").addEventListener("click", () => setViewMode("side"));
  $("modeWipeBtn").addEventListener("click", () => setViewMode("wipe"));
  $("zoomBtn").addEventListener("click", toggleZoom);

  document.querySelectorAll(".brush-btn").forEach((btn) => {
    btn.addEventListener("click", () => setBrushMode(btn.dataset.brush));
  });
  $("aiAutoPaintBtn").addEventListener("click", runAIAutoPaint);
  $("toggleMaskBtn").addEventListener("click", () => setMaskVisibility(!state.showMaskOverlay));
  $("clearBrushBtn").addEventListener("click", () => {
    if (!state.previewWidth) return;
    initBrushCanvases(state.previewWidth, state.previewHeight);
    if (state.config && state.config.stage2_motion) {
      state.config.stage2_motion.ai_pivots = [];
    }
    showToast("Cleared painted brush");
    if (state.imageId) processCurrentPhoto();
  });

  $("visionProviderSelect").addEventListener("change", () => {
    const prov = $("visionProviderSelect").value;
    populateVisionModelsDropdown(state.suggestedModels[prov] || [], "");
  });
  $("fetchModelsBtn").addEventListener("click", fetchLiveVisionModels);

  const controlIds = [
    "s1_enabled", "s1_radius", "s1_falloff_mode", "s1_falloff_strength",
    "s2_enabled", "s2_method", "s2_shake_amount", "s2_ghost_strength",
    "s2_flash_sharpness", "s2_outer_smear", "s2_swing_angle", "s2_swing_intensity",
    "s2_direction_angle",
    "s3_enabled", "s3_glow_amount", "s3_haze_lift", "s3_warmth", "s3_edge_fringing",
    "s4_enabled", "s4_grain_amount", "s4_clumpiness", "s4_isp_smoothing", "s4_digital_zoom", "s4_night_cast",
    "s5_enabled", "s5_insta_softness", "s5_jpeg_quality",
  ];
  for (const id of controlIds) {
    const el = $(id);
    const ev = el.type === "range" ? "input" : "change";
    el.addEventListener(ev, scheduleAutoProcess);
  }

  $("applyBtn").addEventListener("click", () => {
    state.autoMode = false;
    readControlsIntoConfig();
    processCurrentPhoto();
  });

  $("rerollShakeBtn").addEventListener("click", () => {
    if (!state.config) return;
    state.config.stage2_motion.seed = Math.floor(Math.random() * 99999) + 1;
    processCurrentPhoto();
  });

  $("runShowdownBtn").addEventListener("click", runShowdown);
  $("showdownModeSelect").addEventListener("change", () => {
    if (state.imageId && state.activeTab === "showdown") runShowdown();
  });
  $("showdownZoomBtn").addEventListener("click", () => {
    state.showdownZoomed = !state.showdownZoomed;
    $("showdownZoomBtn").classList.toggle("active", state.showdownZoomed);
    document.querySelectorAll(".showdown-img-wrap").forEach((el) => {
      el.classList.toggle("zoomed", state.showdownZoomed);
    });
  });

  $("downloadBtn").addEventListener("click", downloadFinalResult);

  $("savePresetBtn").addEventListener("click", () => {
    $("presetModal").classList.remove("hidden");
    $("presetNameInput").focus();
  });
  $("confirmSavePresetBtn").addEventListener("click", saveCustomPreset);
  $("settingsBtn").addEventListener("click", openSettingsModal);
  $("saveSettingsBtn").addEventListener("click", saveSettings);
  document.querySelectorAll("[data-close-modal]").forEach((btn) => {
    btn.addEventListener("click", () => $(btn.dataset.closeModal).classList.add("hidden"));
  });

  window.addEventListener("resize", () => requestAnimationFrame(() => fitStageImages()));
  if (typeof ResizeObserver !== "undefined") {
    const ro = new ResizeObserver(() => requestAnimationFrame(() => fitStageImages()));
    ro.observe($("viewerStage"));
  }

  window.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (!state.imageId) return;
    if (e.key === "z" || e.key === "Z") {
      toggleZoom();
    } else if (e.key >= "1" && e.key <= "5" && state.advancedOpen) {
      const k = `stage${e.key}`;
      state.selectedStage = k;
      renderFilmstrip();
      updateViewerImages();
      openAccordionStage(k);
    }
  });

  setupSyncedPanning();
  setupWipeSlider();
  setupBrushEvents();
}

window.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  await loadPresets();
});
