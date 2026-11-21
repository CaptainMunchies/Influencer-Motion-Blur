/* Influencer Blur Lab — Plain-English Interactive Visual QA Studio */

const STAGE_META = [
  { key: "original", label: "Original", toggleField: null },
  { key: "stage1", label: "1. Missed Focus", toggleField: "s1_enabled" },
  { key: "stage2", label: "2. Motion & Flash", toggleField: "s2_enabled" },
  { key: "stage3", label: "3. Dirty Lens Glow", toggleField: "s3_enabled" },
  { key: "stage4", label: "4. Nighttime Grain", toggleField: "s4_enabled" },
  { key: "stage5", label: "5. Instagram Polish", toggleField: "s5_enabled" },
];

const state = {
  activeTab: "qa",
  imageId: null,
  filename: "",
  previewWidth: 0,
  previewHeight: 0,
  presets: { builtin: [], custom: [] },
  activePresetId: "flash_party_blur",
  config: null,
  stages: {},
  selectedStage: "stage5",
  viewMode: "side", // "side" | "wipe"
  zoomed: false,
  showdownZoomed: false,
  brushMode: "off", // "off" | "sharp" | "blur"
  brushHasStrokes: false,
  isPainting: false,
  isProcessing: false,
  debounceTimer: null,
  showdownResults: null,
};

// ---------- DOM Helpers ----------
const $ = (id) => document.getElementById(id);

function showToast(msg, duration = 2600) {
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

// ---------- Presets ----------
async function loadPresets() {
  const resp = await fetch("/api/presets");
  const data = await resp.json();
  state.presets = data;
  renderPresetBar();

  if (!state.config && data.builtin.length > 0) {
    applyPresetToState(data.builtin[0], false);
  }
}

function allPresets() {
  return [...state.presets.builtin, ...state.presets.custom];
}

function renderPresetBar() {
  const container = $("presetList");
  container.innerHTML = "";
  for (const p of allPresets()) {
    const btn = document.createElement("button");
    btn.className = "preset-pill" + (state.activePresetId === p.id ? " active" : "");
    btn.title = p.description || p.name;

    const span = document.createElement("span");
    span.textContent = p.name;
    btn.appendChild(span);

    if (!p.builtin) {
      const del = document.createElement("span");
      del.className = "preset-del";
      del.textContent = "✕";
      del.title = "Delete saved preset";
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        await fetch(`/api/presets/${encodeURIComponent(p.id)}`, { method: "DELETE" });
        showToast(`Deleted preset "${p.name}"`);
        await loadPresets();
      });
      btn.appendChild(del);
    }

    btn.addEventListener("click", () => {
      applyPresetToState(p, true);
    });
    container.appendChild(btn);
  }
}

function applyPresetToState(preset, triggerProcess = true) {
  state.activePresetId = preset.id;
  state.config = deepClone(preset.config);
  $("activeLookTitle").textContent = preset.name;
  $("activeLookDesc").textContent = preset.description || "Custom saved look";
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

  // Stage 1
  $("s1_enabled").checked = c.stage1_focus.enabled;
  $("s1_radius").value = c.stage1_focus.radius;
  $("s1_falloff_mode").value = c.stage1_focus.falloff_mode;
  $("s1_falloff_strength").value = Math.round(c.stage1_focus.falloff_strength * 100);

  // Stage 2
  $("s2_enabled").checked = c.stage2_motion.enabled;
  $("s2_method").value = c.stage2_motion.method;
  $("s2_shake_amount").value = c.stage2_motion.shake_amount;
  $("s2_ghost_strength").value = Math.round(c.stage2_motion.ghost_strength * 100);
  $("s2_flash_sharpness").value = Math.round(c.stage2_motion.flash_sharpness * 100);
  $("s2_outer_smear").value = Math.round(c.stage2_motion.outer_smear * 100);
  $("s2_swing_angle").value = c.stage2_motion.swing_angle;
  $("s2_swing_intensity").value = Math.round(c.stage2_motion.swing_intensity * 100);
  $("s2_direction_angle").value = Math.round(c.stage2_motion.direction_angle);

  // Stage 3
  $("s3_enabled").checked = c.stage3_optics.enabled;
  $("s3_glow_amount").value = Math.round(c.stage3_optics.glow_amount * 100);
  $("s3_haze_lift").value = Math.round(c.stage3_optics.haze_lift * 100);
  $("s3_warmth").value = Math.round(c.stage3_optics.warmth * 100);
  $("s3_edge_fringing").value = Math.round(c.stage3_optics.edge_fringing * 100);

  // Stage 4
  $("s4_enabled").checked = c.stage4_grain.enabled;
  $("s4_grain_amount").value = Math.round(c.stage4_grain.grain_amount * 100);
  $("s4_clumpiness").value = Math.round(c.stage4_grain.clumpiness * 100);
  $("s4_isp_smoothing").value = Math.round(c.stage4_grain.isp_smoothing * 100);

  // Stage 5
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

  $("wrap_s2_flash").classList.toggle("hidden", method !== "flash_drag");
  $("wrap_s2_outer").classList.toggle("hidden", method !== "flash_drag");
  $("wrap_s2_swing").classList.toggle("hidden", method !== "pivot_swing");
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

  $("val_s5_soft").textContent = `${$("s5_insta_softness").value}%`;
  $("val_s5_qual").textContent = $("s5_jpeg_quality").value;

  // Dim disabled stage cards
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
  readControlsIntoConfig();
  if (!$("autoApplyCheck").checked || !state.imageId) return;
  clearTimeout(state.debounceTimer);
  state.debounceTimer = setTimeout(() => {
    processCurrentPhoto();
  }, 360);
}

// ---------- Upload ----------
async function uploadFile(file) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  setStatus("Uploading photo…");

  try {
    const resp = await fetch("/api/upload", { method: "POST", body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Upload failed");
    }
    const data = await resp.json();
    state.imageId = data.image_id;
    state.filename = data.filename;
    state.previewWidth = data.preview_width;
    state.previewHeight = data.preview_height;

    initBrushCanvases(data.preview_width, data.preview_height);
    $("emptyState").classList.add("hidden");
    $("downloadBtn").disabled = false;
    switchTab(state.activeTab);

    await processCurrentPhoto();
    showToast(`Loaded "${data.filename}" (${data.full_width}×${data.full_height})`);
  } catch (err) {
    setStatus("Upload error");
    showToast(err.message);
  }
}

// ---------- Processing Pipeline ----------
async function processCurrentPhoto() {
  if (!state.imageId || !state.config) return;
  state.isProcessing = true;
  $("loadingOverlay").classList.remove("hidden");
  setStatus("Processing…");

  try {
    const payload = {
      image_id: state.imageId,
      config: state.config,
      brush_data_url: getBrushDataUrl(),
    };
    const resp = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || "Could not process photo");
    }
    const data = await resp.json();
    state.stages = data.stages;
    renderFilmstrip();
    updateViewerImages();

    const sec = (data.total_ms / 1000).toFixed(2);
    setStatus(`Ready in ${sec}s`);

    const noteBanner = $("noteBanner");
    if (data.notes && data.notes.length > 0) {
      noteBanner.textContent = data.notes.join(" ");
      noteBanner.classList.remove("hidden");
    } else {
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

// ---------- Filmstrip ----------
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

    const thumbWrap = document.createElement("div");
    thumbWrap.className = "film-thumb-wrap";
    const img = document.createElement("img");
    img.className = "film-thumb";
    img.src = state.stages[meta.key] || "";
    img.alt = meta.label;
    thumbWrap.appendChild(img);

    const row = document.createElement("div");
    row.className = "film-meta";

    const title = document.createElement("span");
    title.className = "film-title";
    title.textContent = meta.label;
    row.appendChild(title);

    if (meta.toggleField) {
      const tgl = document.createElement("button");
      tgl.className = "film-toggle" + (isEnabled ? "" : " off");
      tgl.textContent = isEnabled ? "ON" : "OFF";
      tgl.title = `Turn ${meta.label} ${isEnabled ? "off" : "on"}`;
      tgl.addEventListener("click", (e) => {
        e.stopPropagation();
        const cb = $(meta.toggleField);
        cb.checked = !cb.checked;
        readControlsIntoConfig();
        processCurrentPhoto();
      });
      row.appendChild(tgl);
    }

    card.appendChild(thumbWrap);
    card.appendChild(row);

    card.addEventListener("click", () => {
      state.selectedStage = meta.key;
      renderFilmstrip();
      updateViewerImages();
    });
    strip.appendChild(card);
  }
}

function updateViewerImages() {
  const origUrl = state.stages.original || "";
  const selUrl = state.stages[state.selectedStage] || state.stages.stage5 || origUrl;
  const meta = STAGE_META.find((m) => m.key === state.selectedStage) || STAGE_META[5];

  $("origImg").src = origUrl;
  $("stageImg").src = selUrl;
  $("rightBadge").textContent = meta.label;

  $("wipeBeforeImg").src = origUrl;
  $("wipeAfterImg").src = selUrl;
  $("wipeRightBadge").textContent = meta.label;
}

// ---------- View Modes & Synced 100% Zoom ----------
function setViewMode(mode) {
  state.viewMode = mode;
  $("modeSideBtn").classList.toggle("active", mode === "side");
  $("modeWipeBtn").classList.toggle("active", mode === "wipe");
  $("sideBySideView").classList.toggle("hidden", mode !== "side");
  $("wipeView").classList.toggle("hidden", mode !== "wipe");
}

function toggleZoom() {
  state.zoomed = !state.zoomed;
  $("zoomBtn").classList.toggle("active", state.zoomed);
  $("leftViewport").classList.toggle("zoomed", state.zoomed);
  $("rightViewport").classList.toggle("zoomed", state.zoomed);
  $("wipeViewport").classList.toggle("zoomed", state.zoomed);
}

// Sync scrolling between left & right panes when in 100% Zoom
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

// Interactive Wipe Slider
function setupWipeSlider() {
  const stack = $("wipeStack");
  const clip = $("wipeClip");
  const handle = $("wipeHandle");
  let dragging = false;

  const updatePct = (clientX) => {
    const rect = stack.getBoundingClientRect();
    if (rect.width <= 0) return;
    const pct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
    clip.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    handle.style.left = `${pct}%`;
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

// ---------- Sharp / Blur Focus Brush ----------
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

function setBrushMode(mode) {
  state.brushMode = mode;
  document.querySelectorAll(".brush-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.brush === mode);
  });
  const canvas = $("brushCanvas");
  canvas.classList.toggle("painting", mode !== "off");
  $("brushHint").classList.toggle("hidden", mode === "off");
  if (mode !== "off" && state.viewMode !== "side") {
    setViewMode("side");
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
function buildShowdownOptions() {
  const mode = $("showdownModeSelect").value;
  if (mode === "presets" && state.presets.builtin.length >= 3) {
    return state.presets.builtin.slice(0, 3).map((p, idx) => ({
      label: `Option ${["A", "B", "C"][idx]} — ${p.name}`,
      subtitle: p.description,
      config: deepClone(p.config),
    }));
  }

  // Default: compare all 3 motion methods on current look settings
  const base = deepClone(state.config);
  const optA = deepClone(base);
  optA.name = "Handheld Micro-Shake";
  optA.stage2_motion.enabled = true;
  optA.stage2_motion.method = "handheld_shake";
  optA.stage2_motion.shake_amount = Math.max(8, base.stage2_motion.shake_amount);

  const optB = deepClone(base);
  optB.name = "Slow-Sync Flash Drag";
  optB.stage2_motion.enabled = true;
  optB.stage2_motion.method = "flash_drag";
  optB.stage2_motion.shake_amount = Math.max(12, base.stage2_motion.shake_amount);

  const optC = deepClone(base);
  optC.name = "Moving Hands Swing";
  optC.stage2_motion.enabled = true;
  optC.stage2_motion.method = "pivot_swing";
  optC.stage2_motion.swing_angle = Math.max(7.5, base.stage2_motion.swing_angle || 8.5);
  optC.stage2_motion.swing_intensity = Math.max(0.78, base.stage2_motion.swing_intensity || 0.8);

  return [
    {
      label: "Option A — Handheld Shake",
      subtitle: "Whole-frame curved shutter shake with subtle double-image ghosting",
      config: optA,
    },
    {
      label: "Option B — Flash Party Drag",
      subtitle: "Face stays crisp from flash while background & outer edges trail",
      config: optB,
    },
    {
      label: "Option C — Moving Hands Swing",
      subtitle: "Face stays sharp while arms & hands arc naturally in motion",
      config: optC,
    },
  ];
}

async function runShowdown() {
  if (!state.imageId || !state.config) return;
  const grid = $("showdownGrid");
  grid.innerHTML = `<div style="grid-column:1/-1;display:flex;align-items:center;justify-content:center;gap:12px;color:var(--text-muted);"><div class="spinner"></div><span>Building 3 showdown options…</span></div>`;
  setStatus("Running Showdown…");

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
    setStatus(`Showdown ready in ${(data.total_ms / 1000).toFixed(2)}s`);
  } catch (err) {
    grid.innerHTML = "";
    showToast(err.message);
    setStatus("Showdown error");
  }
}

function renderShowdownGrid(data) {
  const grid = $("showdownGrid");
  grid.innerHTML = "";

  // Card 0: Original
  const origCard = document.createElement("div");
  origCard.className = "showdown-card";
  origCard.innerHTML = `
    <div class="showdown-card-head">
      <div class="showdown-card-title">Original Photo</div>
      <div class="showdown-card-sub">Reference before any blur or night optics</div>
    </div>
    <div class="showdown-img-wrap ${state.showdownZoomed ? "zoomed" : ""}">
      <img src="${data.original_url}" alt="Original" />
    </div>
    <div class="showdown-card-foot">
      <span class="showdown-time">Unedited</span>
    </div>
  `;
  grid.appendChild(origCard);

  // Cards A, B, C
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
        <button class="btn btn-primary btn-sm pick-btn">Pick Winner</button>
      </div>
    `;
    card.querySelector(".pick-btn").addEventListener("click", () => {
      state.config = deepClone(opt.config);
      state.activePresetId = "";
      $("activeLookTitle").textContent = opt.label;
      $("activeLookDesc").textContent = opt.subtitle;
      syncControlsFromConfig();
      renderPresetBar();
      switchTab("qa");
      processCurrentPhoto();
      showToast(`Loaded "${opt.label}" into Step-by-Step QA!`);
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

  if (tab === "showdown") {
    runShowdown();
  }
}

// ---------- Download Full-Resolution Result ----------
async function downloadFinalResult() {
  if (!state.imageId || !state.config) return;
  const btn = $("downloadBtn");
  btn.disabled = true;
  btn.textContent = "Preparing Full-Res…";
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
    btn.textContent = "Download Final Result";
  }
}

// ---------- Modals (Save Preset & Settings) ----------
async function openSettingsModal() {
  const resp = await fetch("/api/settings");
  const data = await resp.json();
  $("cloudProviderSelect").value = data.provider || "fal";
  $("cloudKeyInput").value = data.has_api_key ? "__KEEP__" : "";
  $("cloudKeyStatus").textContent = data.has_api_key
    ? `Saved key: ${data.api_key_preview} (clear field to remove)`
    : "No API key set — running 100% locally on your PC.";
  $("cloudModelInput").value = data.model || "";
  $("settingsModal").classList.remove("hidden");
}

async function saveSettings() {
  const resp = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      provider: $("cloudProviderSelect").value,
      api_key: $("cloudKeyInput").value,
      model: $("cloudModelInput").value,
    }),
  });
  if (resp.ok) {
    $("settingsModal").classList.add("hidden");
    showToast("Settings saved!");
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
  state.activePresetId = saved.id;
  $("activeLookTitle").textContent = saved.name;
  renderPresetBar();
  showToast(`Saved preset "${saved.name}"!`);
}

// ---------- Bootstrap ----------
function bindEvents() {
  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // File input + drag & drop anywhere
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
      dz.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    document.body.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.remove("drag-over");
    });
  });
  document.body.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      uploadFile(file);
    }
  });

  // View controls
  $("modeSideBtn").addEventListener("click", () => setViewMode("side"));
  $("modeWipeBtn").addEventListener("click", () => setViewMode("wipe"));
  $("zoomBtn").addEventListener("click", toggleZoom);

  // Brush controls
  document.querySelectorAll(".brush-btn").forEach((btn) => {
    btn.addEventListener("click", () => setBrushMode(btn.dataset.brush));
  });
  $("clearBrushBtn").addEventListener("click", () => {
    if (!state.previewWidth) return;
    initBrushCanvases(state.previewWidth, state.previewHeight);
    showToast("Cleared painted focus brush");
    if (state.imageId) processCurrentPhoto();
  });

  // Controls inputs
  const controlIds = [
    "s1_enabled", "s1_radius", "s1_falloff_mode", "s1_falloff_strength",
    "s2_enabled", "s2_method", "s2_shake_amount", "s2_ghost_strength",
    "s2_flash_sharpness", "s2_outer_smear", "s2_swing_angle", "s2_swing_intensity",
    "s2_direction_angle",
    "s3_enabled", "s3_glow_amount", "s3_haze_lift", "s3_warmth", "s3_edge_fringing",
    "s4_enabled", "s4_grain_amount", "s4_clumpiness", "s4_isp_smoothing",
    "s5_enabled", "s5_insta_softness", "s5_jpeg_quality",
  ];
  for (const id of controlIds) {
    const el = $(id);
    const ev = el.type === "range" ? "input" : "change";
    el.addEventListener(ev, scheduleAutoProcess);
  }

  $("applyBtn").addEventListener("click", () => {
    readControlsIntoConfig();
    processCurrentPhoto();
  });

  $("rerollShakeBtn").addEventListener("click", () => {
    if (!state.config) return;
    state.config.stage2_motion.seed = Math.floor(Math.random() * 99999) + 1;
    processCurrentPhoto();
  });

  // Showdown
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

  // Download
  $("downloadBtn").addEventListener("click", downloadFinalResult);

  // Modals
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

  setupSyncedPanning();
  setupWipeSlider();
  setupBrushEvents();
}

window.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  await loadPresets();
});
