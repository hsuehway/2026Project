(() => {
  "use strict";

  const token = new URLSearchParams(location.search).get("token") || "";
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const state = { session: null, sessionSource: "default", currentPage: "home", library: null, job: null, poller: null, saveTimer: null, activeResultKey: "", qualityPath: "", orbitSummaryPath: "", subviews: { track: "initialize", review: "validate", results: "analyze" }, seeds: { front: [], side: [] } };

  const artifactNames = {
    preview_video: "Background-subtraction preview", preview_metadata: "Preview metadata",
    tracks_csv: "3-D trajectories", track_summary: "Tracking summary", track_plot: "Trajectory projections",
    corrected_tracks_csv: "Identity-corrected trajectories", track_verification_raw: "Raw trajectory overlay",
    track_verification_smoothed: "Smoothed trajectory overlay", motion_csv: "Position, velocity & acceleration",
    translational_state_csv: "Translational state vectors", translational_overlays: "Translational state overlay",
    rotation_state_csv: "Rotation state vectors", reduced_rotation_state_csv: "Reduced full state vectors",
    rotation_diagnostics: "Rotation diagnostics", rotation_overlay: "Full rotational overlay",
    orbit_intervals_csv: "Circular-motion intervals", orbit_timeseries_csv: "Orbit-derived time series",
    orbit_summary_json: "Circular-motion summary", orbit_diagnostic_plot: "Orbit detection diagnostic",
    orbit_xy_plot: "Tracked path diagnostic", orbit_report_html: "Circular-motion HTML report",
    orbit_source_video: "Browser-compatible source video",
    run_manifest: "Reproducibility manifest"
  };

  class SeedCanvas {
    constructor(name) {
      this.name = name;
      this.canvas = $(`#${name}Canvas`);
      this.ctx = this.canvas.getContext("2d");
      this.image = null;
      this.canvas.addEventListener("click", event => this.click(event));
    }
    async load(url) {
      const image = new Image();
      image.src = url;
      await image.decode();
      this.image = image;
      this.canvas.width = image.naturalWidth;
      this.canvas.height = image.naturalHeight;
      this.canvas.parentElement.classList.add("loaded");
      this.draw();
    }
    click(event) {
      if (!this.image) return;
      const count = Number($("#particleCount").value || 1);
      const rect = this.canvas.getBoundingClientRect();
      const point = {
        x: (event.clientX - rect.left) * this.canvas.width / rect.width,
        y: (event.clientY - rect.top) * this.canvas.height / rect.height
      };
      if (state.seeds[this.name].length >= count) state.seeds[this.name] = [];
      state.seeds[this.name].push(point);
      this.draw(); updateSeedUI(); scheduleSave();
    }
    draw() {
      if (!this.image) return;
      this.ctx.drawImage(this.image, 0, 0);
      const scale = Math.max(1, this.canvas.width / 760);
      state.seeds[this.name].forEach((point, index) => {
        this.ctx.beginPath();
        this.ctx.arc(point.x, point.y, 11 * scale, 0, Math.PI * 2);
        this.ctx.fillStyle = this.name === "front" ? "#f09058" : "#3f7cc8";
        this.ctx.fill();
        this.ctx.lineWidth = 2 * scale; this.ctx.strokeStyle = "white"; this.ctx.stroke();
        this.ctx.fillStyle = "white"; this.ctx.font = `bold ${12 * scale}px system-ui`;
        this.ctx.textAlign = "center"; this.ctx.textBaseline = "middle";
        this.ctx.fillText(String(index + 1), point.x, point.y);
      });
    }
  }
  const canvases = { front: new SeedCanvas("front"), side: new SeedCanvas("side") };

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    headers.set("X-Studio-Token", token);
    if (options.body && !(options.body instanceof Blob) && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
    const response = await fetch(path, { ...options, headers });
    const contentType = response.headers.get("Content-Type") || "";
    const value = contentType.includes("json") ? await response.json() : await response.text();
    if (!response.ok) throw new Error(value.error || value || `Request failed (${response.status})`);
    return value;
  }

  function mediaUrl(path, endpoint = "media", extra = {}) {
    const query = new URLSearchParams({ token, path, ...extra });
    return `/api/${endpoint}?${query}`;
  }

  function getNested(object, path) { return path.split(".").reduce((value, key) => value?.[key], object); }
  function setNested(object, path, value) {
    const keys = path.split("."); let target = object;
    keys.slice(0, -1).forEach(key => target = target[key] ||= {});
    target[keys.at(-1)] = value;
  }
  function inputValue(input) {
    if (input.type === "checkbox") return input.checked;
    if (input.type === "number" && input.value !== "") return Number(input.value);
    return input.value;
  }
  function setInput(input, value) {
    if (input.type === "checkbox") input.checked = Boolean(value);
    else input.value = value ?? "";
  }

  function applySession() {
    const trial = state.session.trial;
    $("#videoPath").value = trial.video_path || "";
    $("#backgroundPath").value = trial.background_path || "";
    $("#calibrationPath").value = trial.calibration_path || "";
    $("#trialName").value = trial.name || "";
    $("#fps").value = trial.fps || "";
    $("#particleCount").value = trial.particle_count || 1;
    $("#seedFrame").value = trial.tracking.seed_frame ?? 1;
    $("#seedFrameMirror").value = trial.tracking.seed_frame ?? 1;
    const sameBackground = Boolean(trial.video_path && trial.background_path === trial.video_path);
    $("#useVideoBackground").checked = sameBackground;
    $("#backgroundPath").disabled = sameBackground;
    state.seeds.front = parseSeeds(trial.tracking.front_seeds);
    state.seeds.side = parseSeeds(trial.tracking.side_seeds);
    $("#useFocusBox").checked = !trial.tracking.no_focus_box;
    $$('[data-bind]').forEach(input => {
      const binding = input.dataset.bind;
      let value;
      if (binding.startsWith("shared.")) value = getNested(trial.tracking, binding.slice(7));
      else value = getNested(trial, binding);
      setInput(input, value);
    });
    const results = state.session.results || {};
    $$('[data-results-bind]').forEach(input => setInput(input, getNested(results, input.dataset.resultsBind)));
    $$('[data-artifact]').forEach(input => input.value = trial && state.session.artifacts?.[input.dataset.artifact] || "");
    updateSeedUI(); renderArtifacts(); updateChecks(); renderAnalysisWorkspace(); renderOrbitWorkspace(); renderHome();
    if (trial.video_path) inspectVideo(false);
  }

  function collectSession() {
    const trial = state.session.trial;
    trial.video_path = $("#videoPath").value.trim();
    trial.background_path = $("#useVideoBackground").checked ? trial.video_path : $("#backgroundPath").value.trim();
    trial.calibration_path = $("#calibrationPath").value.trim();
    trial.name = $("#trialName").value.trim();
    trial.fps = $("#fps").value === "" ? "" : Number($("#fps").value);
    trial.particle_count = Number($("#particleCount").value || 1);
    const seedFrame = Number($("#seedFrame").value || 0);
    trial.preview.seed_frame = seedFrame; trial.tracking.seed_frame = seedFrame;
    if (trial.preview.focus_box_mode === "follow" && Number(trial.preview.start_frame) < seedFrame) {
      trial.preview.start_frame = seedFrame;
    }
    trial.preview.front_seeds = formatSeeds(state.seeds.front); trial.preview.side_seeds = formatSeeds(state.seeds.side);
    trial.tracking.front_seeds = formatSeeds(state.seeds.front); trial.tracking.side_seeds = formatSeeds(state.seeds.side);
    trial.preview.no_focus_box = !$("#useFocusBox").checked;
    trial.tracking.no_focus_box = !$("#useFocusBox").checked;
    $$('[data-bind]').forEach(input => {
      const binding = input.dataset.bind; const value = inputValue(input);
      if (binding.startsWith("shared.")) {
        setNested(trial.preview, binding.slice(7), value); setNested(trial.tracking, binding.slice(7), value);
      } else setNested(trial, binding, value);
    });
    const results = state.session.results ||= {};
    $$('[data-results-bind]').forEach(input => setNested(results, input.dataset.resultsBind, inputValue(input)));
    $$('[data-artifact]').forEach(input => {
      const value = input.value.trim();
      if (value) state.session.artifacts[input.dataset.artifact] = value;
      else delete state.session.artifacts[input.dataset.artifact];
    });
    return state.session;
  }

  function parseSeeds(value) {
    if (!value) return [];
    return String(value).split(";").filter(Boolean).map(chunk => {
      const [x, y] = chunk.split(",").map(Number); return { x, y };
    }).filter(point => Number.isFinite(point.x) && Number.isFinite(point.y));
  }
  function formatSeeds(points) { return points.map(point => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(";"); }

  function populateLibraries() {
    const groups = [["videoLibrary", "videos"], ["backgroundLibrary", "backgrounds"], ["calibrationLibrary", "calibrations"]];
    groups.forEach(([id, group]) => {
      const select = $(`#${id}`); const first = select.options[0].cloneNode(true); select.replaceChildren(first);
      state.library[group].forEach(item => select.add(new Option(`${item.name}${item.trial_id ? ` · ${item.trial_id}` : ""}`, item.path)));
    });
  }

  async function inspectVideo(showErrors = true) {
    const path = $("#videoPath").value.trim(); if (!path) return;
    try {
      const info = await api(`/api/inspect?path=${encodeURIComponent(path)}`);
      state.session.trial.media_info = info;
      if (!$("#fps").value && info.fps) $("#fps").value = round(info.fps, 6);
      $("#videoMeta").classList.remove("muted");
      $("#videoMeta").innerHTML = `<span><b>${info.width_px} × ${info.height_px}</b><small>resolution</small></span><span><b>${formatDuration(info.duration_s)}</b><small>duration</small></span><span><b>${info.frame_count?.toLocaleString() || "unknown"}</b><small>frames</small></span><span><b>${escapeHtml(info.codec || "unknown")}</b><small>codec</small></span>`;
      $("#fpsEvidence").textContent = info.fps ? `Detected from media metadata: ${round(info.fps, 6)} fps. Confirm this is the effective experimental frame rate.` : "No frame rate was found in the media metadata. Enter the confirmed value before motion analysis.";
      const suggestion = await api("/api/suggest", { method: "POST", body: JSON.stringify({ video_path: path }) });
      if (!$("#trialName").value) $("#trialName").value = suggestion.name || "";
      if (!$("#backgroundPath").value && suggestion.background_path) $("#backgroundPath").value = suggestion.background_path;
      if (!$("#calibrationPath").value && suggestion.calibration_path) $("#calibrationPath").value = suggestion.calibration_path;
      const evidence = $("#companionEvidence");
      if (suggestion.provenance_source) {
        evidence.textContent = `Recovered from an earlier ${suggestion.trial_id} run${suggestion.calibration_status ? ` · ${suggestion.calibration_status}` : ""}. Visually verify it before accepting results.`;
        evidence.classList.remove("hidden");
      } else evidence.classList.add("hidden");
      updateChecks(); scheduleSave();
    } catch (error) { if (showErrors) notice(error.message); }
  }

  async function chooseVideo(path) {
    if (!path) return;
    const previous = state.session.trial.video_path;
    if (previous && previous !== path) {
      state.session.artifacts = {};
      state.session.stage_status = {};
      state.session.active_result = "";
      state.session.trial.review.raw_overlay_reviewed = false;
      state.session.trial.review.raw_overlay_note = "";
      state.seeds.front = []; state.seeds.side = [];
      $("#backgroundPath").value = ""; $("#calibrationPath").value = "";
      $("#useVideoBackground").checked = false; $("#backgroundPath").disabled = false;
      renderArtifacts(); updateSeedUI();
    }
    $("#videoPath").value = path;
    await inspectVideo();
  }

  async function upload(kind, file) {
    if (!file) return;
    const wrap = $("#uploadProgressWrap"), bar = $("#uploadProgress"); wrap.classList.remove("hidden"); bar.style.width = "15%";
    try {
      const query = new URLSearchParams({ token, kind, filename: file.name });
      const result = await api(`/api/upload?${query}`, { method: "POST", body: file, headers: { "Content-Type": "application/octet-stream" } });
      bar.style.width = "100%";
      if (kind === "video") await chooseVideo(result.path);
      else $("#backgroundPath").value = result.path;
      state.library = await api("/api/library"); populateLibraries(); scheduleSave();
      notice(`${file.name} was imported without modifying the original.`, "success");
    } catch (error) { notice(error.message); }
    setTimeout(() => wrap.classList.add("hidden"), 900);
  }

  async function uploadResult(kind, file) {
    if (!file) return;
    try {
      const query = new URLSearchParams({ token, kind, filename: file.name });
      const result = await api(`/api/upload?${query}`, { method: "POST", body: file, headers: { "Content-Type": "application/octet-stream" } });
      if (kind === "results_csv") {
        $("#resultCsvPath").value = result.path;
        if (! $('[data-results-bind="label"]').value) $('[data-results-bind="label"]').value = file.name.replace(/\.csv$/i, "");
      } else {
        $("#resultVideoPath").value = result.path;
        try {
          const info = await api(`/api/inspect?path=${encodeURIComponent(result.path)}`);
          const fpsInput = $('[data-results-bind="fps"]');
          if (!fpsInput.value && info.fps) fpsInput.value = round(info.fps, 6);
        } catch (_) { /* The video remains usable even if metadata inspection fails. */ }
      }
      scheduleSave();
      renderOrbitWorkspace();
      notice(`${file.name} was imported without modifying the original.`, "success");
    } catch (error) { notice(error.message); }
  }

  function useCurrentTrackingResult() {
    const artifacts = state.session?.artifacts || {};
    const path = scalarArtifact(artifacts.corrected_tracks_csv) || scalarArtifact(artifacts.tracks_csv);
    if (!path) return notice("Track particles or enter an existing analyzer CSV path first.");
    $("#resultCsvPath").value = path;
    if (!$('[data-results-bind="label"]').value) $('[data-results-bind="label"]').value = state.session.trial.name || "tracked particle";
    if (!$('[data-results-bind="fps"]').value && Number(state.session.trial.fps) > 0) $('[data-results-bind="fps"]').value = state.session.trial.fps;
    if (!$("#resultVideoPath").value && state.session.trial.video_path) $("#resultVideoPath").value = state.session.trial.video_path;
    scheduleSave();
    renderOrbitWorkspace();
    notice("Current tracking output attached to the circular-motion workspace.", "success");
  }

  async function loadSeedFrames() {
    collectSession();
    const trial = state.session.trial;
    if (!trial.video_path) return notice("Choose a video first.");
    if (!trial.calibration_path) return notice("Create or choose a calibration before loading camera views.");
    const frame = Number($("#seedFrame").value || 0);
    try {
      await Promise.all(["front", "side"].map(view => canvases[view].load(mediaUrl(trial.video_path, "frame", { frame, calibration: trial.calibration_path, view }))));
      notice("Camera views loaded. Click the same particle order in both panels.", "success");
    } catch (error) { notice(error.message); }
  }

  function updateSeedUI() {
    const count = Number($("#particleCount").value || 1), total = state.seeds.front.length + state.seeds.side.length;
    $("#frontSeeds").textContent = state.seeds.front.length ? formatSeeds(state.seeds.front) : "No particle markers";
    $("#sideSeeds").textContent = state.seeds.side.length ? formatSeeds(state.seeds.side) : "No particle markers";
    $("#seedCounter").textContent = `${total} / ${count * 2} marked`;
    canvases.front.draw(); canvases.side.draw(); updateChecks();
  }

  function validateAction(action) {
    collectSession(); const trial = state.session.trial, artifacts = state.session.artifacts;
    if (action === "orbit_analysis") {
      if (!state.session.results?.source_csv) return "Choose or import a position-versus-time CSV first.";
      return "";
    }
    if (!trial.video_path) return "Choose a source video first.";
    if (["preview", "preview_full", "track", "rotational", "both"].includes(action) && !trial.background_path) return "Choose a background or enable ‘Use a frame from this video’.";
    if (action === "track" && !trial.calibration_path) return "Create or choose a calibration before tracking.";
    if (action === "track" && !(Number(trial.fps) > 0)) return "Confirm the recording frame rate before tracking so time_s is correct.";
    if (["verify_tracking", "review_identities", "translational", "rotational", "both"].includes(action) && !artifacts.tracks_csv && !artifacts.corrected_tracks_csv) return "Track the particles first.";
    if (["translational", "rotational", "both"].includes(action) && !artifacts.preview_video) return "Render a preview first so state-vector overlays have a video background.";
    if (["translational", "rotational", "both"].includes(action) && !(Number(trial.fps) > 0)) return "Confirm the recording frame rate before calculating motion.";
    return "";
  }

  async function runAction(action) {
    const problem = validateAction(action); if (problem) return notice(problem);
    try {
      state.job = await api("/api/run", { method: "POST", body: JSON.stringify({ action, session: state.session }) });
      showJob(state.job); pollJob();
    } catch (error) { notice(error.message); }
  }

  async function runActionFromButton(button) {
    const source = button.dataset.overlaySource;
    if (source === "smoothed") {
      const hasRaw = Boolean(state.session.artifacts?.track_verification_raw);
      if (!hasRaw) return notice("Create and inspect the raw detector overlay before smoothing.");
      if (!$("#rawReviewAccepted").checked) return notice("Confirm that the raw marker follows the particle before rendering smoothing.");
    }
    if (source) {
      state.session.trial.verification.position_source = source;
      const select = $('[data-bind="verification.position_source"]');
      if (select) select.value = source;
    }
    if (button.dataset.action === "track") {
      state.session.trial.review.raw_overlay_reviewed = false;
      state.session.trial.review.raw_overlay_note = "";
      $("#rawReviewAccepted").checked = false;
      const reviewNote = $('[data-bind="review.raw_overlay_note"]');
      if (reviewNote) reviewNote.value = "";
      const staleKeys = [
        "tracks_csv", "track_summary", "track_plot", "corrected_tracks_csv",
        "track_verification_raw", "track_verification_smoothed", "motion_csv",
        "translational_state_csv", "translational_overlays", "rotation_state_csv",
        "reduced_rotation_state_csv", "rotation_diagnostics", "rotation_overlay",
      ];
      staleKeys.forEach(key => delete state.session.artifacts[key]);
      $$('[data-artifact]').forEach(input => { if (staleKeys.includes(input.dataset.artifact)) input.value = ""; });
      state.activeResultKey = ""; state.qualityPath = "";
    }
    await runAction(button.dataset.action);
  }

  async function pollJob() {
    clearTimeout(state.poller);
    if (!state.job) return;
    try {
      state.job = await api(`/api/job?id=${encodeURIComponent(state.job.job_id)}`); showJob(state.job);
      if (["queued", "running"].includes(state.job.status)) state.poller = setTimeout(pollJob, 900);
      else {
        const loaded = await api("/api/session"); state.session = loaded.session; applySession();
        if (state.job.status === "complete" && ["track", "verify_tracking"].includes(state.job.action)) {
          state.activeResultKey = state.job.artifacts?.track_verification_raw ? "track_verification_raw" : state.job.artifacts?.track_verification_smoothed ? "track_verification_smoothed" : state.session.artifacts?.track_verification_raw ? "track_verification_raw" : "track_verification_smoothed";
          renderAnalysisWorkspace(state.activeResultKey);
          showPage("review");
        }
        if (state.job.status === "complete" && state.job.action === "orbit_analysis") {
          state.orbitSummaryPath = "";
          await renderOrbitWorkspace();
          showPage("results", "orbitResultsCard");
        }
        if (state.job.status === "complete") notice(`${state.job.label} completed.`, "success");
        else notice(state.job.message || `${state.job.label} did not complete.`);
      }
    } catch (error) { notice(error.message); }
  }

  function showJob(job) {
    const panel = $("#jobPanel"); panel.classList.add("visible"); panel.setAttribute("aria-hidden", "false");
    $("#jobTitle").textContent = job.label; $("#jobMessage").textContent = job.message;
    $("#jobLog").textContent = job.log || "Waiting for output…"; $("#jobLog").scrollTop = $("#jobLog").scrollHeight;
    const progress = job.step_count ? Math.max(8, ((job.current_step - (job.status === "running" ? .4 : 0)) / job.step_count) * 100) : 8;
    $("#jobProgress").style.width = `${job.status === "complete" ? 100 : progress}%`;
    $("#cancelJob").disabled = !["queued", "running"].includes(job.status);
    $("#serverStatus").innerHTML = `<i></i> ${["queued", "running"].includes(job.status) ? "Working" : job.status === "complete" ? "Complete" : "Needs attention"}`;
  }

  function renderArtifacts() {
    const container = $("#artifactList"), artifacts = state.session?.artifacts || {};
    const entries = [];
    Object.entries(artifacts).forEach(([key, value]) => (Array.isArray(value) ? value : [value]).filter(Boolean).forEach((path, index) => entries.push([key, path, index])));
    if (!entries.length) { container.innerHTML = '<div class="empty-state"><span>⌁</span><b>Your results will appear here</b><p>Start with a short preview, then track the particles.</p></div>'; return; }
    container.innerHTML = entries.map(([key, path, index]) => {
      const ext = path.split(".").pop().toUpperCase(), title = artifactNames[key] || key.replaceAll("_", " ");
      const origin = artifactProvenance(path);
      return `<article class="artifact-card"><span class="artifact-icon">${escapeHtml(ext)}</span><div><b>${escapeHtml(title)}${index ? ` ${index + 1}` : ""}</b><small title="${escapeHtml(path)}">${escapeHtml(path)}</small><em>${escapeHtml(origin)}</em></div><div class="artifact-actions"><button class="button ghost" data-view-path="${escapeHtml(path)}" data-view-title="${escapeHtml(title)}">View</button><a class="button ghost" href="${mediaUrl(path)}" download>Save</a></div></article>`;
    }).join("");
    $$('[data-view-path]', container).forEach(button => button.addEventListener("click", () => viewArtifact(button.dataset.viewPath, button.dataset.viewTitle)));
  }

  const orbitArtifactKeys = [
    "orbit_report_html", "orbit_summary_json", "orbit_intervals_csv", "orbit_timeseries_csv",
    "orbit_diagnostic_plot", "orbit_xy_plot", "orbit_source_video"
  ];

  async function renderOrbitWorkspace() {
    if (!state.session) return;
    const settings = state.session.results || {};
    const artifacts = state.session.artifacts || {};
    const sourceReady = Boolean(settings.source_csv);
    badge("orbitInputCheck", sourceReady, sourceReady ? "CSV attached" : "Choose CSV");
    const summaryPath = scalarArtifact(artifacts.orbit_summary_json);
    badge("orbitResultCheck", Boolean(summaryPath), summaryPath ? "Analysis ready" : "Not run");
    $("#runOrbitDetector").disabled = !sourceReady;

    const video = settings.video_path;
    $("#orbitVideoViewer").innerHTML = video
      ? `<video src="${mediaUrl(video)}" controls playsinline preload="metadata"></video>`
      : "Import the matching video to review it here.";
    const diagnostic = scalarArtifact(artifacts.orbit_diagnostic_plot);
    const xy = scalarArtifact(artifacts.orbit_xy_plot);
    await renderInteractiveOrbitReport(scalarArtifact(artifacts.orbit_report_html));
    $("#orbitDiagnosticViewer").innerHTML = diagnostic ? `<img src="${mediaUrl(diagnostic)}" alt="Circular-motion detection diagnostic">` : "Run the detector to create the position, radius-envelope, and spectrum plot.";
    $("#orbitXyViewer").innerHTML = xy ? `<img src="${mediaUrl(xy)}" alt="Tracked orbit-path diagnostic">` : "Detected paths will appear here.";
    $("#openOrbitDiagnostic").disabled = !diagnostic;
    $("#openOrbitDiagnostic").dataset.path = diagnostic || "";
    renderOrbitArtifacts();

    const intervalPath = scalarArtifact(artifacts.orbit_intervals_csv);
    if (intervalPath) {
      $("#orbitIntervalsViewer").innerHTML = '<div class="empty-stage">Loading intervals…</div>';
      try {
        const table = await api(`/api/table?path=${encodeURIComponent(intervalPath)}&limit=500`);
        $("#orbitIntervalsViewer").innerHTML = tableMarkup(table, 500);
      } catch (error) {
        $("#orbitIntervalsViewer").textContent = error.message;
      }
    } else {
      $("#orbitIntervalsViewer").textContent = "The interval table will appear here.";
    }

    if (!summaryPath) {
      $("#orbitMetricCards").innerHTML = orbitMetricMarkup(null);
      state.orbitSummaryPath = "";
      return;
    }
    try {
      const response = await fetch(mediaUrl(summaryPath));
      if (!response.ok) throw new Error(`Summary request failed (${response.status})`);
      const summary = await response.json();
      if (scalarArtifact(state.session?.artifacts?.orbit_summary_json) !== summaryPath) return;
      $("#orbitMetricCards").innerHTML = orbitMetricMarkup(summary);
      state.orbitSummaryPath = summaryPath;
    } catch (error) {
      $("#orbitMetricCards").innerHTML = `<div class="orbit-metric error"><span>Summary unavailable</span><b>—</b><small>${escapeHtml(error.message)}</small></div>`;
    }
  }

  async function htmlArtifactMarkup(path) {
    const response = await fetch(mediaUrl(path));
    if (!response.ok) throw new Error(`Report request failed (${response.status})`);
    const markup = await response.text();
    const documentValue = new DOMParser().parseFromString(markup, "text/html");
    documentValue.querySelectorAll("[src], [href]").forEach(element => {
      const attribute = element.hasAttribute("src") ? "src" : "href";
      const value = element.getAttribute(attribute) || "";
      if (!value || value.startsWith("#") || /^(?:[a-z]+:|\/)/i.test(value)) return;
      element.setAttribute(attribute, mediaUrl(siblingPath(path, value)));
    });
    documentValue.querySelectorAll("[data-media-path]").forEach(element => {
      const mediaPath = element.getAttribute("data-media-path") || "";
      if (mediaPath) element.setAttribute("src", mediaUrl(mediaPath));
    });
    return `<!doctype html>${documentValue.documentElement.outerHTML}`;
  }

  async function renderInteractiveOrbitReport(path) {
    const shell = $("#orbitInteractiveShell"), frame = $("#orbitInteractiveFrame");
    $("#orbitMetricCards").classList.toggle("hidden", Boolean(path));
    $$("#orbitResultsCard .orbit-review-grid").forEach(grid => grid.classList.toggle("hidden", Boolean(path)));
    if (!path) {
      shell.classList.add("hidden");
      frame.removeAttribute("srcdoc");
      frame.src = "about:blank";
      frame.dataset.path = "";
      return;
    }
    shell.classList.remove("hidden");
    frame.dataset.path = path;
    frame.setAttribute("sandbox", "allow-scripts allow-downloads");
    frame.removeAttribute("srcdoc");
    frame.src = `/current-orbit/?v=${encodeURIComponent(path)}`;
  }

  function orbitMetricMarkup(summary) {
    if (!summary) return ["Episodes", "Frequency", "Period", "Radius", "Duty cycle"].map(label => `<div class="orbit-metric"><span>${label}</span><b>—</b><small>waiting for analysis</small></div>`).join("");
    const unit = summary.position_unit || "unit";
    const unitKey = String(unit).trim().replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "unit";
    const radius = summary[`radius_${unitKey}`] || {};
    const stat = (entry, suffix, digits = 3) => {
      if (!Number.isFinite(entry?.mean)) return "—";
      const spread = Number.isFinite(entry.standard_deviation) ? ` ± ${round(entry.standard_deviation, digits)}` : "";
      return `${round(entry.mean, digits)}${spread} ${suffix}`.trim();
    };
    return [
      ["Episodes", String(summary.episode_count ?? "—"), `${round((summary.orbit_duty_cycle || 0) * 100, 1)}% of recording`],
      ["Frequency", stat(summary.frequency_hz, "Hz"), "episode mean ± SD"],
      ["Period", stat(summary.period_s, "s"), "episode mean ± SD"],
      ["Radius", stat(radius, unit), "episode median mean ± SD"],
      ["Duty cycle", formatPercent(summary.orbit_duty_cycle), `${summary.threshold?.value ?? "—"} ${unit} threshold`],
    ].map(([label, value, note]) => `<div class="orbit-metric"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b><small>${escapeHtml(note)}</small></div>`).join("");
  }

  function renderOrbitArtifacts() {
    const container = $("#orbitArtifactList");
    const artifacts = state.session?.artifacts || {};
    const entries = orbitArtifactKeys.map(key => [key, scalarArtifact(artifacts[key])]).filter(([, path]) => path);
    if (!entries.length) {
      container.innerHTML = '<div class="empty-state"><span>○</span><b>No circular-motion run yet</b><p>Import a CSV and run the detector above.</p></div>';
      return;
    }
    container.innerHTML = entries.map(([key, path]) => {
      const title = artifactNames[key] || key.replaceAll("_", " ");
      const ext = path.split(".").pop().toUpperCase();
      return `<article class="artifact-card"><span class="artifact-icon">${escapeHtml(ext)}</span><div><b>${escapeHtml(title)}</b><small title="${escapeHtml(path)}">${escapeHtml(path)}</small><em>${escapeHtml(artifactProvenance(path))}</em></div><div class="artifact-actions"><button class="button ghost" data-orbit-view-path="${escapeHtml(path)}" data-orbit-view-title="${escapeHtml(title)}">View</button><a class="button ghost" href="${mediaUrl(path)}" download>Save</a></div></article>`;
    }).join("");
    $$('[data-orbit-view-path]', container).forEach(button => button.addEventListener("click", () => viewArtifact(button.dataset.orbitViewPath, button.dataset.orbitViewTitle)));
  }

  function artifactProvenance(path) {
    const history = [...(state.session?.history || [])].reverse();
    for (const run of history) {
      const paths = Object.values(run.artifacts || {}).flatMap(value => Array.isArray(value) ? value : [value]);
      if (paths.includes(path)) return `Created by “${run.label || run.action}” · ${formatDateTime(run.finished_at)}`;
    }
    return "Loaded from the saved workspace; creating run was not recorded here";
  }

  function scalarArtifact(value) {
    if (Array.isArray(value)) return value.filter(Boolean).at(-1) || "";
    return value || "";
  }

  function resultChoices() {
    const artifacts = state.session?.artifacts || {};
    const candidates = [
      ["track_verification_raw", "Raw detector overlay", "Review the retained unsmoothed detections before accepting smoothing."],
      ["track_verification_smoothed", "Smoothed overlay", "See the Savitzky–Golay trajectory over the original camera views."],
      ["rotation_overlay", "Full state", "Review translational and rotational state vectors over the video."],
      ["translational_overlays", "Motion overlay", "Review position, velocity, and acceleration vectors."],
      ["track_plot", "Trajectory plot", "Inspect front, side, and 3-D trajectory projections."],
      ["tracks_csv", "Trajectory data", "Preview time, position, diagnostics, and smoothed coordinates."],
      ["preview_video", "Subtraction preview", "Inspect what the tracker sees after background subtraction."],
    ];
    return candidates.map(([key, label, help]) => ({ key, label, help, path: scalarArtifact(artifacts[key]) })).filter(item => item.path);
  }

  async function renderAnalysisWorkspace(preferredKey = "") {
    if (!state.session) return;
    renderMiniTrajectoryPlot();
    const choices = resultChoices();
    const tabs = $("#resultTabs");
    if (!choices.length) {
      tabs.innerHTML = "";
      $("#primaryResultTitle").textContent = "Your tracked video will appear here";
      $("#primaryResultHelp").textContent = "Run tracking to compare the detected particle position with the recording.";
      $("#primaryResult").innerHTML = '<div class="empty-stage"><span>◎</span><b>No trajectory loaded</b><p>The overlay video, trajectory plot, and CSV can all be inspected here.</p></div>';
      renderQuality();
      return;
    }
    const selected = choices.find(item => item.key === (preferredKey || state.activeResultKey)) || choices[0];
    state.activeResultKey = selected.key;
    tabs.innerHTML = choices.map(item => `<button class="result-tab ${item.key === selected.key ? "active" : ""}" data-result-key="${escapeHtml(item.key)}" role="tab">${escapeHtml(item.label)}</button>`).join("");
    $$('[data-result-key]', tabs).forEach(button => button.addEventListener("click", () => renderAnalysisWorkspace(button.dataset.resultKey)));
    $("#primaryResultTitle").textContent = selected.label;
    $("#primaryResultHelp").textContent = `${selected.help} ${artifactProvenance(selected.path)}.`;
    await renderResultInto($("#primaryResult"), selected.path, selected.label);
    renderQuality();
  }

  function renderMiniTrajectoryPlot() {
    const path = scalarArtifact(state.session?.artifacts?.track_plot);
    const container = $("#miniTrajectoryPlot"), button = $("#openTrajectoryPlot");
    if (!path) {
      container.className = "mini-plot-empty";
      container.textContent = "Available after tracking";
      button.disabled = true; button.dataset.path = "";
      return;
    }
    container.className = "mini-plot-empty";
    container.innerHTML = `<img src="${mediaUrl(path)}" alt="Trajectory position projections">`;
    button.disabled = false; button.dataset.path = path;
  }

  async function renderResultInto(body, path, title) {
    const extension = path.split(".").pop().toLowerCase();
    body.innerHTML = '<div class="empty-stage">Loading result…</div>';
    try {
      if (["mp4", "mov", "mkv", "avi"].includes(extension)) {
        body.innerHTML = `<video src="${mediaUrl(path)}" controls playsinline preload="metadata"></video>`;
      } else if (["png", "jpg", "jpeg", "webp"].includes(extension)) {
        body.innerHTML = `<img src="${mediaUrl(path)}" alt="${escapeHtml(title)}">`;
      } else if (extension === "csv") {
        const data = await api(`/api/table?path=${encodeURIComponent(path)}&limit=200`);
        body.innerHTML = tableMarkup(data, 200);
      } else {
        const response = await fetch(mediaUrl(path));
        body.innerHTML = `<pre>${escapeHtml(await response.text())}</pre>`;
      }
    } catch (error) {
      body.innerHTML = `<div class="empty-stage"><b>Could not load this result</b><p>${escapeHtml(error.message)}</p></div>`;
    }
  }

  function tableMarkup(data, limit) {
    return `<div class="table-wrap"><table><thead><tr>${data.columns.map(cell => `<th>${escapeHtml(cell)}</th>`).join("")}</tr></thead><tbody>${data.rows.map(row => `<tr>${row.map(cell => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table>${data.truncated ? `<p class="field-help">Showing the first ${limit} rows. Use Save for the complete CSV.</p>` : ""}</div>`;
  }

  async function renderQuality() {
    const panel = $("#qualityPanel");
    const path = scalarArtifact(state.session?.artifacts?.corrected_tracks_csv) || scalarArtifact(state.session?.artifacts?.tracks_csv);
    if (!path) {
      state.qualityPath = "";
      panel.innerHTML = '<div class="diagnostics-heading"><p class="eyebrow">TRACK HEALTH</p><span class="diagnostic-badge neutral">Review</span></div><h4>Diagnostics appear after tracking</h4><p>These measurements help you find gaps, camera disagreement, and sudden jumps. They do not automatically prove a track is correct.</p>';
      return;
    }
    if (state.qualityPath === path && panel.dataset.loaded === "true") return;
    state.qualityPath = path; panel.dataset.loaded = "false";
    panel.innerHTML = '<div class="diagnostics-heading"><p class="eyebrow">TRACK HEALTH</p><span class="diagnostic-badge neutral">Loading</span></div><h4>Calculating diagnostics…</h4>';
    try {
      const quality = await api(`/api/quality?path=${encodeURIComponent(path)}`);
      if (state.qualityPath !== path) return;
      const z = quality.z_error_mm;
      const step = quality.step_mm;
      panel.innerHTML = `<div class="diagnostics-heading"><p class="eyebrow">TRACK HEALTH</p><span class="diagnostic-badge neutral">Review</span></div><h4>${quality.particle_count} particle${quality.particle_count === 1 ? "" : "s"} · ${quality.row_count.toLocaleString()} positions</h4><p>Frames ${quality.first_frame.toLocaleString()}–${quality.last_frame.toLocaleString()}</p><div class="metric-grid"><div class="metric"><b>${formatPercent(quality.coverage_fraction)}</b><small>frame coverage</small></div><div class="metric"><b>${quality.missing_frame_count.toLocaleString()}</b><small>missing frames</small></div><div class="metric"><b>${formatMetric(z?.median, " mm")}</b><small>median camera z difference</small></div><div class="metric"><b>${formatMetric(step?.p95, " mm")}</b><small>95th percentile frame step</small></div><div class="metric"><b>${formatPercent(quality.inside_chamber_fraction)}</b><small>inside calibrated chamber</small></div><div class="metric"><b>${quality.missing_position_rows.toLocaleString()}</b><small>rows missing position</small></div></div><div class="diagnostic-note">${escapeHtml(quality.interpretation)}</div>`;
      panel.dataset.loaded = "true";
    } catch (error) {
      panel.innerHTML = `<div class="diagnostics-heading"><p class="eyebrow">TRACK HEALTH</p><span class="diagnostic-badge neutral">Unavailable</span></div><h4>Diagnostics could not load</h4><p>${escapeHtml(error.message)}</p>`;
    }
  }

  async function viewArtifact(path, title) {
    const body = $("#viewerBody"), extension = path.split(".").pop().toLowerCase();
    $("#viewerTitle").textContent = title; $("#viewer").classList.remove("hidden"); body.innerHTML = '<div class="empty-state">Loading…</div>';
    try {
      if (["mp4", "mov", "mkv", "avi"].includes(extension)) body.innerHTML = `<video src="${mediaUrl(path)}" controls playsinline></video>`;
      else if (["png", "jpg", "jpeg", "webp"].includes(extension)) body.innerHTML = `<img src="${mediaUrl(path)}" alt="${escapeHtml(title)}">`;
      else if (extension === "csv") {
        const data = await api(`/api/table?path=${encodeURIComponent(path)}&limit=300`);
        body.innerHTML = tableMarkup(data, 300);
      } else if (extension === "html") {
        const iframe = document.createElement("iframe");
        iframe.className = "artifact-html-frame";
        iframe.title = title;
        iframe.setAttribute("sandbox", "allow-scripts allow-downloads");
        iframe.srcdoc = await htmlArtifactMarkup(path);
        body.replaceChildren(iframe);
      } else {
        const response = await fetch(mediaUrl(path)); const text = await response.text();
        body.innerHTML = `<pre>${escapeHtml(text)}</pre>`;
      }
      $("#viewer").scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (error) { body.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`; }
  }

  function siblingPath(path, relative) {
    const normalized = String(path).replaceAll("\\", "/");
    const base = normalized.slice(0, normalized.lastIndexOf("/") + 1);
    const parts = `${base}${relative}`.split("/");
    const resolved = [];
    parts.forEach(part => {
      if (part === "..") resolved.pop();
      else if (part !== ".") resolved.push(part);
    });
    return resolved.join("/");
  }

  function updateChecks() {
    if (!state.session) return; const trial = state.session.trial, artifacts = state.session.artifacts || {}, count = Number($("#particleCount").value || 1);
    badge("sourceCheck", trial.video_path && Number(trial.fps) > 0, trial.video_path ? (Number(trial.fps) > 0 ? "Source ready" : "Confirm fps") : "Waiting");
    const seeded = state.seeds.front.length === count && state.seeds.side.length === count;
    badge("seedCheck", seeded, seeded ? "Initialized" : trial.calibration_path ? "Mark particles" : "Needs calibration");
    badge("trackCheck", Boolean(artifacts.tracks_csv), artifacts.tracks_csv ? "Tracking complete" : "Ready when initialized");
    badge("resultCheck", Object.keys(artifacts).length > 0, Object.keys(artifacts).length ? `${Object.keys(artifacts).length} result groups` : "No results yet");
    $("#calibrationPrompt").classList.toggle("hidden", Boolean(trial.calibration_path));
    updateReadinessAndNextAction(trial, artifacts, seeded);
    updateReviewProtocol(trial, artifacts);
  }

  function updateReviewProtocol(trial, artifacts) {
    const hasTracks = Boolean(artifacts.tracks_csv || artifacts.corrected_tracks_csv);
    const hasRaw = Boolean(artifacts.track_verification_raw);
    const hasSmooth = Boolean(artifacts.track_verification_smoothed);
    const accepted = Boolean(trial.review?.raw_overlay_reviewed);
    badge("rawOverlayStatus", hasRaw, hasRaw ? "Ready to inspect" : hasTracks ? "Not created" : "Track first");
    badge("smoothOverlayStatus", hasSmooth, hasSmooth ? "Created" : accepted ? "Ready" : "Locked");
    $("#rawReviewStep").classList.toggle("complete", hasRaw && accepted);
    $("#smoothReviewStep").classList.toggle("complete", hasSmooth);
    $("#smoothReviewStep").classList.toggle("locked", !hasRaw || !accepted);
    $("#rawReviewAccepted").disabled = !hasRaw;
    $("#renderSmoothOverlay").disabled = !hasRaw || !accepted;
    const derivedButtons = $$('[data-action="translational"], [data-action="rotational"], [data-action="both"]');
    derivedButtons.forEach(button => button.disabled = !hasTracks || !Number(trial.fps) || !accepted);
  }

  function renderHome() {
    if (!state.session) return;
    const trial = state.session.trial;
    const history = state.session.history || [];
    const artifactCount = Object.values(state.session.artifacts || {}).flatMap(value => Array.isArray(value) ? value : [value]).filter(Boolean).length;
    $("#homeTrialTitle").textContent = trial.name || trial.media_info?.name || "No recording selected";
    $("#sessionOrigin").textContent = artifactCount
      ? `Loaded ${artifactCount} saved output file${artifactCount === 1 ? "" : "s"} from ${history.length} completed run${history.length === 1 ? "" : "s"}. Opening the Studio did not rerun them.`
      : state.sessionSource === "saved" ? "A saved workspace was restored, but it has no current output files." : "This is a new workspace with no saved results.";
    $("#resultsOrigin").querySelector("p").textContent = artifactCount
      ? `These ${artifactCount} files were restored from completed runs. Opening this page did not rerun analysis; each file is labeled with its origin below.`
      : "No saved outputs are attached to this workspace yet.";
    const recent = $("#recentRuns");
    if (!history.length) recent.innerHTML = '<p class="muted-copy">No completed runs yet.</p>';
    else recent.innerHTML = [...history].reverse().slice(0, 6).map(run => {
      const count = Object.values(run.artifacts || {}).flatMap(value => Array.isArray(value) ? value : [value]).filter(Boolean).length;
      return `<div class="run-row"><div><b>${escapeHtml(run.label || run.action)}</b><span>${escapeHtml(formatDateTime(run.finished_at))}</span></div><small>${count} file${count === 1 ? "" : "s"}</small></div>`;
    }).join("");
  }

  function showPage(page, scrollTarget = "") {
    const valid = ["home", "setup", "track", "review", "results"];
    state.currentPage = valid.includes(page) ? page : "home";
    $$('[data-page-section]').forEach(section => section.classList.toggle("page-active", section.dataset.pageSection === state.currentPage));
    $$('[data-page-target]').forEach(button => button.classList.toggle("active", button.dataset.pageTarget === state.currentPage));
    if (state.subviews[state.currentPage]) showSubview(state.currentPage, state.subviews[state.currentPage]);
    if (scrollTarget) {
      const target = $(`#${scrollTarget}`);
      if (target?.dataset.subviewGroup && target?.dataset.subview) {
        showSubview(target.dataset.subviewGroup, target.dataset.subview);
      }
    }
    history.replaceState(null, "", `${location.pathname}${location.search}#${state.currentPage}`);
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (scrollTarget) setTimeout(() => $(`#${scrollTarget}`)?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
  }

  function showSubview(group, view) {
    state.subviews[group] = view;
    $$(`[data-subview-group="${group}"][data-subview]`).forEach(element => element.classList.toggle("subview-active", element.dataset.subview === view));
    $$(`[data-subview-group="${group}"][data-subview-target]`).forEach(button => button.classList.toggle("active", button.dataset.subviewTarget === view));
    if (group === "results" && view === "sp6") {
      const frame = $("#sp6ReportFrame");
      if (frame && frame.src === "about:blank") frame.src = frame.dataset.src;
    }
  }

  function updateReadinessAndNextAction(trial, artifacts, seeded) {
    const checks = [
      [Boolean(trial.video_path), "Video selected", "Recording to analyze"],
      [Number(trial.fps) > 0, "Frame rate confirmed", "Required for time and motion"],
      [Boolean(trial.background_path), "Background ready", "Separates particle from scene"],
      [Boolean(trial.calibration_path), "Calibration ready", "Maps both cameras to 3-D"],
      [seeded, "Particles marked", "Same identity order in both views"],
      [Boolean(artifacts.preview_video), "Preview reviewed", "Recommended visual check"],
    ];
    const requiredReady = checks.slice(0, 5).every(item => item[0]);
    const readyCount = checks.filter(item => item[0]).length;
    $("#readinessCount").textContent = `${readyCount} / ${checks.length} ready`;
    $("#readinessTitle").textContent = requiredReady ? "Ready to track" : "Complete setup to enable tracking";
    $("#readinessList").innerHTML = checks.map(([ready, title, detail]) => `<div class="readiness-item ${ready ? "ready" : ""}"><i>${ready ? "✓" : "·"}</i><div><b>${escapeHtml(title)}</b><small>${escapeHtml(detail)}</small></div></div>`).join("");
    const tracksReady = Boolean(artifacts.tracks_csv || artifacts.corrected_tracks_csv);
    $$('[data-action="track"]').forEach(button => button.disabled = !requiredReady);
    $$('[data-action="verify_tracking"], [data-action="review_identities"]').forEach(button => button.disabled = !tracksReady);
    $$('[data-action="translational"], [data-action="rotational"], [data-action="both"]').forEach(button => button.disabled = !tracksReady || !Number(trial.fps) || !trial.review?.raw_overlay_reviewed);

    let next = ["Choose a source video", "The Studio will preserve the original and inspect its metadata.", "sourceCard", "Go to source"];
    if (trial.video_path && !(Number(trial.fps) > 0)) next = ["Confirm the experimental frame rate", "This makes time, velocity, and acceleration physically meaningful.", "sourceCard", "Confirm fps"];
    else if (trial.video_path && !trial.background_path) next = ["Choose a background", "Use a clean background image/video or select a frame from this recording.", "sourceCard", "Set background"];
    else if (trial.video_path && !trial.calibration_path) next = ["Create or choose calibration", "Calibration identifies the front and side camera regions and maps pixels to millimeters.", "seedCard", "Set calibration"];
    else if (!seeded) next = ["Mark the particle in both views", "Load the seed frame and click the same particle order in each camera.", "seedCard", "Initialize particles"];
    else if (!tracksReady) next = ["Run tracking", "The Studio will export coordinates and render a raw detector overlay.", "trackCard", "Go to tracking"];
    else if (!artifacts.track_verification_raw) next = ["Create the raw detector overlay", "Review unsmoothed retained detections before looking at smoothing.", "resultsCard", "Start raw review"];
    else if (!trial.review?.raw_overlay_reviewed) next = ["Inspect and confirm the raw overlay", "Check that the marker follows the particle throughout the recording.", "resultsCard", "Review raw overlay"];
    else if (!artifacts.track_verification_smoothed) next = ["Render the smoothed overlay", "Raw tracking is confirmed; now compare the post-processed trajectory.", "resultsCard", "Continue to smoothing"];
    else next = ["Analyze circular motion", "Attach the accepted trajectory in Results to detect orbit intervals, period, radius, and recurrence.", "resultsImportCard", "Open results"];
    $("#nextActionTitle").textContent = next[0]; $("#nextActionDetail").textContent = next[1];
    $("#nextActionButton").dataset.scrollTarget = next[2]; $("#nextActionButton").textContent = next[3];
  }
  function badge(id, complete, text) { const element = $(`#${id}`); element.textContent = text; element.classList.toggle("complete", complete); }

  function scheduleSave() {
    if (!state.session) return; collectSession(); clearTimeout(state.saveTimer); $("#saveStatus").textContent = "Saving…";
    state.saveTimer = setTimeout(async () => {
      try { const result = await api("/api/session", { method: "POST", body: JSON.stringify(state.session) }); state.session = result.session; $("#saveStatus").textContent = "Saved locally"; }
      catch (error) { $("#saveStatus").textContent = "Could not save"; }
    }, 500);
  }

  function notice(message, kind = "error") { const box = $("#globalNotice"); box.textContent = message; box.className = `notice ${kind === "success" ? "success" : ""}`; box.scrollIntoView({ behavior: "smooth", block: "nearest" }); if (kind === "success") setTimeout(() => box.classList.add("hidden"), 5000); }
  function formatDuration(seconds) { if (!Number.isFinite(seconds)) return "unknown"; const minutes = Math.floor(seconds / 60), remain = Math.round(seconds % 60); return `${minutes}:${String(remain).padStart(2, "0")}`; }
  function round(value, digits) { return Number(Number(value).toFixed(digits)); }
  function formatMetric(value, suffix = "") { return Number.isFinite(value) ? `${round(value, 4)}${suffix}` : "—"; }
  function formatPercent(value) { return Number.isFinite(value) ? `${round(value * 100, 1)}%` : "—"; }
  function formatDateTime(value) { if (!value) return "time not recorded"; const date = new Date(value); return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" }); }
  function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]); }

  function wireEvents() {
    $$('[data-page-target]').forEach(button => button.addEventListener("click", () => showPage(button.dataset.pageTarget)));
    $$('[data-subview-target]').forEach(button => button.addEventListener("click", () => showSubview(button.dataset.subviewGroup, button.dataset.subviewTarget)));
    $("#videoLibrary").addEventListener("change", event => { if (event.target.value) chooseVideo(event.target.value); });
    $("#backgroundLibrary").addEventListener("change", event => { if (event.target.value) { $("#backgroundPath").value = event.target.value; scheduleSave(); } });
    $("#calibrationLibrary").addEventListener("change", event => { if (event.target.value) { $("#calibrationPath").value = event.target.value; scheduleSave(); } });
    $("#videoUpload").addEventListener("change", event => upload("video", event.target.files[0]));
    $("#backgroundUpload").addEventListener("change", event => upload("background", event.target.files[0]));
    $("#resultCsvUpload").addEventListener("change", event => uploadResult("results_csv", event.target.files[0]));
    $("#resultVideoUpload").addEventListener("change", event => uploadResult("video", event.target.files[0]));
    $("#useCurrentTracks").addEventListener("click", useCurrentTrackingResult);
    $("#resultCsvPath").addEventListener("change", () => { collectSession(); renderOrbitWorkspace(); });
    $("#resultVideoPath").addEventListener("change", () => { collectSession(); renderOrbitWorkspace(); });
    $("#videoPath").addEventListener("change", event => chooseVideo(event.target.value.trim()));
    $("#useVideoBackground").addEventListener("change", event => { $("#backgroundPath").disabled = event.target.checked; scheduleSave(); });
    $("#seedFrame").addEventListener("input", event => { $("#seedFrameMirror").value = event.target.value; scheduleSave(); });
    $("#seedFrameMirror").addEventListener("input", event => { $("#seedFrame").value = event.target.value; scheduleSave(); });
    $("#loadSeedFrames").addEventListener("click", loadSeedFrames);
    $("#clearSeeds").addEventListener("click", () => { state.seeds.front = []; state.seeds.side = []; updateSeedUI(); scheduleSave(); });
    $("#particleCount").addEventListener("change", () => { state.seeds.front = []; state.seeds.side = []; updateSeedUI(); scheduleSave(); });
    $$('[data-action]').forEach(button => button.addEventListener("click", () => runActionFromButton(button)));
    $$('input:not([type=file]), select').forEach(input => input.addEventListener("change", scheduleSave));
    $("#cancelJob").addEventListener("click", async () => { if (state.job) { state.job = await api("/api/cancel", { method: "POST", body: JSON.stringify({ job_id: state.job.job_id }) }); showJob(state.job); } });
    $("#closeJob").addEventListener("click", () => $("#jobPanel").classList.remove("visible"));
    $("#showLog").addEventListener("click", () => $("#jobPanel").classList.toggle("show-log"));
    $("#closeViewer").addEventListener("click", () => { $("#viewer").classList.add("hidden"); $("#viewerBody").innerHTML = ""; });
    $("#openTrajectoryPlot").addEventListener("click", event => { const path = event.currentTarget.dataset.path; if (path) viewArtifact(path, "Trajectory projections"); });
    $("#openOrbitDiagnostic").addEventListener("click", event => { const path = event.currentTarget.dataset.path; if (path) viewArtifact(path, "Orbit detection diagnostic"); });
    $("#rawReviewAccepted").addEventListener("change", () => { collectSession(); updateReviewProtocol(state.session.trial, state.session.artifacts || {}); scheduleSave(); });
    $("#nextActionButton").addEventListener("click", event => {
      const target = $(`#${event.currentTarget.dataset.scrollTarget}`);
      showPage(target?.dataset.pageSection || "home", target?.id || "");
    });
  }

  async function initialize() {
    if (!token) return notice("This tab is missing its local access token. Relaunch the Studio.");
    wireEvents();
    try {
      const bootstrap = await api("/api/bootstrap"); state.session = bootstrap.session; state.sessionSource = bootstrap.session_source || "default"; state.library = bootstrap.library;
      if (state.session.artifacts?.orbit_summary_json) state.subviews.results = "current";
      populateLibraries(); applySession();
      const initialPage = location.hash.slice(1) || (state.session.artifacts?.orbit_summary_json ? "results" : state.session.artifacts?.tracks_csv ? "review" : state.session.trial.video_path ? "track" : "setup");
      showPage(initialPage); $("#serverStatus").innerHTML = "<i></i> Ready";
    } catch (error) { $("#serverStatus").textContent = "Offline"; notice(error.message); }
  }
  initialize();
})();
