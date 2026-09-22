/* ================================================================
   app_v3.js — Kathak Practice Aid v3 (3-mode light theme)
   ================================================================ */

// ── Mode routing ──────────────────────────────────────────────────

const PAGES = ["landing", "beginner", "intermediate", "reference"];

function showMode(mode) {
  PAGES.forEach(p => {
    document.getElementById(`page-${p}`).classList.toggle("hidden", p !== mode);
  });
  const headerRight = document.getElementById("header-right");
  if (mode === "landing") {
    headerRight.innerHTML = "";
  } else {
    const pillClass = {beginner:"beginner", intermediate:"intermediate", reference:"reference"}[mode];
    const label = {beginner:"Beginner", intermediate:"Intermediate", reference:"Reference Review"}[mode];
    headerRight.innerHTML = `
      <span class="mode-pill mode-pill--${pillClass}">${label}</span>
      <button class="back-btn" onclick="showMode('landing')">← All modes</button>`;
  }
}

// ── Shared helpers ────────────────────────────────────────────────

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

function scoreClass(v) {
  if (v === null || v === undefined) return "";
  if (v >= 85) return "good";
  if (v >= 60) return "warn";
  return "bad";
}

function fmtScore(v) {
  return v === null || v === undefined ? "N/A" : v.toFixed(1);
}

function formatTime(sec) {
  const total = Math.round(sec);
  return `${Math.floor(total / 60)}:${(total % 60).toString().padStart(2, "0")}`;
}

function scoreColorVar(v) {
  if (v === null || v === undefined) return "var(--border)";
  if (v >= 85) return "var(--good)";
  if (v >= 60) return "var(--warn)";
  return "var(--bad)";
}

function scoreRingCard(label, value) {
  const pct = value !== null && value !== undefined ? Math.max(0, Math.min(100, value)) : 0;
  const display = value !== null && value !== undefined ? `${fmtScore(value)}%` : "N/A";
  return `
    <div class="score-card score-card-circular">
      <div class="label">${label}</div>
      <div class="ring" style="--pct:${pct};--ring-color:${scoreColorVar(value)}">
        <div class="ring-inner"><span class="ring-value">${display}</span></div>
      </div>
    </div>`;
}

function tempoCircleCard(bpm) {
  const ok = bpm !== null && bpm !== undefined && !Number.isNaN(bpm);
  return `
    <div class="score-card score-card-circular">
      <div class="label">Tempo</div>
      <div class="circle-badge">
        ${ok ? `<span class="circle-value">${Math.round(bpm)}</span><span class="circle-unit">BPM</span>`
              : `<span class="circle-value">N/A</span>`}
      </div>
    </div>`;
}

function renderFlagList(flags) {
  if (!flags || flags.length === 0) {
    return `<div class="empty-state">No issues flagged — clean performance.</div>`;
  }
  return `<div>${flags.map(f =>
    `<div class="flag-item">${formatTime(f.timestamp_sec)} — ${escapeHtml(f.message)}</div>`
  ).join("")}</div>`;
}

function mediaUrl(sessionId, filename) {
  return `/media/${sessionId}/${encodeURIComponent(filename)}`;
}

// ── Intermediate — tab switching ──────────────────────────────────

function intSwitchTab(tabId, btn) {
  document.querySelectorAll("#int-results .tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll("#int-results .tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(tabId).classList.add("active");
  btn.classList.add("active");
}

// ── Intermediate — file drop ──────────────────────────────────────

function intDropOver(e) { e.preventDefault(); }

function intDropFile(e) {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) {
    const input = document.getElementById("int-video-input");
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    intFileSelected();
  }
}

function intFileSelected() {
  const input = document.getElementById("int-video-input");
  const span = document.getElementById("int-filename");
  span.textContent = input.files.length ? input.files[0].name : "";
}

// ── Intermediate — analyze ────────────────────────────────────────

async function intermediateAnalyze() {
  const input = document.getElementById("int-video-input");
  const statusEl = document.getElementById("int-status");

  if (!input.files.length) {
    statusEl.textContent = "Select a video file first.";
    statusEl.className = "status status-error";
    return;
  }

  const btn = document.getElementById("int-analyze-btn");
  btn.disabled = true;
  statusEl.textContent = "Analyzing… this can take a few minutes depending on video length.";
  statusEl.className = "status status-info";

  const formData = new FormData();
  formData.append("video", input.files[0]);
  const taal = document.getElementById("int-taal-input").value;
  const sam = document.getElementById("int-sam-input").value;
  const mudras = document.getElementById("int-mudra-input").value;
  if (taal) formData.append("taal", taal);
  if (sam) formData.append("sam_time", sam);
  if (mudras.trim()) formData.append("expected_mudras", mudras);

  try {
    const resp = await fetch("/analyze", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      statusEl.textContent = `Error: ${data.error || "analysis failed"}`;
      statusEl.className = "status status-error";
      return;
    }
    statusEl.textContent = "";
    statusEl.className = "status";
    intRenderResults(data);
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    statusEl.className = "status status-error";
  } finally {
    btn.disabled = false;
  }
}

function intReset() {
  document.getElementById("int-upload-panel").classList.remove("hidden");
  document.getElementById("int-results").classList.add("hidden");
  document.getElementById("int-status").textContent = "";
  document.getElementById("int-status").className = "status";
}

function intRenderResults(data) {
  document.getElementById("int-upload-panel").classList.add("hidden");
  const results = document.getElementById("int-results");
  results.classList.remove("hidden");

  intRenderOverview(data);
  intRenderChakkar(data.chakkar, data.taal);
  intRenderTiming(data.timing);
  intRenderTatkaar(data.tatkaar);
  intRenderMudra(data.mudra);
  intRenderRasa(data.rasa);
  intRenderPose(data);

  // Switch to overview tab
  intSwitchTab("int-overview", document.querySelector('[data-tab="int-overview"]'));
}

function intRenderOverview(data) {
  const video = document.getElementById("int-overview-video");
  if (data.original_video_filename && data.session_id) {
    video.src = mediaUrl(data.session_id, data.original_video_filename);
    video.classList.remove("hidden");
  } else {
    video.classList.add("hidden");
  }

  const cards = document.getElementById("int-overview-cards");
  cards.innerHTML = scoreRingCard("Overall Score", data.overall_score);

  if (data.chakkar) {
    cards.innerHTML += `
      <div class="score-card">
        <div class="label">Chakkar Quality</div>
        <div class="value ${scoreClass(data.chakkar.quality_score)}">${fmtScore(data.chakkar.quality_score)}%</div>
      </div>
      <div class="score-card">
        <div class="label">Chakkars Detected</div>
        <div class="value">${data.chakkar.event_count}</div>
      </div>`;
  }

  cards.innerHTML += tempoCircleCard(data.timing ? data.timing.tempo_bpm : null);

  if (data.timing) {
    cards.innerHTML += `
      <div class="score-card">
        <div class="label">Timing Accuracy</div>
        <div class="value ${scoreClass(data.timing.accuracy_score)}">${fmtScore(data.timing.accuracy_score)}%</div>
      </div>`;
  }

  if (data.mudra) {
    const hasId = data.mudra.identification_accuracy_score !== null && data.mudra.identification_accuracy_score !== undefined;
    const hasRule = data.mudra.rule_agreement_score !== null && data.mudra.rule_agreement_score !== undefined;
    if (hasId) {
      cards.innerHTML += `
        <div class="score-card">
          <div class="label">Mudra ID Accuracy</div>
          <div class="value ${scoreClass(data.mudra.identification_accuracy_score)}">${fmtScore(data.mudra.identification_accuracy_score)}%</div>
        </div>`;
    } else if (hasRule) {
      cards.innerHTML += `
        <div class="score-card">
          <div class="label">Mudra Rule Agreement</div>
          <div class="value ${scoreClass(data.mudra.rule_agreement_score)}">${fmtScore(data.mudra.rule_agreement_score)}%</div>
        </div>`;
    }
    if (hasId || hasRule) {
      cards.innerHTML += `
        <div class="score-card">
          <div class="label">Mudra Holds</div>
          <div class="value">${data.mudra.events.length}</div>
        </div>`;
    }
  }

  if (!data.chakkar && !data.timing && !data.mudra) {
    cards.innerHTML += `<div class="empty-state">No chakkar, timing, or mudra signal detected in this clip.</div>`;
  }
}

function intRenderChakkar(chakkar, taal) {
  const el = document.getElementById("int-chakkar-content");
  if (!chakkar) {
    el.innerHTML = `<div class="empty-state">No chakkar (rotation of at least half a turn) detected in this clip.</div>`;
    return;
  }
  const taalByEnd = {};
  if (taal) taal.events.forEach(te => { taalByEnd[te.end_sec] = te; });
  const taalNote = taal
    ? `<p class="subtext">Chakkar-vs-sam checked against ${taal.taal[0].toUpperCase() + taal.taal.slice(1)} (sam at ${formatTime(taal.sam_time)}).</p>`
    : "";

  el.innerHTML = `
    <p class="subtext">${chakkar.event_count} chakkar${chakkar.event_count === 1 ? "" : "s"} detected — each scored separately.</p>
    ${taalNote}
    ${chakkar.events.map((e, i) => {
      const r = e.result;
      const te = taalByEnd[r.end_sec];
      const taalHtml = te
        ? `<p class="subtext">Ended ${Math.abs(te.beats_from_sam).toFixed(2)} beats ${te.beats_from_sam < 0 ? "before" : "after"} the nearest sam.</p>`
        : "";
      const hasDrift = r.drift_shoulder_widths !== null && r.drift_shoulder_widths !== undefined;
      const driftHtml = hasDrift
        ? `<p class="subtext">Moved about ${(r.drift_shoulder_widths * 100).toFixed(0)}% of a shoulder-width from start (path straightness: ${r.path_straightness.toFixed(2)}).</p>`
        : "";
      return `
        <h3>Chakkar ${i + 1} <span class="subtext">(${formatTime(r.start_sec)}–${formatTime(r.end_sec)})</span></h3>
        <div class="score-grid">
          <div class="score-card"><div class="label">Quality</div><div class="value ${scoreClass(e.quality_score)}">${fmtScore(e.quality_score)}%</div></div>
          <div class="score-card"><div class="label">Raw count</div><div class="value">${r.raw_count.toFixed(2)}</div></div>
          <div class="score-card"><div class="label">Rounded</div><div class="value">${r.rounded_count.toFixed(1)}</div></div>
          <div class="score-card"><div class="label">Orientation gap</div><div class="value">${r.orientation_gap_deg.toFixed(0)}°</div></div>
          <div class="score-card"><div class="label">Stop</div><div class="value">${r.controlled_stop ? "Controlled" : "Abrupt"}</div></div>
        </div>
        ${taalHtml}${driftHtml}`;
    }).join("")}
    <h3>All flags</h3>
    ${renderFlagList(chakkar.flags.concat(taal ? taal.flags : []))}`;
}

function intRenderTiming(timing) {
  const el = document.getElementById("int-timing-content");
  if (!timing) {
    el.innerHTML = `<div class="empty-state">Not enough audio/beat signal to check timing.</div>`;
    return;
  }
  el.innerHTML = `
    <div class="score-grid">
      <div class="score-card"><div class="label">Tempo</div><div class="value">${Math.round(timing.tempo_bpm)} BPM</div></div>
      <div class="score-card"><div class="label">Timing Accuracy</div><div class="value ${scoreClass(timing.accuracy_score)}">${fmtScore(timing.accuracy_score)}%</div></div>
    </div>
    <h3>Flags</h3>
    ${renderFlagList(timing.flags)}
    <p class="subtext">Note: checks arm-movement timing against the beat, not footwork — see Tatkaar tab for that.</p>`;
}

function intRenderTatkaar(tatkaar) {
  const note = document.getElementById("int-tatkaar-note");
  const content = document.getElementById("int-tatkaar-content");
  if (!tatkaar || !tatkaar.events || tatkaar.events.length === 0) {
    note.textContent = "No sustained rhythmic footwork stretch (tatkaar) detected in this clip's audio.";
    content.innerHTML = "";
    return;
  }
  note.textContent = "UNVALIDATED: detected from audio onsets (foot/ghungroo strikes), not calibrated against real tatkaar footage.";
  content.innerHTML = tatkaar.events.map((e, i) => `
    <div class="flag-item">
      Stretch ${i + 1}: ${formatTime(e.start_sec)}–${formatTime(e.end_sec)} —
      <strong>${e.strike_count} strikes</strong>
      ${e.strikes_per_sec != null ? `(~${e.strikes_per_sec.toFixed(1)}/sec)` : ""}
    </div>`).join("");
}

function intRenderMudra(mudra) {
  const note = document.getElementById("int-mudra-note");
  const content = document.getElementById("int-mudra-content");
  if (!mudra || !mudra.events || mudra.events.length === 0) {
    note.textContent = "No confidently-identified mudra holds found — either no classifier loaded, or no pose was held steadily enough.";
    content.innerHTML = "";
    return;
  }
  note.textContent = "Identification trained on a Bharatanatyam photo dataset, not Kathak footage — treat as a first pass.";
  const hasId = mudra.identification_accuracy_score !== null && mudra.identification_accuracy_score !== undefined;
  const rateHtml = `<div class="score-grid" style="margin-bottom:16px;">
    ${hasId ? `<div class="score-card"><div class="label">ID Accuracy</div><div class="value ${scoreClass(mudra.identification_accuracy_score)}">${fmtScore(mudra.identification_accuracy_score)}%</div></div>` : ""}
    ${mudra.rule_agreement_score != null ? `<div class="score-card"><div class="label">Rule Agreement</div><div class="value ${scoreClass(mudra.rule_agreement_score)}">${fmtScore(mudra.rule_agreement_score)}%</div></div>` : ""}
  </div>`;

  const eventsHtml = mudra.events.map(e => {
    const cls = e.mismatches && e.mismatches.length ? "flag-item" : "";
    const mismatchText = e.mismatches == null
      ? "(no reference rule)"
      : e.mismatches.length ? e.mismatches.join("; ") : "matches reference shape";
    const expectedText = e.expected_mudra
      ? ` — expected <strong>${escapeHtml(e.expected_mudra)}</strong> (${e.matches_expected ? "✓ correct" : "✗ mismatch"})`
      : "";
    return `<div class="${cls}">
      ${formatTime(e.start_sec)}–${formatTime(e.end_sec)} (${e.hand_side}): <strong>${escapeHtml(e.mudra)}</strong>
      (${(e.confidence * 100).toFixed(0)}% confidence) — ${mismatchText}${expectedText}
    </div>`;
  }).join("");

  content.innerHTML = `${rateHtml}<h3>Flags</h3>${renderFlagList(mudra.flags)}<h3>All identified holds</h3>${eventsHtml}`;
}

function intRenderRasa(rasa) {
  const note = document.getElementById("int-rasa-note");
  const content = document.getElementById("int-rasa-content");
  if (!rasa || rasa.length === 0) {
    note.textContent = "No face detected in this clip, so no rasa (facial expression) reading could be made.";
    content.innerHTML = "";
    return;
  }
  note.textContent = "UNVALIDATED: rule-based guess from rasa_reference.py, not calibrated against real navras footage.";
  content.innerHTML = `<h3>Sampled readings (every 1s)</h3>` +
    rasa.map(e => `<div class="flag-item">
      ${formatTime(e.start_sec)}–${formatTime(e.end_sec)}: <strong>${escapeHtml(e.display_name)}</strong>
      (${e.confidence_label} confidence, ${e.mismatch_count === 0 ? "clean match" : `${e.mismatch_count} criteria off`})
    </div>`).join("");
}

function intRenderPose(data) {
  const video = document.getElementById("int-pose-video");
  const note = document.getElementById("int-pose-note");
  if (data.overlay_video_filename && data.session_id) {
    video.src = mediaUrl(data.session_id, data.overlay_video_filename);
    video.classList.remove("hidden");
    note.textContent = "Skeleton overlay on tracked body + hand joints.";
  } else {
    video.classList.add("hidden");
    note.textContent = "Couldn't generate a skeleton overlay for this video.";
  }
}

// ── Reference Review ──────────────────────────────────────────────

function refTeacherSelected() {
  const f = document.getElementById("ref-teacher-input").files[0];
  document.getElementById("ref-teacher-name").textContent = f ? f.name : "";
}

function refStudentSelected() {
  const f = document.getElementById("ref-student-input").files[0];
  document.getElementById("ref-student-name").textContent = f ? f.name : "";
}

async function referenceCompare() {
  const teacherInput = document.getElementById("ref-teacher-input");
  const studentInput = document.getElementById("ref-student-input");
  const statusEl = document.getElementById("ref-status");

  if (!teacherInput.files.length || !studentInput.files.length) {
    statusEl.textContent = "Select both videos before comparing.";
    statusEl.className = "status status-error";
    return;
  }

  const btn = document.getElementById("ref-compare-btn");
  btn.disabled = true;
  statusEl.textContent = "Comparing… this aligns both videos' audio and can take a minute or two.";
  statusEl.className = "status status-info";

  const formData = new FormData();
  formData.append("teacher_video", teacherInput.files[0]);
  formData.append("student_video", studentInput.files[0]);

  try {
    const resp = await fetch("/compare", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      statusEl.textContent = `Error: ${data.error || "comparison failed"}`;
      statusEl.className = "status status-error";
      return;
    }
    statusEl.textContent = "";
    statusEl.className = "status";
    refRenderResults(data);
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    statusEl.className = "status status-error";
  } finally {
    btn.disabled = false;
  }
}

function refReset() {
  document.getElementById("ref-upload-panel").classList.remove("hidden");
  document.getElementById("ref-results").classList.add("hidden");
  document.getElementById("ref-status").textContent = "";
  document.getElementById("ref-status").className = "status";
}

function refRenderResults(data) {
  document.getElementById("ref-upload-panel").classList.add("hidden");
  const results = document.getElementById("ref-results");
  results.classList.remove("hidden");

  const rateHtml = data.match_rate != null
    ? `<div class="score-grid" style="margin-bottom:16px;">
         <div class="score-card"><div class="label">Actions matched</div><div class="value ${scoreClass(data.match_rate)}">${fmtScore(data.match_rate)}%</div></div>
       </div>`
    : "";

  const actionsHtml = (data.actions || []).map(a => {
    const cls = a.match ? "" : "flag-item";
    return `<div class="${cls}">
      ${formatTime(a.teacher_time)} (reference) → ${formatTime(a.student_time)} (you):
      reference had ${a.teacher_count}, you had ${a.student_count} — ${a.match ? "match" : "mismatch"}
    </div>`;
  }).join("");

  const denseSections = data.dense_sections || [];
  const extraTeacher = data.extra_teacher_sections || [];
  const extraStudent = data.extra_student_sections || [];
  const denseHtml = [
    ...denseSections.map(d => `<div class="${d.match ? "" : "flag-item"}">
      ${formatTime(d.teacher_time)} (ref) → ${formatTime(d.student_time)} (you):
      ref did ${d.teacher_count} strikes, you did ${d.student_count} (${d.ratio.toFixed(2)}x) — ${d.match ? "match" : "mismatch"}
    </div>`),
    ...extraTeacher.map(d => `<div class="flag-item">
      ${formatTime(d.start)} (reference only): ${d.count} strikes with no matching section in your video
    </div>`),
    ...extraStudent.map(d => `<div class="flag-item">
      ${formatTime(d.start)} (you only): ${d.count} strikes with no matching section in the reference video
    </div>`),
  ].join("");

  document.getElementById("ref-results-content").innerHTML = `
    <h3>Summary</h3>
    ${rateHtml}
    <h3>Flags</h3>
    ${renderFlagList(data.flags)}
    <h3>Rhythmic / tatkaar sections</h3>
    <p class="subtext">Fast repeated sections are compared by strike count, not sound-by-sound.</p>
    ${denseHtml || '<div class="empty-state">No dense rhythmic sections detected.</div>'}
    <h3>All matched actions</h3>
    ${actionsHtml || '<div class="empty-state">No comparable movement-sound actions detected.</div>'}`;
}

// ── Beginner — live capture ───────────────────────────────────────

let liveSessionId = null;
let liveStream = null;
let liveRunning = false;
let liveStarting = false;
const LIVE_CHUNK_MS = 1500;
const BEG_HISTORY_MAX = 5;
let begHistory = [];

function pickSupportedMimeType() {
  const candidates = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"];
  for (const c of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(c)) return c;
  }
  return "";
}

async function recordOneLiveChunk(mimeType) {
  return new Promise((resolve, reject) => {
    const recorder = new MediaRecorder(liveStream, mimeType ? { mimeType } : undefined);
    const parts = [];
    const startedAt = performance.now();
    recorder.ondataavailable = e => { if (e.data && e.data.size > 0) parts.push(e.data); };
    recorder.onstop = () => resolve({ blob: new Blob(parts, { type: recorder.mimeType }), durationSec: (performance.now() - startedAt) / 1000 });
    recorder.onerror = e => reject(e.error || e);
    recorder.start();
    setTimeout(() => { if (recorder.state !== "inactive") recorder.stop(); }, LIVE_CHUNK_MS);
  });
}

async function liveRecordLoop() {
  const mimeType = pickSupportedMimeType();
  while (liveRunning) {
    try {
      const { blob, durationSec } = await recordOneLiveChunk(mimeType);
      if (!liveRunning) break;
      const ext = blob.type.includes("mp4") ? "mp4" : "webm";
      const resp = await fetch(
        `/live/chunk?session_id=${encodeURIComponent(liveSessionId)}&ext=${ext}&duration_sec=${durationSec}`,
        { method: "POST", headers: { "Content-Type": blob.type || "application/octet-stream" }, body: blob }
      );
      const data = await resp.json();
      if (!liveRunning) break;
      if (resp.ok) beginnerUpdateUI(data);
    } catch (err) {
      const s = document.getElementById("beg-status");
      s.textContent = `Error: ${err.message}`;
      s.className = "status status-error";
    }
  }
}

function beginnerUpdateUI(data) {
  const mudraEl = document.getElementById("beg-mudra-name");
  const scoreEl = document.getElementById("beg-mudra-score");
  const hapticDot = document.getElementById("beg-haptic-dot");
  const hapticText = document.getElementById("beg-haptic-text");
  const historyEl = document.getElementById("beg-mudra-history");

  if (data.mudra && data.mudra.events && data.mudra.events.length > 0) {
    const last = data.mudra.events[data.mudra.events.length - 1];
    const pct = (last.confidence * 100).toFixed(0);
    mudraEl.textContent = last.mudra;
    scoreEl.textContent = `${pct}%`;
    scoreEl.className = "mudra-score " + (last.confidence >= 0.85 ? "good" : last.confidence >= 0.6 ? "warn" : "bad");

    begHistory.unshift({ mudra: last.mudra, pct, time: formatTime(last.start_sec) });
    if (begHistory.length > BEG_HISTORY_MAX) begHistory.pop();
    historyEl.innerHTML = begHistory.map(h =>
      `<div style="padding:4px 0;border-bottom:1px solid var(--border)">${h.time} — ${h.mudra} (${h.pct}%)</div>`
    ).join("");

    // Haptic indication
    if (last.confidence < 0.85) {
      hapticDot.className = "haptic-dot haptic-dot--buzz";
      hapticText.textContent = "Haptic glove: low score — buzz sent";
      setTimeout(() => {
        hapticDot.className = "haptic-dot haptic-dot--active";
        hapticText.textContent = "Haptic glove: connected";
      }, 800);
    } else {
      hapticDot.className = "haptic-dot haptic-dot--active";
      hapticText.textContent = "Haptic glove: connected";
    }
  } else {
    mudraEl.textContent = "—";
    scoreEl.textContent = "—";
    scoreEl.className = "mudra-score";
  }
}

async function beginnerStart() {
  if (liveStarting || liveRunning) return;
  liveStarting = true;
  const statusEl = document.getElementById("beg-status");
  try {
    if (!window.MediaRecorder || !navigator.mediaDevices || !pickSupportedMimeType()) {
      statusEl.textContent = "This browser doesn't support live camera recording — try Chrome or Firefox.";
      statusEl.className = "status status-error";
      return;
    }

    statusEl.textContent = "Starting session…";
    statusEl.className = "status status-info";

    const startResp = await fetch("/live/start", { method: "POST", body: new FormData() });
    const startData = await startResp.json();
    if (!startResp.ok) {
      statusEl.textContent = `Error: ${startData.error || "could not start live session"}`;
      statusEl.className = "status status-error";
      return;
    }
    liveSessionId = startData.session_id;

    try {
      liveStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    } catch (err) {
      statusEl.textContent = `Camera/mic access failed: ${err.message}`;
      statusEl.className = "status status-error";
      const orphan = liveSessionId;
      liveSessionId = null;
      fetch(`/live/stop?session_id=${encodeURIComponent(orphan)}`, { method: "POST" }).catch(() => {});
      return;
    }

    document.getElementById("beginner-preview").srcObject = liveStream;
    statusEl.textContent = "";
    statusEl.className = "status";
    document.getElementById("beg-start-btn").classList.add("hidden");
    document.getElementById("beg-stop-btn").classList.remove("hidden");
    document.getElementById("beg-live-indicator").classList.remove("hidden");

    const hapticDot = document.getElementById("beg-haptic-dot");
    const hapticText = document.getElementById("beg-haptic-text");
    hapticDot.className = "haptic-dot haptic-dot--active";
    hapticText.textContent = "Haptic glove: connected";

    begHistory = [];
    liveRunning = true;
    liveRecordLoop();
  } finally {
    liveStarting = false;
  }
}

async function beginnerStop() {
  liveRunning = false;
  if (liveStream) {
    liveStream.getTracks().forEach(t => t.stop());
    liveStream = null;
  }
  document.getElementById("beginner-preview").srcObject = null;
  document.getElementById("beg-stop-btn").classList.add("hidden");
  document.getElementById("beg-live-indicator").classList.add("hidden");
  document.getElementById("beg-start-btn").classList.remove("hidden");

  const hapticDot = document.getElementById("beg-haptic-dot");
  const hapticText = document.getElementById("beg-haptic-text");
  hapticDot.className = "haptic-dot haptic-dot--idle";
  hapticText.textContent = "Haptic glove: waiting";

  if (liveSessionId) {
    const sid = liveSessionId;
    liveSessionId = null;
    try {
      await fetch(`/live/stop?session_id=${encodeURIComponent(sid)}`, { method: "POST" });
    } catch (_) {}
  }
}

window.addEventListener("beforeunload", () => {
  if (liveSessionId) {
    navigator.sendBeacon(`/live/stop?session_id=${encodeURIComponent(liveSessionId)}`);
  }
});
