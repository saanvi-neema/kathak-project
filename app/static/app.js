const tabs = document.querySelectorAll(".tab:not(.disabled)");
tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    tabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));
    document.getElementById(`tab-${tab.dataset.tab}`).classList.remove("hidden");
  });
});

function scoreClass(value) {
  if (value === null || value === undefined) return "";
  if (value >= 85) return "good";
  if (value >= 60) return "warn";
  return "bad";
}

function fmtScore(value) {
  return value === null || value === undefined ? "N/A" : value.toFixed(1);
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("video-input");
  const status = document.getElementById("status");
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("video", fileInput.files[0]);

  status.textContent = "Analyzing... this can take a few minutes depending on video length.";

  try {
    const resp = await fetch("/analyze", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      status.textContent = `Error: ${data.error || "analysis failed"}`;
      return;
    }
    status.textContent = "";
    renderResults(data);
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  }
});

function renderResults(data) {
  document.getElementById("upload-panel").classList.add("hidden");
  document.getElementById("results").classList.remove("hidden");

  renderOverview(data);
  renderChakkar(data.chakkar);
  renderTiming(data.timing);
  renderMudra(data.mudra);
  renderReports(data.report_lines);
  renderPoseView(data);
}

function mediaUrl(sessionId, filename) {
  return `/media/${sessionId}/${encodeURIComponent(filename)}`;
}

function renderPoseView(data) {
  const video = document.getElementById("pose-video");
  const note = document.getElementById("pose-note");
  if (data.overlay_video_filename && data.session_id) {
    video.src = mediaUrl(data.session_id, data.overlay_video_filename);
    video.classList.remove("hidden");
    note.textContent = "Skeleton overlay on the tracked body + hand joints.";
  } else {
    video.classList.add("hidden");
    note.textContent = "Couldn't generate a skeleton overlay for this video.";
  }
}

function renderOverview(data) {
  const video = document.getElementById("overview-video");
  if (data.original_video_filename && data.session_id) {
    video.src = mediaUrl(data.session_id, data.original_video_filename);
    video.classList.remove("hidden");
  } else {
    video.classList.add("hidden");
  }

  const container = document.getElementById("overview-cards");
  container.innerHTML = "";

  container.innerHTML += scoreRingCard("Overall Score", data.overall_score);

  if (data.chakkar) {
    container.innerHTML += `
      <div class="score-card">
        <div class="label">Chakkar Quality</div>
        <div class="value ${scoreClass(data.chakkar.quality_score)}">${fmtScore(data.chakkar.quality_score)}%</div>
      </div>
      <div class="score-card">
        <div class="label">Chakkar Count</div>
        <div class="value">${data.chakkar.result.rounded_count.toFixed(1)}</div>
      </div>`;
  }

  container.innerHTML += tempoCircleCard(data.timing ? data.timing.tempo_bpm : null);

  if (data.timing) {
    container.innerHTML += `
      <div class="score-card">
        <div class="label">Timing Accuracy</div>
        <div class="value ${scoreClass(data.timing.accuracy_score)}">${fmtScore(data.timing.accuracy_score)}%</div>
      </div>`;
  }

  if (!data.chakkar && !data.timing) {
    container.innerHTML += `<div class="empty-state">No chakkar or timing signal detected in this clip.</div>`;
  }
}

function scoreColorVar(value) {
  if (value === null || value === undefined) return "var(--border)";
  if (value >= 85) return "var(--good)";
  if (value >= 60) return "var(--warn)";
  return "var(--bad)";
}

function scoreRingCard(label, value) {
  const hasValue = value !== null && value !== undefined;
  const pct = hasValue ? Math.max(0, Math.min(100, value)) : 0;
  const display = hasValue ? `${fmtScore(value)}%` : "N/A";
  return `
    <div class="score-card score-card-circular">
      <div class="label">${label}</div>
      <div class="ring" style="--pct: ${pct}; --ring-color: ${scoreColorVar(value)}">
        <div class="ring-inner">
          <span class="ring-value">${display}</span>
        </div>
      </div>
    </div>`;
}

function tempoCircleCard(bpm) {
  const hasValue = bpm !== null && bpm !== undefined && !Number.isNaN(bpm);
  return `
    <div class="score-card score-card-circular">
      <div class="label">Tatkaar Tempo</div>
      <div class="circle-badge ${hasValue ? "" : "circle-badge-empty"}">
        ${hasValue
          ? `<span class="circle-value">${Math.round(bpm)}</span><span class="circle-unit">BPM</span>`
          : `<span class="circle-value">N/A</span>`}
      </div>
    </div>`;
}

function renderChakkar(chakkar) {
  const el = document.getElementById("chakkar-content");
  if (!chakkar) {
    el.innerHTML = `<div class="empty-state">No chakkar (rotation of at least half a turn) detected in this clip.</div>`;
    return;
  }
  const r = chakkar.result;
  el.innerHTML = `
    <div class="score-grid">
      <div class="score-card"><div class="label">Raw count</div><div class="value">${r.raw_count.toFixed(2)}</div></div>
      <div class="score-card"><div class="label">Rounded (clean landing)</div><div class="value">${r.rounded_count.toFixed(1)}</div></div>
      <div class="score-card"><div class="label">Ending orientation gap</div><div class="value">${r.orientation_gap_deg.toFixed(0)}&deg;</div></div>
      <div class="score-card"><div class="label">Stop quality</div><div class="value">${r.controlled_stop ? "Controlled" : "Abrupt"}</div></div>
    </div>
    <h3>Flags</h3>
    ${renderFlagList(chakkar.flags)}
  `;
}

function renderTiming(timing) {
  const el = document.getElementById("tatkaar-content");
  if (!timing) {
    el.innerHTML = `<div class="empty-state">Not enough audio/beat signal detected in this clip to check timing.</div>`;
    return;
  }
  el.innerHTML = `
    <div class="score-grid">
      <div class="score-card"><div class="label">Tempo</div><div class="value">${Math.round(timing.tempo_bpm)} BPM</div></div>
      <div class="score-card"><div class="label">Timing Accuracy</div><div class="value ${scoreClass(timing.accuracy_score)}">${fmtScore(timing.accuracy_score)}%</div></div>
    </div>
    <h3>Flags</h3>
    ${renderFlagList(timing.flags)}
    <p class="subtext">Note: this checks arm-movement timing against the beat, since foot-tracking needs footage validated for footwork specifically. See Tatkaar Analysis limitations in project notes.</p>
  `;
}

function renderMudra(mudra) {
  const note = document.getElementById("mudra-note");
  note.textContent = "Mudra analysis isn't run automatically on arbitrary uploads yet -- the checker can verify a NAMED mudra's shape, but there's no classifier to identify which mudra is happening at a given moment without ground-truth timestamps. This tab will populate once that piece exists.";
}

function bindComparisonFileLabel(inputId, labelId) {
  const input = document.getElementById(inputId);
  const label = document.getElementById(labelId);
  input.addEventListener("change", () => {
    label.textContent = input.files.length ? input.files[0].name : "No file selected";
  });
}
bindComparisonFileLabel("teacher-video-input", "teacher-video-filename");
bindComparisonFileLabel("student-video-input", "student-video-filename");

document.getElementById("comparison-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const teacherInput = document.getElementById("teacher-video-input");
  const studentInput = document.getElementById("student-video-input");
  const status = document.getElementById("comparison-status");

  if (!teacherInput.files.length || !studentInput.files.length) {
    status.textContent = "Select both a teacher video and your own video to compare.";
    return;
  }

  status.textContent = "Comparison analysis isn't built yet -- this is a UI preview of the upload flow only. " +
    "See methods.md for what it's scoped to do once the alignment + event-detection pieces exist.";
});

function renderFlagList(flags) {
  if (!flags || flags.length === 0) {
    return `<div class="empty-state">No issues flagged -- clean performance.</div>`;
  }
  return `<div>${flags.map(f => `<div class="flag-item">${formatTime(f.timestamp_sec)} &mdash; ${f.message}</div>`).join("")}</div>`;
}

function renderReports(lines) {
  const list = document.getElementById("report-list");
  list.innerHTML = "";
  if (!lines || lines.length === 0) {
    list.innerHTML = `<li class="empty-state">No issues flagged -- clean performance.</li>`;
    return;
  }
  lines.forEach(line => {
    const li = document.createElement("li");
    li.textContent = line;
    list.appendChild(li);
  });
}
