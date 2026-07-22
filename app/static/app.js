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
}

function renderOverview(data) {
  const container = document.getElementById("overview-cards");
  container.innerHTML = "";

  const overallVal = data.overall_score;
  container.innerHTML += `
    <div class="score-card">
      <div class="label">Overall Score</div>
      <div class="value ${scoreClass(overallVal)}">${fmtScore(overallVal)}${overallVal !== null ? "%" : ""}</div>
    </div>`;

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

  if (data.timing) {
    container.innerHTML += `
      <div class="score-card">
        <div class="label">Timing Accuracy</div>
        <div class="value ${scoreClass(data.timing.accuracy_score)}">${fmtScore(data.timing.accuracy_score)}%</div>
      </div>
      <div class="score-card">
        <div class="label">Tempo</div>
        <div class="value">${Math.round(data.timing.tempo_bpm)} BPM</div>
      </div>`;
  }

  if (!data.chakkar && !data.timing) {
    container.innerHTML = `<div class="empty-state">No chakkar or timing signal detected in this clip.</div>`;
  }
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
