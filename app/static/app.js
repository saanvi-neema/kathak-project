let hasAnalyzedVideo = false;

const tabs = document.querySelectorAll(".tab:not(.disabled)");
tabs.forEach(tab => {
  tab.addEventListener("click", () => {
    tabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));

    if (tab.dataset.tab === "comparison") {
      // Comparison doesn't depend on the single-video Overview analysis --
      // it's a separate upload flow, so it should work even if no video has
      // been analyzed yet.
      document.getElementById("upload-panel").classList.add("hidden");
      document.getElementById("results").classList.add("hidden");
      document.getElementById("tab-comparison").classList.remove("hidden");
    } else if (hasAnalyzedVideo) {
      document.getElementById("upload-panel").classList.add("hidden");
      document.getElementById("results").classList.remove("hidden");
      document.getElementById(`tab-${tab.dataset.tab}`).classList.remove("hidden");
    } else {
      // No video analyzed yet and this isn't Comparison -- show the upload
      // prompt instead of an empty/broken-looking tab.
      document.getElementById("results").classList.add("hidden");
      document.getElementById("upload-panel").classList.remove("hidden");
    }
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
  const total = Math.round(sec);
  const m = Math.floor(total / 60);
  const s = total % 60;
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
  hasAnalyzedVideo = true;
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
    const chakkarQuality = data.chakkar.quality_score;
    container.innerHTML += `
      <div class="score-card">
        <div class="label">Chakkar Quality</div>
        <div class="value ${scoreClass(chakkarQuality)}">${fmtScore(chakkarQuality)}${chakkarQuality !== null && chakkarQuality !== undefined ? "%" : ""}</div>
      </div>
      <div class="score-card">
        <div class="label">Chakkars Detected</div>
        <div class="value">${data.chakkar.event_count}</div>
      </div>`;
  }

  container.innerHTML += tempoCircleCard(data.timing ? data.timing.tempo_bpm : null);

  if (data.timing) {
    const timingAccuracy = data.timing.accuracy_score;
    container.innerHTML += `
      <div class="score-card">
        <div class="label">Timing Accuracy</div>
        <div class="value ${scoreClass(timingAccuracy)}">${fmtScore(timingAccuracy)}${timingAccuracy !== null && timingAccuracy !== undefined ? "%" : ""}</div>
      </div>`;
  }

  if (data.mudra && data.mudra.accuracy_score !== null && data.mudra.accuracy_score !== undefined) {
    const mudraAccuracy = data.mudra.accuracy_score;
    container.innerHTML += `
      <div class="score-card">
        <div class="label">Mudra Accuracy</div>
        <div class="value ${scoreClass(mudraAccuracy)}">${fmtScore(mudraAccuracy)}%</div>
      </div>
      <div class="score-card">
        <div class="label">Mudra Holds Identified</div>
        <div class="value">${data.mudra.events.length}</div>
      </div>`;
  }

  if (!data.chakkar && !data.timing && !data.mudra) {
    container.innerHTML += `<div class="empty-state">No chakkar, timing, or mudra signal detected in this clip.</div>`;
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
  el.innerHTML = `
    <p class="subtext">${chakkar.event_count} chakkar${chakkar.event_count === 1 ? "" : "s"} detected -- each scored separately, since they're distinct spin events, not one blended average.</p>
    ${chakkar.events.map(renderChakkarEvent).join("")}
    <h3>All flags</h3>
    ${renderFlagList(chakkar.flags)}
  `;
}

function renderChakkarEvent(event, index) {
  const r = event.result;
  const hasDrift = r.drift_shoulder_widths !== null && r.drift_shoulder_widths !== undefined;
  const driftHtml = hasDrift ? `
    <p class="subtext">
      Moved about ${(r.drift_shoulder_widths * 100).toFixed(0)}% of a shoulder-width from where this chakkar started
      (path straightness: ${r.path_straightness.toFixed(2)}, 1.0 = a straight line, lower = more back-and-forth).
      Shown for reference only -- this isn't scored. Traveling while spinning can be a deliberate choreography choice,
      not a mistake, and there's no way to tell intent from video alone.
    </p>` : "";
  return `
    <h3>Chakkar ${index + 1} <span class="subtext">(${formatTime(r.start_sec)}&ndash;${formatTime(r.end_sec)})</span></h3>
    <div class="score-grid">
      <div class="score-card"><div class="label">Quality</div><div class="value ${scoreClass(event.quality_score)}">${fmtScore(event.quality_score)}%</div></div>
      <div class="score-card"><div class="label">Raw count</div><div class="value">${r.raw_count.toFixed(2)}</div></div>
      <div class="score-card"><div class="label">Rounded (clean landing)</div><div class="value">${r.rounded_count.toFixed(1)}</div></div>
      <div class="score-card"><div class="label">Ending orientation gap</div><div class="value">${r.orientation_gap_deg.toFixed(0)}&deg;</div></div>
      <div class="score-card"><div class="label">Stop quality</div><div class="value">${r.controlled_stop ? "Controlled" : "Abrupt"}</div></div>
    </div>
    ${driftHtml}
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
  const content = document.getElementById("mudra-content");

  if (!mudra || !mudra.events || mudra.events.length === 0) {
    note.textContent = "No confidently-identified mudra holds found in this clip -- either no trained classifier exists yet, or no hand pose in this video was held steadily enough / matched a known mudra confidently enough to report. See methods.md for how this identification works and its known limits (trained on a Bharatanatyam photo dataset, not Kathak footage).";
    content.innerHTML = "";
    return;
  }

  note.textContent = "Identification is trained on a Bharatanatyam photo dataset (not Kathak footage) -- treat predictions as a first pass, not ground truth. See methods.md for known limits.";

  const rateHtml = mudra.accuracy_score === null || mudra.accuracy_score === undefined
    ? ""
    : `<div class="score-grid" style="margin-bottom: 16px;">
         <div class="score-card">
           <div class="label">Mudra Accuracy</div>
           <div class="value ${scoreClass(mudra.accuracy_score)}">${fmtScore(mudra.accuracy_score)}%</div>
         </div>
       </div>`;

  const eventsHtml = mudra.events.map(e => {
    const cls = e.mismatches && e.mismatches.length ? "flag-item" : "";
    const mismatchText = e.mismatches === null || e.mismatches === undefined
      ? "(no reference rule to check against)"
      : e.mismatches.length
        ? e.mismatches.join("; ")
        : "matches the reference shape";
    return `<div class="${cls}">
      ${formatTime(e.start_sec)}&ndash;${formatTime(e.end_sec)} (${e.hand_side} hand): <strong>${e.mudra}</strong>
      (confidence ${(e.confidence * 100).toFixed(0)}%) &mdash; ${mismatchText}
    </div>`;
  }).join("");

  content.innerHTML = `
    ${rateHtml}
    <h3>Flags</h3>
    ${renderFlagList(mudra.flags)}
    <h3>All identified holds</h3>
    ${eventsHtml}
  `;
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

document.getElementById("comparison-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const teacherInput = document.getElementById("teacher-video-input");
  const studentInput = document.getElementById("student-video-input");
  const status = document.getElementById("comparison-status");
  const results = document.getElementById("comparison-results");

  if (!teacherInput.files.length || !studentInput.files.length) {
    status.textContent = "Select both a teacher video and your own video to compare.";
    return;
  }

  const formData = new FormData();
  formData.append("teacher_video", teacherInput.files[0]);
  formData.append("student_video", studentInput.files[0]);

  status.textContent = "Comparing... this aligns both videos' audio and can take a minute or two.";
  results.classList.add("hidden");

  try {
    const resp = await fetch("/compare", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      status.textContent = `Error: ${data.error || "comparison failed"}`;
      return;
    }
    status.textContent = "";
    renderComparison(data);
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  }
});

function renderComparison(data) {
  const results = document.getElementById("comparison-results");
  results.classList.remove("hidden");

  const rateHtml = data.match_rate === null || data.match_rate === undefined
    ? ""
    : `<div class="score-grid" style="margin-bottom: 16px;">
         <div class="score-card">
           <div class="label">Actions matched</div>
           <div class="value ${scoreClass(data.match_rate)}">${fmtScore(data.match_rate)}%</div>
         </div>
       </div>`;

  const actionsHtml = (data.actions || []).map(a => {
    const cls = a.match ? "" : "flag-item";
    const status = a.match ? "match" : "mismatch";
    return `<div class="${cls}">
      ${formatTime(a.teacher_time)} (teacher) &rarr; ${formatTime(a.student_time)} (you):
      teacher had ${a.teacher_count}, you had ${a.student_count} &mdash; ${status}
    </div>`;
  }).join("");

  const denseSections = data.dense_sections || [];
  const extraTeacher = data.extra_teacher_sections || [];
  const extraStudent = data.extra_student_sections || [];
  const denseHtml = [
    ...denseSections.map(d => {
      const cls = d.match ? "" : "flag-item";
      const status = d.match ? "match" : "mismatch";
      return `<div class="${cls}">
        ${formatTime(d.teacher_time)} (teacher) &rarr; ${formatTime(d.student_time)} (you):
        teacher did ${d.teacher_count} strikes, you did ${d.student_count} (${d.ratio.toFixed(2)}x) &mdash; ${status}
      </div>`;
    }),
    ...extraTeacher.map(d => `<div class="flag-item">
      ${formatTime(d.start)} (teacher only): ${d.count} strikes with no matching section in your video
    </div>`),
    ...extraStudent.map(d => `<div class="flag-item">
      ${formatTime(d.start)} (you only): ${d.count} strikes with no matching section in the teacher's video
    </div>`),
  ].join("");

  results.innerHTML = `
    <h3>Summary</h3>
    ${rateHtml}
    <h3>Flags</h3>
    ${renderFlagList(data.flags)}
    <h3>Rhythmic / tatkaar sections</h3>
    <p class="subtext">Fast, repeated sections (like tatkaar) are compared by strike count, not matched sound-by-sound -- see methods.md for why.</p>
    ${denseHtml || '<div class="empty-state">No dense rhythmic sections detected.</div>'}
    <h3>All matched actions</h3>
    ${actionsHtml || '<div class="empty-state">No comparable movement-sound actions detected.</div>'}
  `;
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
