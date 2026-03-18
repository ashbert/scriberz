const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("audioInput");
const dropText = document.getElementById("dropText");
const modelSelect = document.getElementById("modelSelect");
const threadsInput = document.getElementById("threads");
const diarizationToggle = document.getElementById("diarizationToggle");
const transcribeBtn = document.getElementById("transcribeBtn");
const status = document.getElementById("status");
const statusStage = document.getElementById("statusStage");
const statusTime = document.getElementById("statusTime");
const statusBar = document.getElementById("statusBar");
const results = document.getElementById("results");
const template = document.getElementById("resultTemplate");

let selectedFiles = [];
let bootstrapModel = "";
let isWhisperReady = false;
let bootstrapInFlight = null;
let bootstrapPollTimer = null;
let transcribePollTimer = null;
let transcribeStartAt = 0;
let transcribeFileCount = 0;

function formatElapsed(totalSeconds) {
  const mins = Math.floor(totalSeconds / 60);
  const secs = String(totalSeconds % 60).padStart(2, "0");
  return `${mins}:${secs}`;
}

function clearBootstrapPolling() {
  if (bootstrapPollTimer) {
    clearInterval(bootstrapPollTimer);
    bootstrapPollTimer = null;
  }
}

function clearTranscribeProgress() {
  if (transcribePollTimer) {
    clearInterval(transcribePollTimer);
    transcribePollTimer = null;
  }
}

function setTranscribeStatus(fileCount) {
  const elapsed = Math.floor((Date.now() - transcribeStartAt) / 1000);
  const stage = transcribePollTimer ? "transcribing" : "ready";
  const safeFiles = Math.max(1, Number(fileCount) || 1);
  const elapsedLabel = formatElapsed(elapsed);
  statusStage.textContent = "Transcribing";
  statusTime.textContent = elapsedLabel;
  const percent = Math.min(
    90,
    elapsed <= 20 ? 5 + Math.floor((elapsed / 20) * 40) : 45 + Math.floor((elapsed - 20) * 0.6)
  );
  setProgress(percent, stage);
  setStatus(`Transcribing ${safeFiles} file(s)... ${elapsedLabel}`);
}

function startTranscribeProgress(fileCount) {
  transcribeFileCount = Math.max(1, Number(fileCount) || 1);
  transcribeStartAt = Date.now();
  clearTranscribeProgress();
  setStatus(`Transcribing ${transcribeFileCount} file(s)... 00:00`);
  setProgress(5, "working");
  transcribePollTimer = setInterval(() => {
    setTranscribeStatus(transcribeFileCount);
  }, 800);
}

function finishTranscribeProgress(success = true, errorMessage = "") {
  clearTranscribeProgress();
  statusTime.textContent = formatElapsed(Math.floor((Date.now() - transcribeStartAt) / 1000));
  if (success) {
    setProgress(100, "done");
    statusStage.textContent = "Done";
    return;
  }
  setProgress(0, "failed");
  statusStage.textContent = "Failed";
  setStatus(errorMessage || "Transcription failed", true);
}

function toStageLabel(stage) {
  switch (stage) {
    case "starting":
      return "Starting";
    case "prepare_binary":
      return "Prepare Binary";
    case "building":
      return "Building";
    case "download_model":
      return "Downloading Model";
    case "binary_ready":
      return "Binary Ready";
    case "model_ready":
      return "Model Ready";
    case "done":
      return "Ready";
    case "failed":
      return "Failed";
    case "idle":
      return "Idle";
    default:
      return stage ? String(stage).replace(/_/g, " ") : "Working";
  }
}

function setProgress(percent, stage = "working") {
  const safePercent = Math.max(0, Math.min(100, Number(percent) || 0));
  statusBar.style.width = `${safePercent}%`;
  statusBar.classList.remove("ready", "failed");
  if (stage === "done") {
    statusBar.classList.add("ready");
  } else if (stage === "failed") {
    statusBar.classList.add("failed");
  }
}

function setStatus(message, isError = false) {
  status.textContent = message || "";
  status.style.color = isError ? "#9f1239" : "";
}

function setBootstrapState(payload, isError = false) {
  const message = payload?.message || "";
  const stage = (payload?.stage || "working").toLowerCase();
  const progress = Number(payload?.progress || 0);
  const model = payload?.model || modelSelect.value || "base.en";
  const elapsed = Number(payload?.elapsedSeconds || 0);

  setProgress(progress, stage);
  statusStage.textContent = toStageLabel(stage);
  statusTime.textContent = formatElapsed(elapsed);
  if (stage === "done" && !isError && !payload.running) {
    setStatus(`Ready to transcribe with ${model}`);
    transcribeBtn.disabled = false;
    return;
  }
  if (isError) {
    setStatus(`Bootstrap failed (${model}): ${message || "Unknown error"}`, true);
    return;
  }

  const percent = `${Math.max(0, Math.min(100, progress))}%`;
  if (message) {
    setStatus(`Preparing local whisper (${model}) — ${message} (${percent})`);
    return;
  }

  setStatus(`Preparing local whisper (${model})`);
}

async function pollBootstrapStatus() {
  try {
    const response = await fetch("/api/bootstrap/status");
    const payload = await response.json();
    if (!response.ok) {
      return;
    }
    const model = modelSelect.value || "base.en";
    if (payload.model && payload.model !== model) return;

    setBootstrapState(payload);

    const stage = (payload.stage || "").toLowerCase();
    if (payload.running === false && stage === "done") {
      isWhisperReady = true;
      bootstrapModel = payload.model || model;
      clearBootstrapPolling();
      return;
    }
    if (stage === "failed") {
      isWhisperReady = false;
      setBootstrapState(payload, true);
      clearBootstrapPolling();
    }
  } catch {
    // non-fatal, keep polling; status may be temporarily unavailable
  }
}

function startBootstrapPolling() {
  clearBootstrapPolling();
  pollBootstrapStatus();
  bootstrapPollTimer = setInterval(() => {
    pollBootstrapStatus();
  }, 800);
}

function updateDropLabel() {
  if (!selectedFiles.length) {
    dropText.textContent = "Drop files here or click to choose";
    return;
  }
  const names = selectedFiles.map((f) => f.name).slice(0, 3).join(", ");
  const suffix = selectedFiles.length > 3 ? ` +${selectedFiles.length - 3} more` : "";
  dropText.textContent = `${selectedFiles.length} file(s): ${names}${suffix}`;
}

function clearResults() {
  results.innerHTML = "";
}

function escapeFileName(fileName, ext) {
  const stem = fileName.replace(/\.[^/.]+$/, "");
  return `${stem}.${ext}`;
}

function renderResult(result) {
  const node = template.content.firstElementChild.cloneNode(true);
  const title = node.querySelector(".fileName");
  const meta = node.querySelector(".meta");
  const transcript = node.querySelector(".transcript");
  const copyBtn = node.querySelector(".copyBtn");
  const saveBtn = node.querySelector(".saveBtn");

  title.textContent = result.filename || "unknown";
  if (result.ok) {
    const metaBits = [`Model: ${result.model}`, `Threads: ${result.threads}`];
    if (result.diarization?.requested) {
      const applied = result.diarization?.applied ? "on" : "off";
      metaBits.push(`Speaker labels: ${applied}`);
      if (result.diarization?.speakerCount) {
        metaBits.push(`Speakers detected: ${result.diarization.speakerCount}`);
      }
    }
    meta.textContent = metaBits.join(" • ");
    transcript.textContent = result.transcript;
    copyBtn.addEventListener("click", async () => {
      await navigator.clipboard.writeText(result.transcript);
      copyBtn.textContent = "Copied";
      setTimeout(() => (copyBtn.textContent = "Copy"), 1200);
    });
    saveBtn.addEventListener("click", () => {
      const blob = new Blob([result.transcript], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = escapeFileName(result.filename || "transcript", "txt");
      anchor.click();
      URL.revokeObjectURL(url);
    });
  } else {
    node.classList.add("error");
    meta.classList.add("errorText");
    meta.textContent = result.error || "Transcription failed";
    transcript.textContent = "";
    copyBtn.disabled = true;
    saveBtn.disabled = true;
  }

  results.appendChild(node);
}

async function bootstrapWhisper() {
  const model = modelSelect.value || "base.en";
  if (isWhisperReady && bootstrapModel === model) {
    return;
  }
  if (bootstrapInFlight) {
    await bootstrapInFlight;
    if (isWhisperReady && bootstrapModel === model) return;
  }

  transcribeBtn.disabled = true;
  setBootstrapState({ message: `Preparing local whisper model (${model})...`, stage: "starting", progress: 5, model }, false);
  startBootstrapPolling();

  const request = (async () => {
    const response = await fetch(`/api/bootstrap?model=${encodeURIComponent(model)}`);
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Bootstrap request failed");
    }
    isWhisperReady = true;
    bootstrapModel = payload.model || model;
    setBootstrapState(
      { message: `Ready to transcribe with ${bootstrapModel}`, stage: "done", progress: 100, model: bootstrapModel, running: false },
      false
    );
    transcribeBtn.disabled = false;
  })();

  bootstrapInFlight = request;
  try {
    await request;
  } finally {
    if (bootstrapInFlight === request) {
      bootstrapInFlight = null;
    }
    clearBootstrapPolling();
  }
}

function bindBootstrapOnChange() {
  isWhisperReady = false;
  bootstrapModel = "";
  transcribeBtn.disabled = true;
  setStatus("Model changed. Re-initializing...");
  bootstrapWhisper().catch((err) => setBootstrapState({ message: err.message, stage: "failed", progress: 0, model: modelSelect.value || "base.en" }, true));
}

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => {
  selectedFiles = Array.from(event.target.files || []);
  updateDropLabel();
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  selectedFiles = Array.from(event.dataTransfer.files || []);
  fileInput.files = event.dataTransfer.files;
  updateDropLabel();
});

modelSelect.addEventListener("change", bindBootstrapOnChange);

transcribeBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) {
    setStatus("Select at least one audio file first.", true);
    return;
  }

  if (!isWhisperReady || bootstrapModel !== modelSelect.value) {
    try {
      await bootstrapWhisper();
    } catch (err) {
      setStatus(`Could not prepare whisper: ${err.message}`, true);
      return;
    }
  }

  transcribeBtn.disabled = true;
  startTranscribeProgress(selectedFiles.length);
  setStatus(`Transcribing ${selectedFiles.length} file(s)...`);
  clearResults();

  const form = new FormData();
  for (const file of selectedFiles) {
    const fileName = file?.name || `audio-${Date.now()}.m4a`;
  form.append("audio", file, fileName);
}
  form.append("model", modelSelect.value);
  form.append("threads", threadsInput.value || "4");
  form.append("diarization", diarizationToggle.checked ? "1" : "0");

  try {
    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: form,
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Transcription request failed");
    }
    for (const result of payload.results) {
      renderResult(result);
    }
    finishTranscribeProgress(true);
    setStatus(`Done. ${payload.results.length} file(s) processed.`);
  } catch (error) {
    finishTranscribeProgress(false, error.message);
    setStatus(`Error: ${error.message}`, true);
  } finally {
    transcribeBtn.disabled = false;
  }
});

bootstrapWhisper().catch((err) => setBootstrapState({ message: err.message, stage: "failed", progress: 0, model: modelSelect.value || "base.en" }, true));
