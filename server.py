#!/usr/bin/env python3
from __future__ import annotations

import cgi
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "tmp_uploads"
WHISPER_ROOT = Path(os.getenv("WHISPER_ROOT", str(BASE_DIR / "whisper.cpp")))

WHISPER_BIN = os.getenv("WHISPER_BINARY", str(WHISPER_ROOT / "main"))
MODELS_DIR = os.getenv("WHISPER_MODELS_DIR", str(WHISPER_ROOT / "models"))
DEFAULT_MODEL = os.getenv("WHISPER_DEFAULT_MODEL", "base.en")
WHISPER_GIT = os.getenv("WHISPER_GIT_REPO", "https://github.com/ggerganov/whisper.cpp")
WHISPER_GGML_HOST = os.getenv("WHISPER_GGML_HOST", "https://huggingface.co")
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
SPEAKER_TAG_RE = re.compile(r"^\s*(?:\[[^\]]+\]\s*)?\(speaker\s+([^)]+)\)\s*(.*)\s*$")


ALLOWED_MODELS = {
    "tiny.en": "tiny.en",
    "base.en": "base.en",
    "small.en": "small.en",
    "medium.en": "medium.en",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
}

MODAL_MODELS = {
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
    "large-v3": "openai/whisper-large-v3",
    "distil-large-v3": "distil-whisper/distil-large-v3",
}

BOOTSTRAP_STATUS: Dict[str, object] = {
    "ok": False,
    "running": False,
    "model": "",
    "stage": "idle",
    "message": "Not started",
    "progress": 0,
    "error": "",
    "startedAt": 0.0,
    "elapsedSeconds": 0,
}


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _set_bootstrap_status(stage: str, message: str, progress: int, model: str | None = None, error: str = "") -> None:
    BOOTSTRAP_STATUS["stage"] = stage
    BOOTSTRAP_STATUS["message"] = message
    BOOTSTRAP_STATUS["progress"] = max(0, min(100, int(progress)))
    BOOTSTRAP_STATUS["error"] = error
    BOOTSTRAP_STATUS["ok"] = not bool(error)
    BOOTSTRAP_STATUS["running"] = stage not in {"idle", "done", "failed"}
    if model is not None:
        BOOTSTRAP_STATUS["model"] = model
    started = float(BOOTSTRAP_STATUS.get("startedAt", 0.0))
    if started:
        BOOTSTRAP_STATUS["elapsedSeconds"] = max(0, int(time.time() - started))


def _bootstrap_snapshot() -> Dict[str, object]:
    snapshot = dict(BOOTSTRAP_STATUS)
    started = float(snapshot.get("startedAt", 0.0))
    if started:
        snapshot["elapsedSeconds"] = max(0, int(time.time() - started))
    return snapshot


def _run_command(cmd: list[str], cwd: Path | None = None, timeout: int | None = 120) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "Command failed"
        raise RuntimeError(f"Command failed: {err}")
    return proc.stdout.strip()


def _model_path(model: str) -> Path:
    return Path(MODELS_DIR) / f"ggml-{model}.bin"


def _ensure_whisper_binary() -> None:
    whisper_binary = Path(WHISPER_BIN)
    if whisper_binary.exists():
        _set_bootstrap_status("binary_ready", "Whisper binary already exists", 95, model=BOOTSTRAP_STATUS.get("model"))
        return

    if whisper_binary == WHISPER_ROOT / "main":
        fallback_bins = [
            WHISPER_ROOT / "build" / "bin" / "whisper-cli",
            WHISPER_ROOT / "build" / "bin" / "main",
            WHISPER_ROOT / "bin" / "whisper-cli",
            WHISPER_ROOT / "bin" / "main",
        ]
        for fallback in fallback_bins:
            if fallback.exists():
                whisper_binary.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(fallback, whisper_binary)
                except Exception:
                    try:
                        whisper_binary.unlink()
                    except Exception:
                        pass
                    whisper_binary.symlink_to(fallback)
                _set_bootstrap_status("binary_ready", "Whisper binary ready from build output", 95, model=BOOTSTRAP_STATUS.get("model"))
                return

    if whisper_binary != WHISPER_ROOT / "main":
        raise RuntimeError(f"Whisper binary not found at {whisper_binary}.")

    _set_bootstrap_status("prepare_binary", "Checking build tools", 8, model=BOOTSTRAP_STATUS.get("model"))
    git = shutil.which("git")
    if not git:
        raise RuntimeError("git is required to download whisper.cpp.")

    cmake = shutil.which("cmake")
    make = shutil.which("make")

    if not WHISPER_ROOT.exists():
        _set_bootstrap_status("prepare_binary", "Cloning whisper.cpp repository", 20, model=BOOTSTRAP_STATUS.get("model"))
        _run_command([git, "clone", "--depth", "1", WHISPER_GIT, str(WHISPER_ROOT)], timeout=300)

    if cmake:
        _set_bootstrap_status("building", "Configuring with cmake", 40, model=BOOTSTRAP_STATUS.get("model"))
        _run_command([cmake, "-B", "build", "-S", "."], cwd=WHISPER_ROOT, timeout=600)
        _set_bootstrap_status("building", "Building whisper.cpp binary", 70, model=BOOTSTRAP_STATUS.get("model"))
        _run_command([cmake, "--build", "build", "-j"], cwd=WHISPER_ROOT, timeout=600)
        return

    if make:
        makefile = WHISPER_ROOT / "Makefile"
        if not makefile.exists():
            raise RuntimeError(
                "cmake is required to build whisper.cpp. makefile is not present for fallback build."
            )
        _set_bootstrap_status("building", "Building whisper.cpp binary with make", 70, model=BOOTSTRAP_STATUS.get("model"))
        _run_command([make, "-j"], cwd=WHISPER_ROOT, timeout=600)
        return

    raise RuntimeError("Install cmake (preferred) or make to build whisper.cpp.")


def _download_model(model: str) -> None:
    model_file = _model_path(model)
    if model_file.exists() and model_file.stat().st_size > 0:
        _set_bootstrap_status("model_ready", f"Model {model} already present", 100, model=model)
        return

    model_file.parent.mkdir(parents=True, exist_ok=True)
    default_model_dir = WHISPER_ROOT / "models"

    script = WHISPER_ROOT / "models" / "download-ggml-model.sh"
    if script.exists() and model_file.parent == default_model_dir:
        _set_bootstrap_status("download_model", f"Downloading model {model}", 80, model=model)
        _run_command(["bash", str(script), model], cwd=WHISPER_ROOT, timeout=600)
        if model_file.exists():
            _set_bootstrap_status("model_ready", f"Model {model} ready", 100, model=model)
            return

    _set_bootstrap_status("download_model", f"Downloading model {model} from remote", 80, model=model)
    model_url = f"{WHISPER_GGML_HOST}/ggerganov/whisper.cpp/resolve/main/{model_file.name}?download=true"
    tmp_file = model_file.with_suffix(".downloading")
    try:
        with urllib.request.urlopen(model_url, timeout=180) as response, open(tmp_file, "wb") as output:
            shutil.copyfileobj(response, output)
        if tmp_file.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty")
        tmp_file.replace(model_file)
        _set_bootstrap_status("model_ready", f"Model {model} ready", 100, model=model)
    except Exception as err:
        if tmp_file.exists():
            tmp_file.unlink()
        raise RuntimeError(f"Could not download model '{model}': {err}")


def _ensure_setup(model: str) -> None:
    if model not in ALLOWED_MODELS:
        raise RuntimeError(f"Unsupported model '{model}'.")
    _ensure_whisper_binary()
    _download_model(model)


def _bootstrap_if_needed(model: str) -> dict:
    if model not in ALLOWED_MODELS:
        raise RuntimeError(f"Unsupported model '{model}'.")

    _set_bootstrap_status("starting", f"Preparing whisper with {model}", 5, model=model)
    _ensure_setup(model)
    _set_bootstrap_status("done", f"Ready with model {model}", 100, model=model)
    return {
        "ok": True,
        "model": model,
        "ready": True,
        "whisperBinary": WHISPER_BIN,
        "modelPath": str(_model_path(model)),
    }


def _format_diarized_transcript(text: str) -> tuple[str, dict]:
    speaker_map: dict[str, str] = {}
    speaker_count = 0
    formatted_lines: list[str] = []
    segments: list[dict[str, str]] = []

    for line in text.splitlines():
        match = SPEAKER_TAG_RE.match(line.strip())
        if not match:
            if line.strip():
                formatted_lines.append(line.rstrip())
            continue

        raw_speaker = (match.group(1) or "").strip() or "unknown"
        spoken = (match.group(2) or "").strip()

        if raw_speaker not in speaker_map:
            speaker_count += 1
            speaker_map[raw_speaker] = f"person{speaker_count}"
        speaker_label = speaker_map[raw_speaker]

        segments.append({"speakerRaw": raw_speaker, "speaker": speaker_label, "text": spoken})
        if spoken:
            formatted_lines.append(f"[{speaker_label}] {spoken}")

    normalized = "\n".join(formatted_lines).strip()
    if not normalized:
        normalized = text.strip()

    return normalized, {"speakerMap": speaker_map, "speakerCount": speaker_count, "segments": segments}


def _transcribe_file(
    audio_path: Path, model: str, threads: int, out_prefix: Path, diarization: bool = False
) -> tuple[str, dict]:
    model_file = _model_path(model)
    if not model_file.exists():
        raise RuntimeError(f"Model file missing: {model_file}")
    if not Path(WHISPER_BIN).exists():
        raise RuntimeError(f"Whisper binary not found: {WHISPER_BIN}")

    base_cmd = [WHISPER_BIN, "-m", str(model_file), "-f", str(audio_path), "-t", str(threads)]

    commands = [
        base_cmd + (["-di"] if diarization else []) + ["-otxt", "-of", str(out_prefix)],
        base_cmd + (["-di"] if diarization else []) + ["-ng", "-otxt", "-of", str(out_prefix)],
        base_cmd + ["-otxt", "-of", str(out_prefix)],
        base_cmd + ["-ng", "-otxt", "-of", str(out_prefix)],
        base_cmd,
        base_cmd + ["-ng"],
    ]
    text = ""
    last_err = ""
    used_diarization = False
    for cmd in commands:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode == 0:
            out_txt = out_prefix.with_suffix(".txt")
            if out_txt.exists():
                text = out_txt.read_text(encoding="utf-8", errors="replace")
                used_diarization = used_diarization or "-di" in cmd
                break
            text = proc.stdout.strip()
            used_diarization = used_diarization or "-di" in cmd
            break
        last_err = proc.stderr.strip() or proc.stdout.strip() or "Unknown transcription error"

    if not text.strip():
        raise RuntimeError(last_err or "No transcript produced")

    if not used_diarization:
        return text.strip(), {"requested": bool(diarization), "applied": False, "speakerMap": {}, "speakerCount": 0}

    formatted, metadata = _format_diarized_transcript(text)
    speaker_count = int(metadata.get("speakerCount", 0))
    return formatted, {
        "requested": True,
        "applied": speaker_count > 0,
        "speakerMap": metadata.get("speakerMap", {}),
        "speakerCount": speaker_count,
        "segments": metadata.get("segments", []),
    }


def _transcribe_file_modal(
    audio_path: Path, model: str, diarization: bool = False
) -> tuple[str, dict]:
    """Send audio to Modal GPU for transcription. Audio is processed in memory only."""
    try:
        import modal
    except ImportError:
        raise RuntimeError("Modal is not installed. Run: pip install modal")

    audio_bytes = audio_path.read_bytes()
    hf_model_id = MODAL_MODELS.get(model, "openai/whisper-large-v3-turbo")

    transcribe_fn = modal.Function.from_name("scriber-gpu", "transcribe")
    result = transcribe_fn.remote(
        audio_bytes=audio_bytes, model_id=hf_model_id, diarization=diarization
    )

    if not result.get("ok"):
        raise RuntimeError(result.get("error", "Modal transcription failed"))

    diar_meta = result.get("diarization", {})
    meta = {
        "requested": diarization,
        "applied": diar_meta.get("applied", False),
        "speakerMap": diar_meta.get("speakerMap", {}),
        "speakerCount": diar_meta.get("speakerCount", 0),
        "segments": diar_meta.get("segments", []),
    }
    if diar_meta.get("error"):
        meta["error"] = diar_meta["error"]
    return result["transcript"], meta


def _prepare_audio(original: Path) -> Path:
    suffix = original.suffix.lower()
    if suffix in SUPPORTED_AUDIO_EXTS:
        return original

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            f"Cannot process '{original.name}'. Install ffmpeg to convert {suffix or 'this'} files to WAV before transcription."
        )

    converted = original.with_suffix(".wav")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(original),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(converted),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        if not converted.exists() or converted.stat().st_size == 0:
            raise RuntimeError("ffmpeg conversion failed: output file missing or empty")
        return converted
    except subprocess.CalledProcessError:
        raise RuntimeError(f"ffmpeg conversion failed for {original.name}")


def _cleanup_outputs(out_prefix: Path) -> None:
    for suffix in [".txt", ".srt", ".vtt", ".json"]:
        output_file = out_prefix.with_suffix(suffix)
        if output_file.exists():
            try:
                output_file.unlink()
            except OSError:
                pass


def _coerce_uploads(form: cgi.FieldStorage) -> tuple[list[tuple[str, cgi.FieldStorage]], list[str], dict[str, str]]:
    field_entries: list[tuple[str, cgi.FieldStorage]] = []

    for key in form.keys():
        value = form[key]
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if not isinstance(candidate, cgi.FieldStorage):
                continue
            if not getattr(candidate, "file", None):
                continue
            field_entries.append((key, candidate))

    explicit_audio = [entry for entry in field_entries if entry[0] == "audio"]
    if explicit_audio:
        uploads = explicit_audio
    else:
        uploads = field_entries

    seen_fields = sorted(set(form.keys()))
    model_field = form.getfirst("model", "")
    threads_field = form.getfirst("threads", "")
    debug_fields = {
        "fields": ", ".join(seen_fields) if seen_fields else "none",
        "field_count": str(len(field_entries)),
        "model": model_field or "missing",
        "threads": threads_field or "missing",
    }
    return uploads, seen_fields, debug_fields


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _send_json(self, code: int, payload: dict) -> None:
        data = _json_bytes(payload)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, code: int, message: str) -> None:
        self._send_json(code, {"ok": False, "error": message})

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/bootstrap/status":
            self._send_json(200, _bootstrap_snapshot())
            return

        if parsed.path == "/api/bootstrap":
            query = parse_qs(parsed.query or "")
            model = (query.get("model", [DEFAULT_MODEL])[0] or DEFAULT_MODEL).strip()
            if model not in ALLOWED_MODELS:
                self._send_error_json(400, f"Invalid model '{model}'. Allowed: {', '.join(ALLOWED_MODELS)}")
                return

            try:
                BOOTSTRAP_STATUS["startedAt"] = time.time()
                _set_bootstrap_status("starting", f"Preparing whisper with model {model}", 5, model=model)
                payload = _bootstrap_if_needed(model)
            except Exception as err:
                _set_bootstrap_status(
                    "failed",
                    f"Could not prepare whisper with {model}",
                    0,
                    model=model,
                    error=str(err),
                )
                self._send_error_json(500, str(err))
                return

            self._send_json(200, payload)
            return

        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/transcribe":
            self._send_error_json(404, "Endpoint not found")
            return

        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._send_error_json(400, "Expected multipart/form-data")
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": ctype,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )

        upload_items: list[tuple[str, cgi.FieldStorage]] = []
        raw_uploads = form.getlist("audio")
        if raw_uploads and all(isinstance(item, cgi.FieldStorage) for item in raw_uploads):
            upload_items = [("audio", item) for item in raw_uploads]
        else:
            # Fallback: accept any upload-like field if key differs.
            upload_items, seen_fields, debug_fields = _coerce_uploads(form)
            if not upload_items:
                form_debug = "; ".join(f"{k}={v}" for k, v in debug_fields.items())
                self._send_error_json(400, f"No audio files provided. Form debug: {form_debug}")
                return

        backend = (form.getfirst("backend", "local") or "local").strip().lower()
        model = form.getfirst("model", DEFAULT_MODEL)

        if backend == "modal":
            if model not in MODAL_MODELS:
                self._send_error_json(400, f"Invalid modal model '{model}'. Allowed: {', '.join(MODAL_MODELS)}")
                return
        else:
            if model not in ALLOWED_MODELS:
                self._send_error_json(400, f"Invalid model '{model}'. Allowed: {', '.join(ALLOWED_MODELS)}")
                return
            try:
                _ensure_setup(model)
            except Exception as err:
                self._send_error_json(500, str(err))
                return

        try:
            threads = int(form.getfirst("threads", "4"))
            threads = max(1, min(threads, 32))
        except ValueError:
            threads = 4
        diarization_raw = (form.getfirst("diarization", "0") or "").strip().lower()
        enable_diarization = diarization_raw in {"1", "true", "on", "yes"}

        UPLOAD_DIR.mkdir(exist_ok=True)
        results: List[Dict] = []
        had_input = False

        for _field_name, upload in upload_items:
            if not isinstance(upload, cgi.FieldStorage):
                continue
            file_name = getattr(upload, "filename", None) or f"{uuid.uuid4().hex}.m4a"
            file_stream = getattr(upload, "file", None)
            file_value = getattr(upload, "value", None)

            if file_stream is None and file_value in (None, ""):
                continue

            had_input = True
            input_path: Path | None = None
            converted_path: Path | None = None
            out_prefix: Path | None = None
            try:
                safe_name = Path(file_name).name or "audio"
                suffix = Path(safe_name).suffix or ".m4a"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as temp_file:
                    if file_stream is not None:
                        try:
                            file_stream.seek(0)
                        except Exception:
                            pass
                        shutil.copyfileobj(file_stream, temp_file)
                    else:
                        if isinstance(file_value, str):
                            temp_file.write(file_value.encode())
                        else:
                            temp_file.write(bytes(file_value or b""))
                    input_path = Path(temp_file.name)

                converted_path = _prepare_audio(input_path)
                if backend == "modal":
                    transcript, diarization_meta = _transcribe_file_modal(converted_path, model, enable_diarization)
                    out_prefix = None
                else:
                    out_prefix = UPLOAD_DIR / f"transcript-{uuid.uuid4().hex}"
                    transcript, diarization_meta = _transcribe_file(converted_path, model, threads, out_prefix, enable_diarization)
                result_entry = {
                    "filename": safe_name,
                    "ok": True,
                    "model": model,
                    "diarization": diarization_meta,
                    "transcript": transcript.strip(),
                }
                if backend != "modal":
                    result_entry["threads"] = threads
                results.append(result_entry)
            except Exception as err:
                results.append(
                    {
                        "filename": file_name,
                        "ok": False,
                        "error": str(err),
                    }
                )
            finally:
                if out_prefix is not None:
                    _cleanup_outputs(out_prefix)
                if input_path and input_path.exists():
                    input_path.unlink()
                if converted_path and converted_path != input_path and converted_path.exists():
                    converted_path.unlink()

        if not results and not had_input:
            self._send_error_json(400, "No valid uploaded files")
            return

        self._send_json(200, {"ok": True, "results": results})


def run() -> None:
    port = int(os.environ.get("SCRIBER_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Local transcriber running on http://127.0.0.1:{port}")
    print(f"Whisper root:  {WHISPER_ROOT}")
    print(f"Whisper bin:   {WHISPER_BIN}")
    print(f"Models dir:    {MODELS_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    run()
