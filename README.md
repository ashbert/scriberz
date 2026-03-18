# Local Apple Watch Audio Transcriber

A local-only web app that transcribes `.m4a` / `.wav` / `.mp3` / `.flac` / `.ogg` files using `whisper.cpp` in your browser UI.

No audio leaves your machine. No API keys required.

---

## Quick start

From this project root:

```bash
python3 server.py
```

Then open:

```
http://127.0.0.1:8000
```

### What happens on first run

- If `whisper.cpp` is not already built, the server clones/builds it automatically.
- If the selected model is not present, it is downloaded automatically.
- The UI shows progress while bootstrap runs.

The app now works with your existing local files and outputs text directly in the page.

---

## How to use

1. Open `http://127.0.0.1:8000`.
2. Drag and drop one or more audio files, or click the drop zone.
3. Choose a model preset:
   - `tiny.en` (fastest)
   - `base.en` (balanced)
   - `small.en` / `medium.en` / `large-v3` / `large-v3-turbo`
4. Set worker threads (1–32).
5. (Optional) enable **Speaker labels** to ask whisper for diarized output.
6. Click **Transcribe**.
7. Copy transcript to clipboard or download as `.txt`.

---

## Features

- Local-only transcription (no upload to cloud APIs).
- Automatic bootstrap for whisper binary + selected model.
- Progress bar/status for:
  - dependency/bootstrap stage,
  - transcription stage (including elapsed time).
- Drag-and-drop + multi-file support.
- Clipboard copy and `.txt` export.
- Optional speaker labels:
  - Output format becomes `[person1] ...`, `[person2] ...` when diarization is available.
  - Speaker labels are not real names; `whisper.cpp` diarization provides speaker IDs only.

---

## Apple Watch file transfer (practical note)

`scribe` only needs the audio file on your Mac. A quick path:

- On Watch app, choose the recording file in Files/Voice Memos, and AirDrop/send/share to your Mac.
- Save to a folder and drag into this app.

---

## Environment options

You can point the app to an existing whisper checkout:

```bash
export WHISPER_BINARY=/path/to/whisper.cpp/main
export WHISPER_MODELS_DIR=/path/to/whisper.cpp/models
export SCRIBER_PORT=8000
```

You can also start directly against a specific model:

```bash
http://127.0.0.1:8000/api/bootstrap?model=base.en
```

---

## Troubleshooting

- `cmake is required to build whisper.cpp`
  - Install cmake (`brew install cmake`) and rerun server.
- `Whisper binary not found`
  - Ensure the first run can build `whisper.cpp` or set `WHISPER_BINARY` manually.
- `No valid uploaded files`
  - Use the drop zone or file chooser; ensure file input actually contains audio.
- `No transcript produced`
  - Retry with a smaller model (`base.en`) and `-ng` fallback path already enabled in server.

---

## Optional future improvements

- Add optional name mapping (e.g., treat `person1` as Alice) on a per-file basis.
- Add a per-result speaker-JSON download format.
- Add a small batch queue/status for many long files.

---

## Files in repo

- `server.py` – local HTTP backend and whisper/bootstrap/transcription pipeline.
- `web/index.html` – UI structure.
- `web/styles.css` – UI styles.
- `web/app.js` – drag/drop, bootstrap status polling, transcription flow, actions.
