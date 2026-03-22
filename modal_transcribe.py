"""Modal GPU transcription backend for Scriber.

Deploy once:  modal deploy modal_transcribe.py
Test:         modal run modal_transcribe.py

For diarization, create a Modal secret with your HuggingFace token:
  modal secret create huggingface HF_TOKEN=hf_xxxxx

You must also accept the pyannote model terms at:
  https://huggingface.co/pyannote/speaker-diarization-3.1
  https://huggingface.co/pyannote/segmentation-3.0
"""
from __future__ import annotations

import modal

app = modal.App("scriber-gpu")

MODELS = [
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "distil-whisper/distil-large-v3",
]

whisper_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "soundfile",
        "librosa",
        "pyannote.audio",
    )
)


def _download_whisper_models():
    """Pre-download whisper model weights into the container image layer."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    for model_id in MODELS:
        AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        AutoProcessor.from_pretrained(model_id)


whisper_image = whisper_image.run_function(_download_whisper_models)

# Cache the pyannote diarization pipeline in memory across calls.
# Downloaded on first diarization request; stays loaded since min_containers=1.
_diar_pipeline_cache = None


def _align_speakers(whisper_chunks: list[dict], diarization) -> tuple[str, dict]:
    """Align whisper timestamp chunks with pyannote speaker segments.

    Returns formatted transcript and diarization metadata matching the local format.
    """
    speaker_map: dict[str, str] = {}
    speaker_count = 0
    segments: list[dict] = []
    lines: list[str] = []

    for chunk in whisper_chunks:
        ts = chunk.get("timestamp", (None, None))
        text = chunk.get("text", "").strip()
        if not text or ts[0] is None:
            continue

        mid = (ts[0] + (ts[1] or ts[0])) / 2.0
        raw_speaker = "unknown"
        # pyannote 4.x returns DiarizeOutput; extract the Annotation object
        annotation = getattr(diarization, "speaker_diarization", diarization)
        for seg, _track, spk in annotation.itertracks(yield_label=True):
            if seg.start <= mid <= seg.end:
                raw_speaker = spk
                break

        if raw_speaker not in speaker_map:
            speaker_count += 1
            speaker_map[raw_speaker] = f"person{speaker_count}"
        label = speaker_map[raw_speaker]

        segments.append({"speakerRaw": raw_speaker, "speaker": label, "text": text})

        if lines and lines[-1][0] == label:
            lines[-1] = (label, lines[-1][1] + " " + text)
        else:
            lines.append((label, text))

    formatted = "\n".join(f"[{lbl}] {txt}" for lbl, txt in lines).strip()
    meta = {
        "speakerMap": speaker_map,
        "speakerCount": speaker_count,
        "segments": segments,
    }
    return formatted, meta


def _get_diar_pipeline():
    """Lazy-load and cache the pyannote diarization pipeline."""
    global _diar_pipeline_cache
    if _diar_pipeline_cache is not None:
        return _diar_pipeline_cache

    import os

    import torch
    from pyannote.audio import Pipeline as PyannotePipeline

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Run: modal secret create huggingface HF_TOKEN=hf_xxx "
            "and accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    _diar_pipeline_cache = PyannotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _diar_pipeline_cache.to(torch.device(device))
    return _diar_pipeline_cache


@app.function(
    image=whisper_image,
    gpu="A10G",
    timeout=600,
    min_containers=1,
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def transcribe(
    audio_bytes: bytes,
    model_id: str = "openai/whisper-large-v3-turbo",
    diarization: bool = False,
) -> dict:
    """Transcribe audio entirely in memory. Never writes audio to disk."""
    import io

    import librosa
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    if model_id not in MODELS:
        return {"ok": False, "error": f"Unknown model: {model_id}"}

    try:
        buf = io.BytesIO(audio_bytes)
        audio_array, sr = librosa.load(buf, sr=16000, mono=True)
    except Exception as e:
        return {"ok": False, "error": f"Could not decode audio: {e}"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )

    result = pipe(
        {"array": audio_array, "sampling_rate": 16000},
        return_timestamps=True,
        generate_kwargs={"language": "en"},
    )

    transcript = result.get("text", "").strip()
    if not transcript:
        return {"ok": False, "error": "No transcript produced"}

    if not diarization:
        return {
            "ok": True,
            "transcript": transcript,
            "model": model_id,
            "diarization": {
                "requested": False,
                "applied": False,
                "speakerMap": {},
                "speakerCount": 0,
            },
        }

    # Run pyannote diarization
    try:
        diar_pipeline = _get_diar_pipeline()

        waveform = torch.from_numpy(audio_array).unsqueeze(0)
        diar_result = diar_pipeline({"waveform": waveform, "sample_rate": 16000})

        chunks = result.get("chunks", [])
        if not chunks:
            return {
                "ok": True,
                "transcript": transcript,
                "model": model_id,
                "diarization": {
                    "requested": True,
                    "applied": False,
                    "speakerMap": {},
                    "speakerCount": 0,
                    "error": "Whisper returned no timestamp chunks for alignment",
                },
            }

        formatted, meta = _align_speakers(chunks, diar_result)
        return {
            "ok": True,
            "transcript": formatted or transcript,
            "model": model_id,
            "diarization": {
                "requested": True,
                "applied": meta["speakerCount"] > 0,
                **meta,
            },
        }
    except Exception as e:
        # Diarization failed but transcription succeeded — return transcript without labels
        return {
            "ok": True,
            "transcript": transcript,
            "model": model_id,
            "diarization": {
                "requested": True,
                "applied": False,
                "speakerMap": {},
                "speakerCount": 0,
                "error": str(e),
            },
        }


@app.local_entrypoint()
def main():
    """Quick test: transcribe a short silence to verify the pipeline works."""
    import io

    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sr = 16000
    silence = np.zeros(sr * 2, dtype=np.float32)  # 2s silence
    sf.write(buf, silence, sr, format="WAV")
    result = transcribe.remote(audio_bytes=buf.getvalue())
    print(result)
