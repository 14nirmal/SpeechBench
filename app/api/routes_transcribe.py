"""
Transcription API Routes
Handles audio file uploads, model selection, and returns transcription with latency and RTF.
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.audio_processor import AudioProcessor
from app.services.asr_service import ASRService

router = APIRouter(prefix="/api", tags=["Transcription"])

# Lazy singletons
_audio_proc = None
_asr = None


def get_services():
    global _audio_proc, _asr
    if _audio_proc is None:
        _audio_proc = AudioProcessor()
    if _asr is None:
        _asr = ASRService(default_model="tiny")
    return _audio_proc, _asr


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_name: str = Form("tiny"),
    language: str = Form("en"),
):
    audio_proc, asr = get_services()

    # Save to a temporary file
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        is_valid, err_msg = audio_proc.validate_audio(tmp_path)
        if not is_valid:
            raise HTTPException(status_code=400, detail=err_msg)

        meta = audio_proc.get_audio_metadata(tmp_path)
        duration = meta["duration_seconds"]

        # Run synchronized inference
        result = asr.transcribe(tmp_path, model_name=model_name, language=language)
        latency = result["latency_seconds"]
        rtf = latency / max(duration, 1e-4)

        return {
            "status": "success",
            "model_name": result["model_name"],
            "device": result["device"],
            "transcription": result["text"],
            "detected_language": result["detected_language"],
            "duration_seconds": duration,
            "latency_seconds": latency,
            "rtf": round(rtf, 3),
            "rms_energy": meta["rms_energy"],
        }
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
