"""
Evaluation API Routes
Upload audio and reference text to receive WER, CER, S/D/I alignment, RTF, and Entity Error Rate.
"""

import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.audio_processor import AudioProcessor
from app.services.asr_service import ASRService
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/api", tags=["Evaluation"])

_audio_proc = None
_asr = None
_eval_service = None


def get_services():
    global _audio_proc, _asr, _eval_service
    if _audio_proc is None:
        _audio_proc = AudioProcessor()
    if _asr is None:
        _asr = ASRService(default_model="tiny")
    if _eval_service is None:
        _eval_service = EvaluationService()
    return _audio_proc, _asr, _eval_service


@router.post("/evaluate")
async def evaluate_audio(
    file: UploadFile = File(...),
    reference: str = Form(...),
    model_name: str = Form("tiny"),
    normalize_text: bool = Form(True),
):
    audio_proc, asr, eval_service = get_services()

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

        # Run transcription
        result = asr.transcribe(tmp_path, model_name=model_name)
        prediction = result["text"]
        latency = result["latency_seconds"]

        # Evaluate against ground truth
        evaluation = eval_service.evaluate_pair(
            reference=reference,
            hypothesis=prediction,
            audio_duration=duration,
            latency_seconds=latency,
            normalize_text=normalize_text,
        )

        return {
            "status": "success",
            "model_name": model_name,
            "device": result["device"],
            "evaluation": evaluation,
        }
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
