"""
ASR Inference Service
Loads and manages OpenAI Whisper models (tiny, base, small, etc.) with device auto-detection,
warmup passes, GPU synchronization, and accurate inference latency timing.
"""

import os
import sys
import time
from typing import Dict, Any, Optional
import numpy as np

# Ensure local venv/Scripts and imageio-ffmpeg binaries are in PATH
try:
    import imageio_ffmpeg
    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    scripts_dir = os.path.dirname(sys.executable)
    for p in [scripts_dir, ffmpeg_dir]:
        if p and p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import torch
import whisper


class ASRService:
    """Manages Whisper model loading, device dispatch, and synchronized inference."""

    AVAILABLE_MODELS = ["tiny", "base", "small", "medium"]

    def __init__(self, default_model: str = "base", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model_name: Optional[str] = None
        self.model = None
        self.warm_models = set()

        if default_model in self.AVAILABLE_MODELS:
            self.load_model(default_model)

    def load_model(self, model_name: str) -> None:
        """Loads a Whisper model if not already loaded into memory."""
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Choose from {self.AVAILABLE_MODELS}")

        if self.current_model_name == model_name and self.model is not None:
            return

        # Unload previous model to free VRAM
        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Loading Whisper model '{model_name}' on device '{self.device}'...")
        self.model = whisper.load_model(model_name, device=self.device)
        self.current_model_name = model_name

        # Perform a warm-up inference pass if this model has not been warmed up yet
        if model_name not in self.warm_models:
            self._warmup()
            self.warm_models.add(model_name)

    def _warmup(self) -> None:
        """Executes a short dummy inference to compile kernels and eliminate initial latency spike."""
        if self.model is None:
            return

        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        try:
            _ = self.model.transcribe(dummy_audio, language="en", fp16=(self.device == "cuda"))
            if self.device == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            print(f"Warmup notice: {e}")

    def transcribe(
        self,
        audio: Any,  # np.ndarray or str filepath
        model_name: Optional[str] = None,
        language: str = "en",
        fp16: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes the input audio and measures precise inference latency.

        Args:
            audio: Filepath string or 1D numpy float32 array at 16kHz.
            model_name: Whisper model size to use.
            language: Target transcription language code ('en').
            fp16: Whether to use FP16 half precision (defaults to True on CUDA).

        Returns:
            Dict containing transcription text, language, latency_seconds, and model name.
        """
        if model_name and model_name != self.current_model_name:
            self.load_model(model_name)

        if self.model is None:
            raise RuntimeError("No model is loaded.")

        if fp16 is None:
            fp16 = (self.device == "cuda")

        # Synchronize CUDA before starting the timer
        if self.device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # If filepath is given, decode with soundfile for fast in-memory processing
        if isinstance(audio, str) and os.path.exists(audio):
            try:
                import soundfile as sf
                data, sr = sf.read(audio, dtype="float32")
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if sr != 16000:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                audio = data
            except Exception:
                pass

        result = self.model.transcribe(
            audio,
            language=language,
            fp16=fp16,
            verbose=False,
        )

        # Synchronize CUDA before ending the timer for accurate GPU timing
        if self.device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        latency = end_time - start_time

        return {
            "text": result.get("text", "").strip(),
            "detected_language": result.get("language", language),
            "latency_seconds": round(latency, 4),
            "model_name": self.current_model_name,
            "device": self.device,
            "segments": result.get("segments", []),
        }

    def get_hardware_info(self) -> Dict[str, Any]:
        """Returns hardware specifications (CPU/GPU, VRAM) for benchmark reporting."""
        info = {
            "device": self.device,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        else:
            info["gpu_name"] = "None (CPU only)"
            info["gpu_vram_gb"] = 0.0

        return info
