"""
Audio Processor Service
Handles audio loading, mono conversion, 16kHz resampling, peak normalization,
duration validation, and audio feature extraction.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import soundfile as sf
import librosa


class AudioProcessor:
    """Preprocesses audio inputs to Whisper-standard 16kHz mono audio."""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file, converts it to mono if stereo/multichannel,
        resamples to target_sr (default 16kHz), and peak-normalizes.

        Returns:
            Tuple of (audio_array: np.ndarray [float32, 1D], sample_rate: int)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # First try soundfile for fast decoding
            data, sr = sf.read(file_path, dtype="float32")
            if data.ndim > 1:
                # Average channels for mono conversion
                data = np.mean(data, axis=1)

            # Resample if sample rate differs from target_sr
            if sr != self.target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
        except Exception:
            # Fallback to librosa's native loader
            data, sr = librosa.load(file_path, sr=self.target_sr, mono=True, dtype=np.float32)

        # Normalize audio amplitude safely
        data = self.normalize_amplitude(data)
        return data, sr

    @staticmethod
    def normalize_amplitude(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Peak-normalizes audio amplitude to avoid clipping while maximizing SNR."""
        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            return (audio / max_val) * target_peak
        return audio

    @staticmethod
    def save_audio(audio: np.ndarray, output_path: str, sr: int = 16000) -> str:
        """Saves a 1D float32 numpy array as a WAV file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Ensure float32 in [-1, 1] range
        clipped = np.clip(audio, -1.0, 1.0)
        sf.write(output_path, clipped, sr, subtype="PCM_16")
        return output_path

    def get_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """Calculates audio duration, sample rate, channels, RMS energy, and file size."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        file_size_bytes = os.path.getsize(file_path)
        data, _ = self.load_audio(file_path)
        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))

        try:
            info = sf.info(file_path)
            duration = float(info.duration)
            orig_sr = info.samplerate
            channels = info.channels
        except Exception:
            # Fallback for MP3 / WebM or containers unsupported by libsndfile
            orig_sr = self.target_sr
            channels = 1
            duration = len(data) / float(self.target_sr)

        return {
            "file_path": file_path,
            "duration_seconds": round(duration, 3),
            "original_sample_rate": orig_sr,
            "channels": channels,
            "file_size_kb": round(file_size_bytes / 1024.0, 2),
            "rms_energy": round(rms, 4),
            "peak_amplitude": round(peak, 4),
            "target_sample_rate": self.target_sr,
        }

    def validate_audio(
        self,
        file_path: str,
        min_duration: float = 0.4,
        max_duration: float = 300.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates if the audio file can be loaded, has valid non-zero duration,
        and is within operational duration limits.
        """
        if not os.path.exists(file_path):
            return False, f"File does not exist: {file_path}"

        try:
            meta = self.get_audio_metadata(file_path)
        except Exception as e:
            return False, f"Failed to read audio metadata: {str(e)}"

        duration = meta["duration_seconds"]
        if duration < min_duration:
            return False, f"Audio is too short ({duration:.2f}s < {min_duration}s minimum)"
        if duration > max_duration:
            return False, f"Audio exceeds maximum limit ({duration:.2f}s > {max_duration}s maximum)"
        if meta["rms_energy"] < 1e-4:
            return False, "Audio is nearly completely silent (RMS energy < 0.0001)"

        return True, None
