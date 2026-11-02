"""
Noise Augmentation Service
Generates calibrated noisy audio variants at specific Signal-to-Noise Ratios (SNR in dB).
Supports both synthetic noise (Gaussian white noise, pink/ambient noise)
and real-world noise audio profiles (control room background, crowd chatter, street sounds).
"""

import os
from typing import Optional, Dict
import numpy as np
import soundfile as sf
import librosa


class NoiseAugmenter:
    """Injects calibrated acoustic noise to test ASR model robustness across SNRs."""

    # Pre-set SNR benchmark conditions
    PRESETS = {
        "clean": None,
        "light_noise": 20.0,      # 20 dB SNR: Subtle background noise
        "moderate_noise": 10.0,   # 10 dB SNR: Standard noisy room / office
        "heavy_noise": 5.0,       # 5 dB SNR: Difficult / challenging environment
        "extreme_noise": 0.0,     # 0 dB SNR: Signal and noise power are equal
    }

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    @staticmethod
    def calculate_power(signal: np.ndarray) -> float:
        """Calculates mean square power of a signal."""
        return float(np.mean(signal**2))

    @staticmethod
    def generate_white_noise(length: int) -> np.ndarray:
        """Generates standard Gaussian white noise."""
        return np.random.normal(0, 1, length).astype(np.float32)

    @staticmethod
    def generate_pink_noise(length: int) -> np.ndarray:
        """
        Generates pink noise (1/f spectral density) which closely mimics
        natural ambient acoustic background environments.
        """
        white = np.random.normal(0, 1, length)
        # Apply FFT, filter 1/sqrt(f), and inverse FFT
        x_fft = np.fft.rfft(white)
        frequencies = np.fft.rfftfreq(length)
        frequencies[0] = 1e-6  # avoid division by zero at DC
        pink_fft = x_fft / np.sqrt(frequencies)
        pink = np.fft.irfft(pink_fft, n=length).astype(np.float32)
        return pink

    def load_noise_profile(self, noise_file_path: str, target_length: int) -> np.ndarray:
        """
        Loads an audio noise profile, resamples, and loops or trims
        to match the target length.
        """
        if not os.path.exists(noise_file_path):
            raise FileNotFoundError(f"Noise profile not found: {noise_file_path}")

        noise, sr = sf.read(noise_file_path, dtype="float32")
        if noise.ndim > 1:
            noise = np.mean(noise, axis=1)

        if sr != self.target_sr:
            noise = librosa.resample(noise, orig_sr=sr, target_sr=self.target_sr)

        if len(noise) < target_length:
            # Tile/repeat noise to match target length
            repeats = int(np.ceil(target_length / len(noise)))
            noise = np.tile(noise, repeats)

        # Randomly choose a start offset if noise is longer than signal
        max_offset = max(0, len(noise) - target_length)
        start_idx = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
        return noise[start_idx : start_idx + target_length].astype(np.float32)

    def add_noise(
        self,
        clean_audio: np.ndarray,
        snr_db: Optional[float],
        noise_profile_path: Optional[str] = None,
        noise_type: str = "pink",
    ) -> np.ndarray:
        """
        Mixes noise into clean audio to achieve the exact target SNR in dB.
        If snr_db is None, returns original audio unmodified.

        SNR formula:
        SNR_dB = 10 * log10(P_signal / (alpha^2 * P_noise))
        alpha = sqrt(P_signal / (P_noise * 10^(SNR_dB / 10)))
        """
        if snr_db is None:
            return clean_audio.copy()

        sig_power = self.calculate_power(clean_audio)
        if sig_power < 1e-8:
            return clean_audio.copy()

        length = len(clean_audio)

        # Generate or load noise
        if noise_profile_path and os.path.exists(noise_profile_path):
            noise = self.load_noise_profile(noise_profile_path, length)
        elif noise_type == "white":
            noise = self.generate_white_noise(length)
        else:
            noise = self.generate_pink_noise(length)

        noise_power = self.calculate_power(noise)
        if noise_power < 1e-8:
            return clean_audio.copy()

        # Compute scaling factor alpha
        snr_linear = 10.0 ** (snr_db / 10.0)
        alpha = np.sqrt(sig_power / (noise_power * snr_linear))

        # Mix and prevent hard clipping
        noisy_audio = clean_audio + alpha * noise
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val * 0.98

        return noisy_audio.astype(np.float32)

    def generate_conditions(
        self,
        clean_audio: np.ndarray,
        noise_profile_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Convenience generator that returns a dictionary of audio arrays
        for standard benchmark conditions: clean, light (20dB), moderate (10dB), heavy (5dB).
        """
        results = {}
        for condition_name, snr in self.PRESETS.items():
            results[condition_name] = self.add_noise(
                clean_audio=clean_audio,
                snr_db=snr,
                noise_profile_path=noise_profile_path,
            )
        return results
