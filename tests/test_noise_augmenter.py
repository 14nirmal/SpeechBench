"""
Tests for Noise Augmenter and SNR calculations
"""

import numpy as np
import pytest
from app.services.noise_augmenter import NoiseAugmenter


def test_calculate_power():
    signal = np.ones(1000, dtype=np.float32)
    power = NoiseAugmenter.calculate_power(signal)
    assert np.isclose(power, 1.0)


def test_pink_noise_generation():
    length = 16000
    pink = NoiseAugmenter.generate_pink_noise(length)
    assert len(pink) == length
    assert np.all(np.isfinite(pink))


def test_add_noise_snr():
    augmenter = NoiseAugmenter(target_sr=16000)
    # Generate 1-second pure tone
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    clean_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    noisy_audio = augmenter.add_noise(clean_audio, snr_db=10.0, noise_type="white")
    assert len(noisy_audio) == len(clean_audio)
    assert not np.array_equal(clean_audio, noisy_audio)

    # Check clean passthrough when snr is None
    clean_passthrough = augmenter.add_noise(clean_audio, snr_db=None)
    assert np.array_equal(clean_audio, clean_passthrough)
