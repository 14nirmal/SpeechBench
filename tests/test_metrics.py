"""
Tests for WER, CER, S/D/I alignments, RTF, and evaluation service
"""

import pytest
from app.services.evaluation_service import EvaluationService


def test_perfect_transcription_metrics():
    eval_service = EvaluationService()
    ref = "the security officer is at the main entrance"
    hyp = "the security officer is at the main entrance"

    res = eval_service.evaluate_pair(ref, hyp, audio_duration=5.0, latency_seconds=1.0)
    assert res["wer"] == 0.0
    assert res["cer"] == 0.0
    assert res["substitutions"] == 0
    assert res["deletions"] == 0
    assert res["insertions"] == 0
    assert res["rtf"] == 0.2


def test_example_from_spec():
    # Specification example:
    # Reference: "the security officer is at the main entrance" (8 words)
    # Prediction: "the security guard is at main entrance"
    # Substitutions: 1 ("officer" -> "guard")
    # Deletions: 1 ("the")
    # Insertions: 0
    # Expected WER = 2/8 = 25% (0.25)
    eval_service = EvaluationService()
    ref = "the security officer is at the main entrance"
    hyp = "the security guard is at main entrance"

    res = eval_service.evaluate_pair(ref, hyp, audio_duration=4.0, latency_seconds=1.0)
    assert res["substitutions"] == 1
    assert res["deletions"] == 1
    assert res["insertions"] == 0
    assert res["reference_words"] == 8
    assert res["wer"] == 0.25
    assert res["rtf"] == 0.25


def test_aggregate_metrics():
    results = [
        {"wer": 0.1, "cer": 0.05, "latency_seconds": 1.0, "rtf": 0.2, "reference_words": 10, "substitutions": 1, "deletions": 0, "insertions": 0},
        {"wer": 0.3, "cer": 0.15, "latency_seconds": 2.0, "rtf": 0.4, "reference_words": 10, "substitutions": 2, "deletions": 1, "insertions": 0},
    ]
    agg = EvaluationService.calculate_aggregate_metrics(results)
    assert agg["sample_count"] == 2
    assert agg["total_words"] == 20
    assert agg["mean_wer"] == 0.2
    assert agg["latency"]["mean"] == 1.5
    assert agg["latency"]["p50"] == 1.5
