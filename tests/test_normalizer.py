"""
Tests for Text Normalizer and Entity Extraction
"""

import pytest
from app.services.normalizer import TextNormalizer


def test_basic_normalization():
    raw = "  The Security Officer is at the MAIN Entrance!  "
    clean = TextNormalizer.normalize(raw)
    assert clean == "the security officer is at the main entrance"


def test_contractions_expansion():
    raw = "They won't enter because he didn't confirm."
    clean = TextNormalizer.normalize(raw)
    assert "will not" in clean
    assert "did not" in clean


def test_punctuation_stripping():
    raw = "Alert: Zone-B, Gate #3 (Code-Red)."
    clean = TextNormalizer.normalize(raw)
    assert clean == "alert zone b gate 3 code red"


def test_extract_security_entities():
    sample = "Officer Davis dispatched to Gate 4 and Zone B for Code 3 at 10:30 am."
    entities = TextNormalizer.extract_security_entities(sample)
    types = [e["entity_type"] for e in entities]
    assert "LOCATION_ID" in types
    assert "INCIDENT_CODE" in types
    assert "UNIT_CALLSIGN" in types


def test_entity_error_rate():
    ref = "The alarm was triggered at Gate 4 by Unit 12."
    pred_correct = "The alarm was triggered at Gate 4 by Unit 12."
    pred_wrong_gate = "The alarm was triggered at Gate 2 by Unit 12."

    res_correct = TextNormalizer.evaluate_entities(ref, pred_correct)
    assert res_correct["entity_error_rate"] == 0.0

    res_wrong = TextNormalizer.evaluate_entities(ref, pred_wrong_gate)
    assert res_wrong["entity_error_rate"] > 0.0
    assert any("gate 4" in m for m in res_wrong["missed_entities"])
