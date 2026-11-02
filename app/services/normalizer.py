"""
Text Normalizer Service
Provides standardized text cleaning for ASR evaluation (WER/CER)
and domain-specific entity extraction (Security Control Rooms / Call Centers).
"""

import re
from typing import List, Dict, Any, Tuple


class TextNormalizer:
    """Standardizes transcripts for unbiased WER/CER comparison and entity tracking."""

    # Common English contractions mapping
    CONTRACTIONS = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am",
    }

    # Number words to digits mapping for common single digits and teens
    NUMBER_WORDS = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "first": "1st", "second": "2nd",
        "third": "3rd", "fourth": "4th", "fifth": "5th",
    }

    @classmethod
    def expand_contractions(cls, text: str) -> str:
        """Expands common contractions in English text."""
        result = text
        for contraction, replacement in cls.CONTRACTIONS.items():
            result = re.sub(re.escape(contraction), replacement, result, flags=re.IGNORECASE)
        return result

    @classmethod
    def normalize_numbers(cls, text: str) -> str:
        """Standardizes common number words to digits for fair comparison."""
        tokens = text.split()
        normalized_tokens = [cls.NUMBER_WORDS.get(token.lower(), token) for token in tokens]
        return " ".join(normalized_tokens)

    @classmethod
    def normalize(
        cls,
        text: str,
        strip_punctuation: bool = True,
        expand_contractions: bool = True,
        convert_numbers: bool = False,
    ) -> str:
        """
        Cleans and standardizes text:
        - Lowercases
        - Expands contractions (optional)
        - Removes punctuation (commas, periods, question marks, quotes, brackets, hyphens)
        - Normalizes multiple spaces into a single space
        """
        if not text:
            return ""

        text = text.lower().strip()

        if expand_contractions:
            text = cls.expand_contractions(text)

        if strip_punctuation:
            # Replace hyphens with spaces to keep compound words separated
            text = text.replace("-", " ").replace("_", " ")
            # Remove punctuation characters except alphanumeric and whitespace
            text = re.sub(r"[^\w\s]", "", text)

        if convert_numbers:
            text = cls.normalize_numbers(text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def extract_security_entities(text: str) -> List[Dict[str, Any]]:
        """
        Extracts domain-critical entities relevant to security control rooms,
        such as Gate/Door/Zone identifiers, emergency codes, building numbers, and callsigns.
        """
        entities = []
        patterns = [
            (r"\b(?:gate|door|zone|sector|bay|dock|camera|cam)\s+([a-z0-9]+)\b", "LOCATION_ID"),
            (r"\b(?:building|tower|block|level|floor)\s+([a-z0-9]+)\b", "BUILDING_LEVEL"),
            (r"\b(?:code|priority)\s+([a-z0-9]+)\b", "INCIDENT_CODE"),
            (r"\b(?:unit|officer|patrol|guard|car)\s+([a-z0-9]+)\b", "UNIT_CALLSIGN"),
            (r"\b\d{1,5}(?:\s+[a-z]+)?\s+(?:street|st|avenue|ave|road|rd|blvd|lane|way)\b", "ADDRESS"),
            (r"\b\d{1,2}:\d{2}(?:\s*(?:am|pm))?\b", "TIME_STAMP"),
        ]

        normalized_text = text.lower()
        for pattern, entity_type in patterns:
            for match in re.finditer(pattern, normalized_text):
                entities.append({
                    "entity_type": entity_type,
                    "matched_text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                })

        return entities

    @classmethod
    def evaluate_entities(
        cls,
        reference_text: str,
        predicted_text: str,
    ) -> Dict[str, Any]:
        """
        Compares critical security entities in the reference vs predicted text.
        Calculates Entity Error Rate (EER) specifically for mission-critical tokens.
        """
        ref_entities = cls.extract_security_entities(reference_text)
        if not ref_entities:
            return {
                "total_entities": 0,
                "matched_entities": 0,
                "missed_entities": [],
                "entity_error_rate": 0.0,
            }

        pred_normalized = cls.normalize(predicted_text)
        matched = 0
        missed = []

        for entity in ref_entities:
            entity_val = cls.normalize(entity["matched_text"])
            if entity_val in pred_normalized:
                matched += 1
            else:
                missed.append(entity["matched_text"])

        total = len(ref_entities)
        eer = (total - matched) / total if total > 0 else 0.0

        return {
            "total_entities": total,
            "matched_entities": matched,
            "missed_entities": missed,
            "entity_error_rate": round(eer, 4),
        }
