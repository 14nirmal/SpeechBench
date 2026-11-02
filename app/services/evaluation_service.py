"""
Evaluation Service
Calculates Word Error Rate (WER), Character Error Rate (CER),
detailed S/D/I alignment breakdowns, Real-Time Factor (RTF),
and latency quantiles (p50, p95, p99).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import jiwer
from app.services.normalizer import TextNormalizer


class EvaluationService:
    """Computes speech recognition accuracy, alignment errors, and latency metrics."""

    def __init__(self, normalizer: Optional[TextNormalizer] = None):
        self.normalizer = normalizer or TextNormalizer()

    def evaluate_pair(
        self,
        reference: str,
        hypothesis: str,
        audio_duration: float,
        latency_seconds: float,
        normalize_text: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluates a single prediction against its human ground truth.
        Computes WER, CER, S/D/I breakdown, RTF, and entity errors.
        """
        raw_reference = reference
        raw_hypothesis = hypothesis

        if normalize_text:
            ref_clean = self.normalizer.normalize(reference)
            hyp_clean = self.normalizer.normalize(hypothesis)
        else:
            ref_clean = reference.strip()
            hyp_clean = hypothesis.strip()

        # Handle edge case where reference is empty
        if not ref_clean:
            wer_val = 1.0 if hyp_clean else 0.0
            cer_val = 1.0 if hyp_clean else 0.0
            return {
                "wer": wer_val,
                "cer": cer_val,
                "substitutions": 0,
                "deletions": 0,
                "insertions": len(hyp_clean.split()) if hyp_clean else 0,
                "reference_words": 0,
                "raw_reference": raw_reference,
                "raw_hypothesis": raw_hypothesis,
                "normalized_reference": ref_clean,
                "normalized_hypothesis": hyp_clean,
                "audio_duration": round(audio_duration, 3),
                "latency_seconds": round(latency_seconds, 3),
                "rtf": round(latency_seconds / max(audio_duration, 1e-4), 3),
                "alignments": [],
                "entity_evaluation": {},
            }

        # Calculate WER and detailed alignments via jiwer
        out = jiwer.process_words(ref_clean, hyp_clean)
        wer_val = float(out.wer)
        substitutions = int(out.substitutions)
        deletions = int(out.deletions)
        insertions = int(out.insertions)
        hits = int(out.hits)
        ref_word_count = len(ref_clean.split())

        # Calculate CER
        cer_val = float(jiwer.cer(ref_clean, hyp_clean))

        # Build detailed alignment breakdown
        alignments = []
        for op in out.alignments[0]:
            chunk_type = op.type  # 'equal', 'substitute', 'delete', 'insert'
            ref_chunk = ref_clean.split()[op.ref_start_idx : op.ref_end_idx]
            hyp_chunk = hyp_clean.split()[op.hyp_start_idx : op.hyp_end_idx]
            alignments.append({
                "type": chunk_type,
                "reference": " ".join(ref_chunk),
                "hypothesis": " ".join(hyp_chunk),
            })

        # Calculate Real-Time Factor
        rtf = latency_seconds / max(audio_duration, 1e-4)

        # Domain Entity Evaluation (security keywords)
        entity_eval = self.normalizer.evaluate_entities(reference, hypothesis)

        return {
            "wer": round(wer_val, 4),
            "cer": round(cer_val, 4),
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "hits": hits,
            "reference_words": ref_word_count,
            "raw_reference": raw_reference,
            "raw_hypothesis": raw_hypothesis,
            "normalized_reference": ref_clean,
            "normalized_hypothesis": hyp_clean,
            "audio_duration": round(audio_duration, 3),
            "latency_seconds": round(latency_seconds, 3),
            "rtf": round(rtf, 3),
            "alignments": alignments,
            "entity_evaluation": entity_eval,
        }

    @staticmethod
    def calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates batch benchmark results into mean, std,
        and latency quantiles (p50, p90, p95, p99).
        """
        if not results:
            return {}

        wers = [r["wer"] for r in results]
        cers = [r["cer"] for r in results]
        latencies = [r["latency_seconds"] for r in results]
        rtfs = [r["rtf"] for r in results]

        total_words = sum(r["reference_words"] for r in results)
        total_subs = sum(r["substitutions"] for r in results)
        total_dels = sum(r["deletions"] for r in results)
        total_ins = sum(r["insertions"] for r in results)

        # Micro-average WER across all words
        corpus_wer = (total_subs + total_dels + total_ins) / max(total_words, 1)

        # Aggregate Entity Error Rate
        total_entities = sum(r.get("entity_evaluation", {}).get("total_entities", 0) for r in results)
        matched_entities = sum(r.get("entity_evaluation", {}).get("matched_entities", 0) for r in results)
        corpus_eer = (total_entities - matched_entities) / max(total_entities, 1) if total_entities > 0 else 0.0

        return {
            "sample_count": len(results),
            "total_words": total_words,
            "mean_wer": round(float(np.mean(wers)), 4),
            "corpus_wer": round(float(corpus_wer), 4),
            "mean_cer": round(float(np.mean(cers)), 4),
            "total_substitutions": total_subs,
            "total_deletions": total_dels,
            "total_insertions": total_ins,
            "latency": {
                "mean": round(float(np.mean(latencies)), 3),
                "std": round(float(np.std(latencies)), 3),
                "p50": round(float(np.percentile(latencies, 50)), 3),
                "p90": round(float(np.percentile(latencies, 90)), 3),
                "p95": round(float(np.percentile(latencies, 95)), 3),
                "p99": round(float(np.percentile(latencies, 99)), 3),
            },
            "rtf": {
                "mean": round(float(np.mean(rtfs)), 3),
                "p50": round(float(np.percentile(rtfs, 50)), 3),
                "p95": round(float(np.percentile(rtfs, 95)), 3),
            },
            "entity_error_rate": round(float(corpus_eer), 4),
            "total_domain_entities": total_entities,
        }
