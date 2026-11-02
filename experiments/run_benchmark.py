"""
Benchmark Runner CLI
Automates evaluation sweeps across models (tiny, base, small), conditions (clean, noise SNRs),
and dataset splits. Computes distributions (mean, p50, p95, p99), logs to SQLite, and exports CSV.
"""

import os
import sys
import argparse
import pandas as pd
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.audio_processor import AudioProcessor
from app.services.noise_augmenter import NoiseAugmenter
from app.services.asr_service import ASRService
from app.services.evaluation_service import EvaluationService
from app.database.db import BenchmarkRepository


def run_benchmark(
    metadata_path: str = "datasets/metadata.csv",
    models: Optional[List[str]] = None,
    output_csv: str = "results/benchmark_results.csv",
    run_name: str = "ASR Benchmark Run",
    apply_noise_sweep: bool = False,
    device: Optional[str] = None,
):
    models = models or ["tiny", "base"]
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)
    print(f"\n=======================================================")
    print(f" Starting SpeechBench Benchmark: '{run_name}'")
    print(f" Datasets samples: {len(df_meta)} | Models: {models}")
    print(f"=======================================================\n")

    audio_proc = AudioProcessor()
    noise_aug = NoiseAugmenter()
    eval_service = EvaluationService()
    asr = ASRService(default_model=models[0], device=device)
    repo = BenchmarkRepository()

    hw_info = asr.get_hardware_info()
    print(f"Hardware execution environment: {hw_info}\n")

    records = []

    for model_name in models:
        print(f"\n>>> Benchmarking Model: Whisper '{model_name}' on {asr.device}...")
        asr.load_model(model_name)

        for idx, row in df_meta.iterrows():
            audio_path = row["audio_path"]
            reference = str(row["reference"])
            condition = row.get("condition", "clean")
            accent = row.get("accent", "standard")
            domain_cat = row.get("domain_category", "general")

            if not os.path.exists(audio_path):
                print(f"Skipping missing audio: {audio_path}")
                continue

            # Load audio and determine duration
            audio_data, sr = audio_proc.load_audio(audio_path)
            duration = len(audio_data) / float(sr)

            # Test conditions: if apply_noise_sweep is requested on clean samples, evaluate across SNRs
            test_variants = [(condition, audio_data, None)]
            if apply_noise_sweep and condition == "clean":
                test_variants.extend([
                    ("light_noise", noise_aug.add_noise(audio_data, snr_db=20.0), 20.0),
                    ("moderate_noise", noise_aug.add_noise(audio_data, snr_db=10.0), 10.0),
                    ("heavy_noise", noise_aug.add_noise(audio_data, snr_db=5.0), 5.0),
                ])

            for cond_name, audio_input, snr_val in test_variants:
                # Synchronized ASR inference
                transcription = asr.transcribe(audio_input, model_name=model_name)
                prediction = transcription["text"]
                latency = transcription["latency_seconds"]

                # Metric computation
                eval_res = eval_service.evaluate_pair(
                    reference=reference,
                    hypothesis=prediction,
                    audio_duration=duration,
                    latency_seconds=latency,
                    normalize_text=True,
                )

                rec = {
                    "model_name": model_name,
                    "audio_path": audio_path,
                    "condition": cond_name,
                    "snr_db": snr_val,
                    "accent": accent,
                    "domain_category": domain_cat,
                    "duration_seconds": duration,
                    "latency_seconds": latency,
                    "rtf": eval_res["rtf"],
                    "raw_reference": reference,
                    "raw_prediction": prediction,
                    "normalized_reference": eval_res["normalized_reference"],
                    "normalized_prediction": eval_res["normalized_hypothesis"],
                    "wer": eval_res["wer"],
                    "cer": eval_res["cer"],
                    "substitutions": eval_res["substitutions"],
                    "deletions": eval_res["deletions"],
                    "insertions": eval_res["insertions"],
                    "hits": eval_res["hits"],
                    "reference_words": eval_res["reference_words"],
                    "entity_error_rate": eval_res["entity_evaluation"].get("entity_error_rate", 0.0),
                    "alignments": eval_res["alignments"],
                }
                records.append(rec)

    # Save to SQLite
    run_id = repo.create_run(
        run_name=run_name,
        device=hw_info["device"],
        gpu_name=hw_info.get("gpu_name", "None"),
        models=models,
        total_samples=len(records),
    )
    repo.insert_records_batch(run_id, records)

    # Export to CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_results = pd.DataFrame(records)
    # Drop complex object columns for CSV export
    df_results_export = df_results.drop(columns=["alignments"], errors="ignore")
    df_results_export.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Results exported to '{output_csv}' and SQLite database.")

    # Print summary statistics
    print("\n================ SUMMARY BENCHMARK REPORT ================")
    summary = df_results.groupby(["model_name", "condition"]).agg(
        samples=("wer", "count"),
        mean_wer=("wer", lambda x: f"{x.mean()*100:.2f}%"),
        mean_cer=("cer", lambda x: f"{x.mean()*100:.2f}%"),
        mean_latency=("latency_seconds", lambda x: f"{x.mean():.3f}s"),
        p50_latency=("latency_seconds", lambda x: f"{x.quantile(0.50):.3f}s"),
        p95_latency=("latency_seconds", lambda x: f"{x.quantile(0.95):.3f}s"),
        mean_rtf=("rtf", lambda x: f"{x.mean():.3f}"),
    )
    print(summary.to_string())
    print("==========================================================\n")
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpeechBench ASR Benchmark")
    parser.add_argument("--metadata", default="datasets/metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--models", nargs="+", default=["tiny", "base"], help="Whisper models to test")
    parser.add_argument("--output", default="results/benchmark_results.csv", help="Output CSV path")
    parser.add_argument("--noise-sweep", action="store_true", help="Generate 20dB/10dB/5dB noisy variants")
    parser.add_argument("--name", default="Benchmark Run", help="Run name")
    args = parser.parse_args()

    run_benchmark(
        metadata_path=args.metadata,
        models=args.models,
        output_csv=args.output,
        run_name=args.name,
        apply_noise_sweep=args.noise_sweep,
    )
