"""
Noise Robustness Experiments Runner
Evaluates ASR performance degradation across calibrated Signal-to-Noise Ratios:
- Clean (no added noise)
- Light Noise (20 dB SNR)
- Moderate Noise (10 dB SNR)
- Heavy Noise (5 dB SNR)
Computes Delta WER (WER_noisy - WER_clean) to assess acoustic resilience.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.audio_processor import AudioProcessor
from app.services.noise_augmenter import NoiseAugmenter
from app.services.asr_service import ASRService
from app.services.evaluation_service import EvaluationService
from app.database.db import BenchmarkRepository


def run_noise_experiment(
    metadata_path: str = "datasets/metadata.csv",
    model_name: str = "base",
    snr_levels: list = [20.0, 10.0, 5.0],
    output_csv: str = "results/noise_experiment_results.csv",
):
    print(f"\n============================================================")
    print(f" Starting Acoustic Noise Robustness Experiment: Whisper '{model_name}'")
    print(f" Tested SNR Levels: Clean, {snr_levels} dB")
    print(f"============================================================\n")

    audio_proc = AudioProcessor()
    noise_aug = NoiseAugmenter()
    eval_service = EvaluationService()
    asr = ASRService(default_model=model_name)
    repo = BenchmarkRepository()

    df_meta = pd.read_csv(metadata_path)
    records = []

    for idx, row in df_meta.iterrows():
        audio_path = row["audio_path"]
        reference = str(row["reference"])
        domain_cat = row.get("domain_category", "general")

        if not os.path.exists(audio_path):
            continue

        clean_audio, sr = audio_proc.load_audio(audio_path)
        duration = len(clean_audio) / float(sr)

        # Baseline Clean
        trans_clean = asr.transcribe(clean_audio, model_name=model_name)
        eval_clean = eval_service.evaluate_pair(reference, trans_clean["text"], duration, trans_clean["latency_seconds"])
        clean_wer = eval_clean["wer"]

        records.append({
            "model_name": model_name,
            "sample_id": row.get("sample_id", f"sample_{idx}"),
            "condition": "clean",
            "snr_db": None,
            "duration": duration,
            "latency": trans_clean["latency_seconds"],
            "wer": clean_wer,
            "cer": eval_clean["cer"],
            "delta_wer": 0.0,
            "prediction": trans_clean["text"],
            "reference": reference,
        })

        # Noisy variants across target SNRs
        for snr in snr_levels:
            noisy_audio = noise_aug.add_noise(clean_audio, snr_db=snr, noise_type="pink")
            trans_noisy = asr.transcribe(noisy_audio, model_name=model_name)
            eval_noisy = eval_service.evaluate_pair(reference, trans_noisy["text"], duration, trans_noisy["latency_seconds"])
            delta_wer = eval_noisy["wer"] - clean_wer

            records.append({
                "model_name": model_name,
                "sample_id": row.get("sample_id", f"sample_{idx}"),
                "condition": f"noisy_{int(snr)}dB",
                "snr_db": snr,
                "duration": duration,
                "latency": trans_noisy["latency_seconds"],
                "wer": eval_noisy["wer"],
                "cer": eval_noisy["cer"],
                "delta_wer": round(delta_wer, 4),
                "prediction": trans_noisy["text"],
                "reference": reference,
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Noise robustness results exported to '{output_csv}'!")

    # Summary table
    print("\n================ NOISE DEGRADATION SUMMARY ================")
    summary = df.groupby("condition").agg(
        samples=("wer", "count"),
        mean_wer=("wer", lambda x: f"{x.mean()*100:.2f}%"),
        mean_delta_wer=("delta_wer", lambda x: f"+{x.mean()*100:.2f}%" if x.mean() >= 0 else f"{x.mean()*100:.2f}%"),
        mean_cer=("cer", lambda x: f"{x.mean()*100:.2f}%"),
        mean_latency=("latency", lambda x: f"{x.mean():.3f}s"),
    )
    print(summary.to_string())
    print("===========================================================\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ASR Noise Robustness Experiment")
    parser.add_argument("--metadata", default="datasets/metadata.csv", help="Metadata path")
    parser.add_argument("--model", default="base", help="Whisper model to test")
    args = parser.parse_args()

    run_noise_experiment(metadata_path=args.metadata, model_name=args.model)
