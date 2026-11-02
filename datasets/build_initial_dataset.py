"""
Initial Dataset Generator
Generates curated audio recordings with exact ground-truth reference transcripts
using Windows native speech synthesis (male and female voices), covering:
1. Standard Clean Speech (LibriSpeech style sentences)
2. Security & Control Room Domain Speech (Callease AI target domain: gates, zones, units, codes)
3. Emergency Response & Incident Calls
Also prepares metadata.csv for automated benchmark sweeps.
"""

import os
import subprocess
import csv
from typing import List, Dict

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
CLEAN_DIR = os.path.join(AUDIO_DIR, "clean")
SECURITY_DIR = os.path.join(AUDIO_DIR, "domain_security")

# Curated benchmark test cases
BENCHMARK_SAMPLES: List[Dict[str, str]] = [
    # Clean baseline sentences
    {
        "id": "clean_001",
        "category": "clean",
        "voice": "Microsoft David Desktop",
        "accent": "US_standard",
        "reference": "the security officer is at the main entrance",
    },
    {
        "id": "clean_002",
        "category": "clean",
        "voice": "Microsoft Zira Desktop",
        "accent": "US_standard",
        "reference": "all visitors must register their identification at the reception desk",
    },
    {
        "id": "clean_003",
        "category": "clean",
        "voice": "Microsoft David Desktop",
        "accent": "US_standard",
        "reference": "the automated access control system verified the credentials successfully",
    },
    {
        "id": "clean_004",
        "category": "clean",
        "voice": "Microsoft Zira Desktop",
        "accent": "US_standard",
        "reference": "the perimeter sensor detected unusual movement along the north perimeter fence",
    },
    {
        "id": "clean_005",
        "category": "clean",
        "voice": "Microsoft David Desktop",
        "accent": "US_standard",
        "reference": "dispatch confirmation received for regular night patrol check",
    },
    # Security Control Room & Incident Dispatch (Callease AI key domain)
    {
        "id": "sec_001",
        "category": "security_control_room",
        "voice": "Microsoft David Desktop",
        "accent": "dispatch_male",
        "reference": "the alarm was triggered at gate 4 near zone b",
    },
    {
        "id": "sec_002",
        "category": "security_control_room",
        "voice": "Microsoft Zira Desktop",
        "accent": "dispatch_female",
        "reference": "code red reported at building 3 level 2 immediate backup requested",
    },
    {
        "id": "sec_003",
        "category": "security_control_room",
        "voice": "Microsoft David Desktop",
        "accent": "dispatch_male",
        "reference": "patrol unit 12 please respond to loading dock camera 5",
    },
    {
        "id": "sec_004",
        "category": "security_control_room",
        "voice": "Microsoft Zira Desktop",
        "accent": "dispatch_female",
        "reference": "unauthorized vehicle spotted outside sector 7 near westfield avenue",
    },
    {
        "id": "sec_005",
        "category": "security_control_room",
        "voice": "Microsoft David Desktop",
        "accent": "dispatch_male",
        "reference": "officer davis confirmed clear status at north checkpoint gate 2",
    },
    {
        "id": "sec_006",
        "category": "security_control_room",
        "voice": "Microsoft Zira Desktop",
        "accent": "dispatch_female",
        "reference": "fire alarm notification from server room bay 4 at 10:15 pm",
    },
    {
        "id": "sec_007",
        "category": "security_control_room",
        "voice": "Microsoft David Desktop",
        "accent": "dispatch_male",
        "reference": "maintenance team required at south tower elevator 3 immediately",
    },
    {
        "id": "sec_008",
        "category": "security_control_room",
        "voice": "Microsoft Zira Desktop",
        "accent": "dispatch_female",
        "reference": "incident reported at 742 evergreen terrace security dispatched",
    },
]


def synthesize_wav(text: str, output_wav_path: str, voice_name: str) -> bool:
    """Invokes PowerShell System.Speech synthesizer to generate a clean 16-bit WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
    ps_script = f"""
    Add-Type -AssemblyName System.Speech;
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
    $synth.SelectVoice('{voice_name}');
    $synth.SetOutputToWaveFile('{output_wav_path}');
    $synth.Speak('{text}');
    $synth.Dispose();
    """
    try:
        subprocess.run(["powershell", "-Command", ps_script], check=True, capture_output=True)
        return os.path.exists(output_wav_path)
    except Exception as e:
        print(f"Error synthesizing {output_wav_path}: {e}")
        return False


def build_dataset():
    """Generates WAV files and builds datasets/metadata.csv."""
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(SECURITY_DIR, exist_ok=True)

    records = []
    print("Generating speech benchmark dataset...")

    for item in BENCHMARK_SAMPLES:
        subfolder = CLEAN_DIR if item["category"] == "clean" else SECURITY_DIR
        wav_filename = f"{item['id']}.wav"
        full_wav_path = os.path.join(subfolder, wav_filename)

        rel_path = os.path.relpath(full_wav_path, os.path.dirname(DATASET_DIR)).replace("\\", "/")

        print(f"Synthesizing [{item['voice']}] -> {wav_filename}...")
        success = synthesize_wav(item["reference"], full_wav_path, item["voice"])
        if not success:
            print(f"Failed to generate {full_wav_path}")
            continue

        records.append({
            "sample_id": item["id"],
            "audio_path": rel_path,
            "reference": item["reference"],
            "language": "en",
            "accent": item["accent"],
            "condition": "clean",
            "domain_category": item["category"],
        })

    # Save metadata.csv
    meta_path = os.path.join(DATASET_DIR, "metadata.csv")
    if records:
        fieldnames = list(records[0].keys())
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    print(f"\n[SUCCESS] Generated {len(records)} test recordings and saved {meta_path}!")


if __name__ == "__main__":
    build_dataset()
