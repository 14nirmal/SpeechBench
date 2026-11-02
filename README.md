# 🎙️ SpeechBench: Speech Recognition Evaluation & Robustness Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%20%2B%20cu121-ee4c2c.svg)](https://pytorch.org/)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI-Whisper-brightgreen.svg)](https://github.com/openai/whisper)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard%20UI-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20Endpoints-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed%20(100%25)-success.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade **Automated Speech Recognition (ASR) Benchmarking & Evaluation Platform** engineered to assess speech-to-text models under real-world operational stress: calibrated acoustic noise degradation across Signal-to-Noise Ratios (SNR), speaker accent variation, latency distributions ($p_{50}$, $p_{95}$, $p_{99}$), Real-Time Factor (RTF), and mission-critical domain vocabulary (security control room dispatch, incident management, call centers).

---

## 📌 Executive Summary & Motivation (Beyond Simple Transcription)

In production voice AI systems (e.g., **Callease AI** control rooms, dispatch hubs, and conversational call centers), speech recognition cannot be evaluated solely on clean laboratory benchmarks. Standard applications convert speech to text; **SpeechBench** evaluates *how accurately, rapidly, and reliably* ASR models perform under real-world acoustic distortion.

### Core Evaluation Questions Addressed by SpeechBench:
1. **Acoustic Noise Robustness**: How severely does Word Error Rate (WER) degrade when background radio chatter, sirens, or control room noise increase from 20 dB to 5 dB SNR?
2. **Tail Latency & Real-Time Viability**: Does a larger model like Whisper Small violate latency budgets compared to Whisper Tiny or Base? How do $p_{95}$ and $p_{99}$ tail latencies behave under load?
3. **Domain Entity Accuracy (Beyond WER)**: Standard WER treats all word substitutions equally. In a security or emergency dispatch, transcribing *"Zone B"* as *"Zone D"* or misrecognizing an incident code has catastrophic operational consequences despite a low overall sentence WER. SpeechBench introduces **Entity Error Rate (EER)** to monitor critical operational keywords.

---

## 🏛️ System Architecture

```text
SpeechBench/
├── app/
│   ├── main.py                     # FastAPI application entrypoint with lifespan DB init
│   ├── api/
│   │   ├── routes_transcribe.py    # POST /api/transcribe (Whisper inference + RTF)
│   │   ├── routes_evaluate.py      # POST /api/evaluate (WER, CER, S/D/I alignment, EER)
│   │   └── routes_benchmark.py     # GET /api/benchmark/runs & /records
│   ├── services/
│   │   ├── audio_processor.py      # 16kHz mono resampling, peak normalization, RMS
│   │   ├── noise_augmenter.py      # Calibrated additive pink noise at exact SNR (dB)
│   │   ├── asr_service.py          # Whisper loader, warmup passes & CUDA synchronized timing
│   │   ├── evaluation_service.py   # JiWER WER/CER, token alignment, latency percentiles
│   │   └── normalizer.py           # Contraction expansion, number normalization, security entities
│   ├── database/
│   │   └── db.py                   # SQLite repository pattern (runs & evaluation_records)
│   └── dashboard/
│       └── app.py                  # Streamlit 5-tab intelligence dashboard
├── datasets/
│   ├── audio/
│   │   ├── clean/                  # Baseline read speech WAV recordings
│   │   └── domain_security/        # Security control room & dispatch WAV audio
│   ├── build_initial_dataset.py    # Dataset synthesis & metadata generator
│   └── metadata.csv                # Sample registry, transcripts, accents, conditions
├── experiments/
│   ├── run_benchmark.py            # CLI benchmark sweep across models & noise conditions
│   └── run_noise_experiments.py    # Standalone SNR degradation script
├── results/
│   ├── benchmark_results.csv       # Exported benchmark evaluations
│   └── speechbench.db              # SQLite persistent benchmark database
├── tests/
│   ├── test_normalizer.py          # Text normalization & entity extraction tests
│   ├── test_noise_augmenter.py     # SNR power calculation & pink noise tests
│   ├── test_metrics.py             # WER/CER/Levenshtein alignment & spec tests
│   └── test_api.py                 # FastAPI route handler & query filter tests
├── docs/
│   └── architecture_spec.md        # Detailed system design specification
├── pytest.ini                      # Pytest configuration
├── requirements.txt                # Production & test dependencies
├── run_api.bat                     # Quick launcher for FastAPI server
└── run_dashboard.bat               # Quick launcher for Streamlit dashboard
```

---

## 📐 Mathematical Formulation & Evaluation Metrics

### 1. Word Error Rate (WER) & Character Error Rate (CER)
Measures minimum edit distance (Levenshtein distance) at token and character levels normalized by the reference length:

$$\text{WER} = \frac{S + D + I}{N}$$

$$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$

- **$S$ (Substitutions)**: Words incorrectly replaced by the model.
- **$D$ (Deletions)**: Words present in the human reference omitted by the model.
- **$I$ (Insertions)**: Hallucinated words inserted by the model.
- **$N$**: Total word count in the human-verified reference transcript.

### 2. Real-Time Factor (RTF)
Measures operational speed relative to input audio duration:

$$\text{RTF} = \frac{T_{\text{processing}}}{T_{\text{audio}}}$$

- $\text{RTF} < 1.0$: Processing is faster than real-time (e.g., $\text{RTF} = 0.06$ means a 10-second call is transcribed in 0.6 seconds).

### 3. Latency Quantiles ($p_{50}, p_{95}, p_{99}$)
- **$p_{50}$ (Median)**: Typical transcription latency.
- **$p_{95} / p_{99}$ (Tail Latency)**: Response time experienced by the slowest 5% and 1% of calls, critical for voice agent timeouts and streaming SLA guarantees.

### 4. Acoustic Noise Degradation ($\Delta \text{WER}$)
Calibrated using Signal-to-Noise Ratio (SNR) in decibels:

$$\text{SNR}_{\text{dB}} = 10 \log_{10} \left( \frac{P_{\text{signal}}}{P_{\text{noise}}} \right)$$

$$\Delta \text{WER} = \text{WER}_{\text{noisy}} - \text{WER}_{\text{clean}}$$

| Condition | Target SNR | Description |
| :--- | :---: | :--- |
| **Clean** | Baseline | No injected noise; human-verified baseline audio |
| **Light Noise** | $20\text{ dB}$ | Subtle ambient office / air-conditioning background |
| **Moderate Noise** | $10\text{ dB}$ | Control room chatter, call center floor noise |
| **Heavy Noise** | $5\text{ dB}$ | High-interference radio static, vehicle sirens, street dispatch |

### 5. Domain Entity Error Rate (EER)
Evaluates extraction accuracy on mission-critical security keywords (`GATE_ID`, `INCIDENT_CODE`, `UNIT_CALLSIGN`, `LOCATION`):

$$\text{EER} = \frac{\text{Missed Entities}}{\text{Total Ground-Truth Entities}}$$

---

## 📊 Empirical Benchmark Results

Below is the verified performance baseline recorded using **Whisper Tiny** across 52 evaluations on an NVIDIA GeForce GTX 1650 (CUDA 12.1):

| Acoustic Condition | SNR (dB) | Mean WER | Mean CER | Mean Latency | Mean RTF | Speed Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Clean Baseline** | $\infty$ | **8.12%** | **2.65%** | $0.264\text{ s}$ | $0.052$ | ~19x faster than real-time |
| **Light Noise** | $20\text{ dB}$ | **11.45%** | **4.10%** | $0.278\text{ s}$ | $0.055$ | ~18x faster than real-time |
| **Moderate Noise** | $10\text{ dB}$ | **19.80%** | **7.84%** | $0.292\text{ s}$ | $0.058$ | ~17x faster than real-time |
| **Heavy Noise** | $5\text{ dB}$ | **30.82%** | **14.20%** | $0.318\text{ s}$ | $0.063$ | ~16x faster than real-time |

> **Key Observation**: While inference latency remains stable ($<0.32\text{s}$), Word Error Rate increases from **8.12%** in clean audio to **30.82%** at 5 dB SNR ($\Delta \text{WER} = +22.7\%$), showing the necessity of noise-robust frontends for mission-critical deployments.

---

## 🖥️ Interactive Dashboard Walkthrough

Launch the Streamlit intelligence suite via:
```bash
streamlit run app/dashboard/app.py
```

### The 5 Analysis Views:
1. **📊 Overview & KPIs**: High-level telemetry displaying total evaluations, model counts, mean WER/CER, average latency, and best-performing model recommendations.
2. **⚖️ Model Trade-off Matrix**: Pareto frontier scatter plot (Accuracy vs. Latency) and grouped bar charts visualizing $p_{50}$, $p_{95}$, and $p_{99}$ tail latency distributions.
3. **🔊 Noise & Accent Robustness**: Acoustic degradation curves charting $\Delta \text{WER}$ as a function of SNR decibels and breakdown by speaker accent.
4. **🔍 Error & Alignment Inspector**: Granular sample inspector with embedded audio playback, S/D/I counts, and color-coded token diffs (`Ref ➔ Hyp`).
5. **🧪 Live Playground**:
   - **In-Browser Microphone**: Record your own voice in real time using `st.audio_input`.
   - **Preset Verified Samples**: 6 preloaded clean and dispatch audio files.
   - **Live Noise Injector**: Real-time slider (0 to 30 dB SNR) with dual audio players (Clean vs. Synthesized Noisy).
   - **Domain Entity Chips**: Visual status badges showing whether critical callsigns or gate IDs were captured or lost.
   - **Persistent Logging**: Automatically records every live test directly into the SQLite benchmark database.

---

## 🔌 FastAPI REST Endpoints

Start the backend API via:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Interactive Swagger documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### API Reference:

#### 1. Transcribe Audio
- **Endpoint**: `POST /api/transcribe`
- **Payload**: `multipart/form-data` with `file` (WAV/MP3/FLAC), optional `model_name` (default: `"tiny"`).
- **Response**:
```json
{
  "status": "success",
  "text": "The security officer is at the main entrance.",
  "latency_seconds": 0.281,
  "rtf": 0.056,
  "audio_duration_seconds": 5.0,
  "detected_language": "en"
}
```

#### 2. Evaluate Audio Against Ground Truth
- **Endpoint**: `POST /api/evaluate`
- **Payload**: `multipart/form-data` with `file`, `reference` string, optional `model_name`.
- **Response**:
```json
{
  "status": "success",
  "raw_prediction": "The security officer is at the main entrance.",
  "normalized_reference": "the security officer is at the main entrance",
  "normalized_prediction": "the security officer is at the main entrance",
  "wer": 0.0,
  "cer": 0.0,
  "substitutions": 0,
  "deletions": 0,
  "insertions": 0,
  "hits": 8,
  "entity_evaluation": {
    "total_entities": 1,
    "matched_entities": 1,
    "entity_error_rate": 0.0
  }
}
```

#### 3. Benchmark Queries
- `GET /api/benchmark/runs` — Lists all completed benchmark sweeps.
- `GET /api/benchmark/records?model_name=tiny&condition=clean` — Queries filtered evaluation records.

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA drivers (optional, falls back gracefully to CPU)
- FFmpeg (bundled with `imageio-ffmpeg` or installed in system PATH)

### 2. Installation
```bash
# Clone repository
git clone https://github.com/your-username/SpeechBench.git
cd SpeechBench

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Verified Datasets
```bash
python datasets/build_initial_dataset.py
```

### 4. Run Benchmark Suite (CLI)
```bash
python experiments/run_benchmark.py --models tiny --noise-sweep --name "Production Baseline Sweep"
```

### 5. Launch UI & API Services
- **Launch Streamlit Dashboard**:
  ```bash
  .\run_dashboard.bat
  # or: streamlit run app/dashboard/app.py
  ```
- **Launch FastAPI REST Server**:
  ```bash
  .\run_api.bat
  # or: uvicorn app.main:app --port 8000
  ```

---

## 🧪 Automated Test Suite

SpeechBench includes comprehensive automated tests covering normalization, noise generation, error rate arithmetic, and API route responses.

Run all tests via Pytest:
```bash
pytest -v tests
```
*Result: 16 passed in ~2.8s (unit) / 25s (with PyTorch cold-start).*


---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
