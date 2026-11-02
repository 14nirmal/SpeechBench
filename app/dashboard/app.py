"""
SpeechBench Streamlit Intelligence Dashboard
Interactive visualization of ASR benchmarking, model comparisons (Tiny/Base/Small),
noise degradation curves across SNRs, tail latency distributions, and live audio testing.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.database.db import BenchmarkRepository
from app.services.audio_processor import AudioProcessor
from app.services.noise_augmenter import NoiseAugmenter
from app.services.asr_service import ASRService
from app.services.evaluation_service import EvaluationService
from app.services.normalizer import TextNormalizer

st.set_page_config(
    page_title="SpeechBench | ASR Evaluation Platform",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: #1e222d;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2e3440;
        margin-bottom: 12px;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #4f8bf9;
    }
    .metric-label {
        font-size: 13px;
        color: #9aa0a6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .highlight-sub { color: #f59e0b; font-weight: 600; }
    .highlight-del { color: #ef4444; font-weight: 600; text-decoration: line-through; }
    .highlight-ins { color: #10b981; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


import io

@st.cache_resource
def load_services():
    """Initializes reusable backend services."""
    repo = BenchmarkRepository()
    audio_proc = AudioProcessor()
    noise_aug = NoiseAugmenter()
    eval_service = EvaluationService()
    asr = ASRService(default_model="tiny")
    return repo, audio_proc, noise_aug, eval_service, asr


repo, audio_proc, noise_aug, eval_service, asr_engine = load_services()

# Header
st.title("🎙️ SpeechBench: Speech Recognition Evaluation Platform")
st.caption("Benchmarking ASR models on noise robustness, accent diversity, latency distributions, and domain intelligence.")

# Sidebar Filters & Navigation
st.sidebar.title("Configuration & Filters")

# Load records from DB or CSV fallback
db_records = repo.get_all_records()
if db_records:
    df_all = pd.DataFrame(db_records)
elif os.path.exists(os.path.join(ROOT_DIR, "results", "benchmark_results.csv")):
    df_all = pd.read_csv(os.path.join(ROOT_DIR, "results", "benchmark_results.csv"))
else:
    df_all = pd.DataFrame()

# Navigation tabs
tabs = st.tabs([
    "📊 Overview & KPIs",
    "⚖️ Model Trade-off Matrix",
    "🔊 Noise & Accent Robustness",
    "🔍 Error & Alignment Inspector",
    "🧪 Live Playground",
])

# ---------------- TAB 1: OVERVIEW & KPIS ----------------
with tabs[0]:
    if df_all.empty:
        st.info("No benchmark results found yet. Run `python experiments/run_benchmark.py` or use the Playground to generate data.")
    else:
        # Top-level metrics
        total_evals = len(df_all)
        unique_models = df_all["model_name"].nunique()
        avg_wer = df_all["wer"].mean() * 100.0
        avg_cer = df_all["cer"].mean() * 100.0
        avg_latency = df_all["latency_seconds"].mean()
        avg_rtf = df_all["rtf"].mean()

        # Best model
        model_scores = df_all.groupby("model_name")["wer"].mean()
        best_model = model_scores.idxmin()
        best_wer = model_scores.min() * 100.0

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Evaluations", f"{total_evals:,}")
        with col2:
            st.metric("Models Benchmarked", f"{unique_models}")
        with col3:
            st.metric("Mean WER", f"{avg_wer:.2f}%")
        with col4:
            st.metric("Mean Latency", f"{avg_latency:.3f}s", delta=f"RTF {avg_rtf:.2f}")
        with col5:
            st.metric("Best Model (Lowest WER)", f"{best_model}", delta=f"{best_wer:.2f}% WER")

        st.markdown("---")
        st.subheader("Model Summary Statistics")
        summary_df = df_all.groupby("model_name").agg(
            evaluations=("wer", "count"),
            mean_wer=("wer", lambda x: f"{x.mean()*100:.2f}%"),
            mean_cer=("cer", lambda x: f"{x.mean()*100:.2f}%"),
            p50_latency=("latency_seconds", lambda x: f"{x.quantile(0.50):.3f}s"),
            p95_latency=("latency_seconds", lambda x: f"{x.quantile(0.95):.3f}s"),
            mean_rtf=("rtf", lambda x: f"{x.mean():.3f}"),
            eer=("entity_error_rate", lambda x: f"{x.mean()*100:.2f}%" if "entity_error_rate" in df_all.columns else "N/A"),
        ).reset_index()
        st.dataframe(summary_df, use_container_width=True)

# ---------------- TAB 2: MODEL COMPARISON ----------------
with tabs[1]:
    if not df_all.empty:
        st.subheader("Accuracy vs. Speed Trade-off (Pareto Frontier)")
        st.caption("Identify optimal balance between transcription accuracy (WER) and inference latency / Real-Time Factor (RTF).")

        fig_scatter = px.scatter(
            df_all,
            x="latency_seconds",
            y="wer",
            color="model_name",
            size="duration_seconds",
            hover_data=["condition", "raw_reference", "raw_prediction"],
            labels={"latency_seconds": "Latency (seconds)", "wer": "Word Error Rate (WER)", "model_name": "Model"},
            title="Per-Sample Latency vs. WER",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("WER & CER by Model")
            model_metrics = df_all.groupby("model_name")[["wer", "cer"]].mean().reset_index()
            fig_bar = go.Figure(data=[
                go.Bar(name='WER', x=model_metrics['model_name'], y=model_metrics['wer']),
                go.Bar(name='CER', x=model_metrics['model_name'], y=model_metrics['cer']),
            ])
            fig_bar.update_layout(barmode='group', yaxis_title="Error Rate (0.0 to 1.0)")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            st.subheader("Tail Latency Distribution (p50 / p95 / p99)")
            lat_stats = []
            for m in df_all["model_name"].unique():
                m_lat = df_all[df_all["model_name"] == m]["latency_seconds"]
                lat_stats.append({
                    "model": m,
                    "p50": m_lat.quantile(0.50),
                    "p95": m_lat.quantile(0.95),
                    "p99": m_lat.quantile(0.99),
                })
            df_lat = pd.DataFrame(lat_stats)
            fig_lat = go.Figure(data=[
                go.Bar(name='p50 (Median)', x=df_lat['model'], y=df_lat['p50']),
                go.Bar(name='p95', x=df_lat['model'], y=df_lat['p95']),
                go.Bar(name='p99', x=df_lat['model'], y=df_lat['p99']),
            ])
            fig_lat.update_layout(barmode='group', yaxis_title="Latency (seconds)")
            st.plotly_chart(fig_lat, use_container_width=True)

# ---------------- TAB 3: ROBUSTNESS ANALYSIS ----------------
with tabs[2]:
    if not df_all.empty:
        st.subheader("Noise Robustness Across Signal-to-Noise Ratios (SNR)")
        st.caption("Quantifies performance degradation as background noise increases: Clean -> 20dB -> 10dB -> 5dB SNR.")

        noise_df = df_all.groupby(["model_name", "condition"])["wer"].mean().reset_index()
        # Sort conditions logically
        cond_order = ["clean", "light_noise", "moderate_noise", "heavy_noise", "extreme_noise"]
        noise_df["sort_key"] = noise_df["condition"].apply(lambda x: cond_order.index(x) if x in cond_order else 99)
        noise_df = noise_df.sort_values("sort_key")

        fig_noise = px.line(
            noise_df,
            x="condition",
            y="wer",
            color="model_name",
            markers=True,
            title="Acoustic Noise Degradation Curve (WER vs. Condition)",
            labels={"wer": "Word Error Rate", "condition": "Acoustic Condition"},
        )
        st.plotly_chart(fig_noise, use_container_width=True)

        if "accent" in df_all.columns and df_all["accent"].nunique() > 1:
            st.subheader("Performance Across Accent Groups")
            accent_df = df_all.groupby(["model_name", "accent"])["wer"].mean().reset_index()
            fig_accent = px.bar(
                accent_df,
                x="accent",
                y="wer",
                color="model_name",
                barmode="group",
                title="WER by Speaker Accent",
            )
            st.plotly_chart(fig_accent, use_container_width=True)

# ---------------- TAB 4: ERROR ANALYSIS ----------------
with tabs[3]:
    if not df_all.empty:
        st.subheader("Granular Transcript Error & S/D/I Alignment Inspector")
        st.caption("Inspect substitutions, deletions, and insertions between human reference and model hypothesis.")

        selected_model = st.selectbox("Filter by Model", options=["All"] + list(df_all["model_name"].unique()))
        selected_cond = st.selectbox("Filter by Condition", options=["All"] + list(df_all["condition"].unique()))

        filtered_df = df_all.copy()
        if selected_model != "All":
            filtered_df = filtered_df[filtered_df["model_name"] == selected_model]
        if selected_cond != "All":
            filtered_df = filtered_df[filtered_df["condition"] == selected_cond]

        # Table showing errors
        show_cols = ["model_name", "condition", "wer", "cer", "substitutions", "deletions", "insertions", "raw_reference", "raw_prediction"]
        st.dataframe(filtered_df[[c for c in show_cols if c in filtered_df.columns]], use_container_width=True)

        # Detailed item inspector
        if not filtered_df.empty:
            st.markdown("### Inspect Individual Sample")
            sample_idx = st.selectbox("Select Sample ID to Inspect", options=filtered_df.index)
            sample_row = filtered_df.loc[sample_idx]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Reference Transcript:**")
                st.info(sample_row.get("raw_reference", ""))
            with c2:
                st.markdown(f"**Predicted Transcript ({sample_row.get('model_name')})**: ")
                st.success(sample_row.get("raw_prediction", ""))

            audio_file = sample_row.get("audio_path")
            if audio_file:
                resolved_path = audio_file if os.path.isabs(audio_file) else os.path.join(ROOT_DIR, audio_file)
                if os.path.exists(resolved_path):
                    st.audio(resolved_path)

            # Metrics summary for sample
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Sample WER", f"{sample_row.get('wer', 0.0)*100:.1f}%")
            with m_col2:
                st.metric("Sample CER", f"{sample_row.get('cer', 0.0)*100:.1f}%")
            with m_col3:
                st.metric("S / D / I", f"{sample_row.get('substitutions', 0)} / {sample_row.get('deletions', 0)} / {sample_row.get('insertions', 0)}")
            with m_col4:
                st.metric("Latency / RTF", f"{sample_row.get('latency_seconds', 0.0):.2f}s", delta=f"RTF {sample_row.get('rtf', 0.0):.2f}")

            # Alignments visualization if available
            alignments_raw = sample_row.get("alignments_json") or sample_row.get("alignments")
            if alignments_raw:
                try:
                    alignments = json.loads(alignments_raw) if isinstance(alignments_raw, str) else alignments_raw
                    if isinstance(alignments, list) and alignments:
                        st.markdown("##### 🔍 Alignment Diff")
                        sample_chips = []
                        for align in alignments:
                            atype = align.get("type")
                            ref_w = align.get("reference", "")
                            hyp_w = align.get("hypothesis", "")
                            if atype == "equal":
                                sample_chips.append(f"<span style='color:#e5e7eb; padding:3px 6px;'>{ref_w}</span>")
                            elif atype == "substitute":
                                sample_chips.append(f"<span style='background:#f59e0b33; color:#f59e0b; border:1px solid #f59e0b; padding:3px 8px; border-radius:5px; font-weight:600;'>{ref_w} ➔ {hyp_w}</span>")
                            elif atype == "delete":
                                sample_chips.append(f"<span style='background:#ef444433; color:#ef4444; border:1px solid #ef4444; padding:3px 8px; border-radius:5px; text-decoration:line-through;'>{ref_w}</span>")
                            elif atype == "insert":
                                sample_chips.append(f"<span style='background:#10b98133; color:#10b981; border:1px solid #10b981; padding:3px 8px; border-radius:5px; font-weight:600;'>+{hyp_w}</span>")

                        st.markdown(f"""
                        <div style="background:#131720; border-radius:8px; padding:12px; border:1px solid #2e3440; line-height:2.0; font-size:15px; margin-top:8px;">
                            {' '.join(sample_chips)}
                        </div>
                        """, unsafe_allow_html=True)
                except Exception:
                    pass

# ---------------- TAB 5: LIVE PLAYGROUND ----------------
with tabs[4]:
    st.subheader("🧪 Live ASR, Voice Recording & Noise Stress-Test Playground")
    st.caption("Record your voice live via your microphone, upload an audio file, or test preset benchmark samples. Simulate real-world background noise and evaluate transcription accuracy in real time.")

    # 1. Input source selection
    input_mode = st.radio(
        "Choose Audio Source",
        ["🎙️ Record Voice with Microphone", "⚡ Load Preset Benchmark Sample", "📁 Upload Audio File"],
        horizontal=True,
    )

    audio_bytes = None
    default_ref = ""
    source_name = "live_input.wav"

    col_in, col_ctrl = st.columns([1.6, 1.2])

    with col_in:
        if input_mode == "🎙️ Record Voice with Microphone":
            st.markdown("##### 🎙️ Speak into your microphone")
            mic_audio = st.audio_input("Click the microphone button to record your voice")
            if mic_audio is not None:
                audio_bytes = mic_audio.read()
                source_name = "microphone_recording.wav"
                st.caption("Voice recording captured successfully!")

        elif input_mode == "⚡ Load Preset Benchmark Sample":
            st.markdown("##### ⚡ Select a Verified Test Audio Sample")
            preset_options = {
                "clean_001.wav (Clean Baseline)": {
                    "path": os.path.join(ROOT_DIR, "datasets", "audio", "clean", "clean_001.wav"),
                    "ref": "the security officer is at the main entrance",
                    "domain": "Clean Baseline",
                },
                "sec_001.wav (Security Dispatch)": {
                    "path": os.path.join(ROOT_DIR, "datasets", "audio", "domain_security", "sec_001.wav"),
                    "ref": "the alarm was triggered at gate 4 near zone b",
                    "domain": "Security Control Room",
                },
                "sec_002.wav (Incident Code)": {
                    "path": os.path.join(ROOT_DIR, "datasets", "audio", "domain_security", "sec_002.wav"),
                    "ref": "code red reported at building 3 level 2 immediate backup requested",
                    "domain": "Incident Dispatch",
                },
                "sec_003.wav (Patrol Callsign)": {
                    "path": os.path.join(ROOT_DIR, "datasets", "audio", "domain_security", "sec_003.wav"),
                    "ref": "patrol unit 12 please respond to loading dock camera 5",
                    "domain": "Patrol Callsign",
                },
                "sec_006.wav (Emergency Alert)": {
                    "path": os.path.join(ROOT_DIR, "datasets", "audio", "domain_security", "sec_006.wav"),
                    "ref": "fire alarm notification from server room bay 4 at 10:15 pm",
                    "domain": "Emergency Alert",
                },
                "sec_008.wav (Address Dispatch)": {
                    "path": os.path.join(ROOT_DIR, "datasets", "audio", "domain_security", "sec_008.wav"),
                    "ref": "incident reported at 742 evergreen terrace security dispatched",
                    "domain": "Address & Dispatch",
                },
            }
            selected_sample = st.selectbox("Pick Benchmark Sample", list(preset_options.keys()))
            chosen = preset_options[selected_sample]
            source_name = os.path.basename(chosen["path"])
            default_ref = chosen["ref"]
            if os.path.exists(chosen["path"]):
                with open(chosen["path"], "rb") as f:
                    audio_bytes = f.read()

        elif input_mode == "📁 Upload Audio File":
            st.markdown("##### 📁 Drag & Drop an Audio File")
            uploaded_file = st.file_uploader("Upload Audio (WAV, MP3, FLAC)", type=["wav", "mp3", "flac"])
            if uploaded_file is not None:
                audio_bytes = uploaded_file.read()
                source_name = uploaded_file.name

    with col_ctrl:
        st.markdown("##### ⚙️ Evaluation Settings")
        playground_model = st.selectbox("Select Whisper Model", ["tiny", "base"], index=0)
        inject_noise = st.checkbox("🔊 Inject Background Noise", value=False)
        target_snr = st.slider(
            "Target SNR (dB) - Lower is Noisier",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=2.0,
            help="20dB = subtle background, 10dB = typical noisy control room, 5dB = heavy interference",
        ) if inject_noise else None

        ground_truth = st.text_area(
            "Reference Transcript (for WER / CER calculation)",
            value=default_ref,
            placeholder="Type spoken words here to calculate WER & Entity Accuracy...",
            height=85,
        )
        save_to_db = st.checkbox("💾 Save test to Benchmark Database & History", value=True)

    st.markdown("---")

    # If audio is available, process and display
    if audio_bytes is not None:
        temp_input_path = os.path.join(ROOT_DIR, "results", f"playground_input_{source_name}")
        with open(temp_input_path, "wb") as f:
            f.write(audio_bytes)

        try:
            audio_arr, sr = audio_proc.load_audio(temp_input_path)
            meta = audio_proc.get_audio_metadata(temp_input_path)
            duration = meta["duration_seconds"]

            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown("##### 🎧 Clean Audio Input")
                st.audio(temp_input_path)

            audio_for_model = audio_arr
            if inject_noise and target_snr is not None:
                noisy_arr = noise_aug.add_noise(audio_arr, snr_db=target_snr, noise_type="pink")
                audio_for_model = noisy_arr
                temp_noisy_path = os.path.join(ROOT_DIR, "results", "playground_noisy_preview.wav")
                audio_proc.save_audio(noisy_arr, temp_noisy_path, sr=sr)
                with p_col2:
                    st.markdown(f"##### 🔊 Noisy Audio Preview ({target_snr} dB SNR)")
                    st.audio(temp_noisy_path)

            if st.button("🚀 Run Live Transcription & Evaluation", type="primary", use_container_width=True):
                with st.spinner(f"Running Whisper '{playground_model}' on GPU..."):
                    result = asr_engine.transcribe(audio_for_model, model_name=playground_model)

                st.markdown("### 📝 Transcription Result")
                st.markdown(f"""
                <div style="background:#1e222d; border-radius:8px; padding:16px; border:1px solid #3b82f6; font-size:18px;">
                    {result['text']}
                </div>
                """, unsafe_allow_html=True)

                rtf = result["latency_seconds"] / max(duration, 1e-4)

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Latency", f"{result['latency_seconds']:.3f} s")
                with m2:
                    st.metric("Real-Time Factor (RTF)", f"{rtf:.3f}", delta=f"{'Faster than real-time' if rtf < 1.0 else 'Slower'}")
                with m3:
                    st.metric("Detected Language", result["detected_language"])
                with m4:
                    st.metric("Audio Duration", f"{duration:.2f} s")

                # If reference text is provided, calculate WER and show visual alignment
                eval_res = {}
                if ground_truth.strip():
                    eval_res = eval_service.evaluate_pair(
                        reference=ground_truth,
                        hypothesis=result["text"],
                        audio_duration=duration,
                        latency_seconds=result["latency_seconds"],
                    )

                    st.markdown("### 📊 Accuracy & Error Breakdown")
                    e1, e2, e3, e4 = st.columns(4)
                    with e1:
                        st.metric("Word Error Rate (WER)", f"{eval_res['wer']*100:.2f}%")
                    with e2:
                        st.metric("Character Error Rate (CER)", f"{eval_res['cer']*100:.2f}%")
                    with e3:
                        st.metric(
                            "Substitutions / Deletions / Insertions",
                            f"{eval_res['substitutions']} / {eval_res['deletions']} / {eval_res['insertions']}",
                        )
                    with e4:
                        eer = eval_res["entity_evaluation"].get("entity_error_rate", 0.0)
                        st.metric("Entity Error Rate (EER)", f"{eer*100:.2f}%")

                    # Visual token-by-token alignment diff
                    st.markdown("#### 🔍 Token-by-Token Visual Diff")
                    diff_chips = []
                    for align in eval_res.get("alignments", []):
                        atype = align["type"]
                        ref_w = align["reference"]
                        hyp_w = align["hypothesis"]
                        if atype == "equal":
                            diff_chips.append(f"<span style='color:#e5e7eb; padding:3px 6px;'>{ref_w}</span>")
                        elif atype == "substitute":
                            diff_chips.append(
                                f"<span style='background:#f59e0b33; color:#f59e0b; border:1px solid #f59e0b; padding:3px 8px; border-radius:5px; font-weight:600;' title='Substitution'>{ref_w} ➔ {hyp_w}</span>"
                            )
                        elif atype == "delete":
                            diff_chips.append(
                                f"<span style='background:#ef444433; color:#ef4444; border:1px solid #ef4444; padding:3px 8px; border-radius:5px; text-decoration:line-through;' title='Deletion'>{ref_w}</span>"
                            )
                        elif atype == "insert":
                            diff_chips.append(
                                f"<span style='background:#10b98133; color:#10b981; border:1px solid #10b981; padding:3px 8px; border-radius:5px; font-weight:600;' title='Insertion'>+{hyp_w}</span>"
                            )

                    st.markdown(
                        f"""
                        <div style="background:#131720; border-radius:8px; padding:16px; border:1px solid #2e3440; line-height:2.2; font-size:16px;">
                            {' '.join(diff_chips)}
                        </div>
                        <div style="margin-top:8px; font-size:12px; color:#9aa0a6;">
                            Legend: <span style="color:#e5e7eb;">Match</span> | <span style="color:#f59e0b;">Substitution (Ref ➔ Hyp)</span> | <span style="color:#ef4444; text-decoration:line-through;">Deletion</span> | <span style="color:#10b981;">+Insertion</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Security entity detection breakdown
                    ent_data = eval_res.get("entity_evaluation", {})
                    if ent_data.get("total_entities", 0) > 0:
                        st.markdown("#### 🛡️ Domain Security Entity Analysis")
                        st.caption("Checks mission-critical security keywords (Gate IDs, incident codes, unit callsigns).")
                        e_cols = st.columns(min(4, max(1, ent_data["total_entities"])))
                        missed_list = ent_data.get("missed_entities", [])
                        for i, ent_item in enumerate(TextNormalizer.extract_security_entities(ground_truth)):
                            col_target = e_cols[i % len(e_cols)]
                            is_matched = ent_item["matched_text"] not in missed_list
                            with col_target:
                                badge = "✅ Captured" if is_matched else "❌ Missed"
                                color = "#10b981" if is_matched else "#ef4444"
                                st.markdown(f"""
                                <div style="background:#1e222d; border-left:4px solid {color}; padding:8px 12px; border-radius:4px; margin-bottom:6px;">
                                    <strong>{ent_item['matched_text'].upper()}</strong><br/>
                                    <span style="font-size:11px; color:#9aa0a6;">{ent_item['entity_type']}</span> — <span style="color:{color}; font-size:12px; font-weight:bold;">{badge}</span>
                                </div>
                                """, unsafe_allow_html=True)

                # Save to SQLite database if requested
                if save_to_db:
                    cond_tag = f"noisy_{int(target_snr)}dB" if (inject_noise and target_snr is not None) else "clean"
                    rec = {
                        "model_name": playground_model,
                        "audio_path": source_name,
                        "condition": cond_tag,
                        "snr_db": target_snr if inject_noise else None,
                        "accent": "user_microphone" if input_mode == "🎙️ Record Voice with Microphone" else "standard",
                        "domain_category": "live_playground",
                        "duration_seconds": duration,
                        "latency_seconds": result["latency_seconds"],
                        "rtf": rtf,
                        "raw_reference": ground_truth,
                        "raw_prediction": result["text"],
                        "normalized_reference": eval_res.get("normalized_reference", ""),
                        "normalized_prediction": eval_res.get("normalized_hypothesis", ""),
                        "wer": eval_res.get("wer", 0.0),
                        "cer": eval_res.get("cer", 0.0),
                        "substitutions": eval_res.get("substitutions", 0),
                        "deletions": eval_res.get("deletions", 0),
                        "insertions": eval_res.get("insertions", 0),
                        "hits": eval_res.get("hits", 0),
                        "reference_words": eval_res.get("reference_words", 0),
                        "entity_error_rate": eval_res.get("entity_evaluation", {}).get("entity_error_rate", 0.0),
                        "alignments": eval_res.get("alignments", []),
                    }
                    repo.insert_record(run_id=None, record=rec)
                    st.success("✅ Test saved to Benchmark Database! Visit the Overview and Error Inspector tabs to see this record.")

        finally:
            if os.path.exists(temp_input_path):
                try:
                    os.remove(temp_input_path)
                except Exception:
                    pass
    else:
        st.info("👆 Select an audio source above: record your voice with the microphone, select a preset sample, or upload a file.")

