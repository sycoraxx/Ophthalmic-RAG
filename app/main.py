"""
app.py — Streamlit Frontend for the Multimodal Self-Correcting Ophthalmology RAG
UPDATED: Voice input (ASR) + session-aware conversation handling
─────────────────────────────────────────────────────────────────────────────────

Launch:
    streamlit run app/main.py -- --gpus 0,1

Supports:
  - Text-only queries (same as before)
  - Voice queries (🎙️ record → transcribe → editable preview → send)
  - Multimodal queries (text + eye image → EyeCLIP analysis → augmented RAG)
  - Multi-turn conversations with clinical context persistence

The QueryEngine is loaded ONCE and cached in the Streamlit session.
"""

import sys
import os
import argparse
import json
import re

# ─── GPU Selection ────────────────────────────────────────────────────────────
# Must happen before QueryEngine is imported
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU IDs (e.g., '0' or '2,3')")
args, _ = parser.parse_known_args()

if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print(f"[Main] Setting CUDA_VISIBLE_DEVICES={args.gpus}")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import time
import uuid
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────
ROLLING_HISTORY_SIZE = 3  # queries-only, so we can afford more turns
SESSION_DIR = Path("./data/sessions")  # Match engine.py

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="👁️ LVPEI DocBot",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }

    /* Title styling */
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 1rem !important;
    }

    /* Pipeline status badges */
    .pipeline-step {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 2px 4px 2px 0;
    }
    .step-refine   { background: #1e1b4b; color: #a5b4fc; }
    .step-retrieve { background: #1a2e05; color: #86efac; }
    .step-rerank   { background: #422006; color: #fcd34d; }
    .step-generate { background: #3b0764; color: #d8b4fe; }
    .step-verify   { background: #0c4a6e; color: #7dd3fc; }
    .step-correct  { background: #7f1d1d; color: #fca5a5; }
    .step-pass     { background: #064e3b; color: #6ee7b7; }
    .step-eyeclip  { background: #0f172a; color: #38bdf8; }
    .step-asr      { background: #3b1f2b; color: #f9a8d4; }

    /* Source citation cards */
    .src-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
        position: relative;
        cursor: help;
    }
    .src-card .src-title {
        font-weight: 600;
        color: #a5b4fc;
    }
    .src-card .src-path {
        color: #9ca3af;
        font-size: 0.8rem;
    }

    /* Tooltip */
    .src-card .src-tooltip {
        visibility: hidden;
        min-width: 300px;
        max-width: 500px;
        background-color: #1f2937;
        color: #e5e7eb;
        text-align: left;
        border-radius: 8px;
        padding: 14px;
        position: absolute;
        z-index: 1000;
        bottom: 115%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        line-height: 1.4;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        border: 1px solid #374151;
        pointer-events: none;
        max-height: 400px;
        overflow-y: hidden;
    }
    .src-card .src-tooltip::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -8px;
        border-width: 8px;
        border-style: solid;
        border-color: #374151 transparent transparent transparent;
    }
    .src-card:hover .src-tooltip {
        visibility: visible;
        opacity: 1;
    }

    /* Loading animation */
    .loading-dots::after {
        content: '';
        animation: dots 1.5s steps(3, end) infinite;
    }
    @keyframes dots {
        0%   { content: '.'; }
        33%  { content: '..'; }
        66%  { content: '...'; }
    }

    /* ─── Voice Input Styling ──────────────────────────────────────────── */
    .voice-container {
        background: linear-gradient(135deg, rgba(59,31,43,0.3) 0%, rgba(30,27,75,0.3) 100%);
        border: 1px solid rgba(249,168,212,0.2);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .voice-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #f9a8d4;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .voice-label .pulse {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #f9a8d4;
        animation: pulse-glow 2s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(249,168,212,0.4); }
        50%       { box-shadow: 0 0 0 8px rgba(249,168,212,0); }
    }
    .transcription-preview {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 8px;
        font-size: 0.9rem;
        color: #e5e7eb;
    }
    .transcription-meta {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-top: 4px;
    }

    /* Hide default Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── Load Engine (Global Cached Singleton) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_engine_cached(patient_memory_backend: str | None):
    """Load QueryEngine strictly once per server instance."""
    from src.engine import QueryEngine
    print("[Main] 🚀 Starting global QueryEngine warm start...")
    engine = QueryEngine(patient_memory_backend=patient_memory_backend)
    print("[Main] ✓ Global QueryEngine warm start complete.")
    return engine


def get_engine(patient_memory_backend: str | None = None):
    """Proxy to get engine, with a UI spinner if it's currently loading."""
    # Ensure consistent default with engine.py and sidebar radio to avoid cache misses
    requested_backend = (patient_memory_backend or "mempalace").strip().lower()
    current_backend = st.session_state.get("engine_patient_memory_backend")
    if "engine" in st.session_state and current_backend == requested_backend:
        return st.session_state.engine

    if "engine" in st.session_state:
        st.session_state.pop("engine", None)
        st.session_state.pop("engine_patient_memory_backend", None)

    with st.spinner("Loading RAG engine (models, indexes)... This takes ~60s on first load."):
        engine = _load_engine_cached(requested_backend)
        st.session_state.engine = engine
        st.session_state.engine_patient_memory_backend = getattr(engine, "patient_memory_backend", requested_backend)
        return engine


def _render_split_response(content: str):
    """Render a response that may be split by ---DETAILS--- marker."""
    # Search for marker case-insensitively
    marker_pattern = re.compile(r"---DETAILS---", re.IGNORECASE)
    match = marker_pattern.search(content)
    
    if match:
        start, end = match.span()
        summary = content[:start].strip()
        details = content[end:].strip()

        # Clean up common model prefixes (Summary, Response, Section 1, etc.)
        summary = re.sub(r"^(Response|Summary|Clinical Summary|summary|Result|Assistant|Section \d+):\s*", "", summary, flags=re.I).strip()
        # Handle if "summary" or "Section 1" is just a loose phrase at the start
        summary = re.sub(r"^(summary|Section \d+)\s+", "", summary, flags=re.I).strip()
        
        # Strip accidental brackets [...]
        summary = summary.strip("[]").strip()
        details = details.strip("[]").strip()
        
        # Clean up details prefixes too
        details = re.sub(r"^(Detailed Explanation|Details|details|Section \d+):\s*", "", details, flags=re.I).strip()
        details = re.sub(r"^Section \d+\s*", "", details, flags=re.I).strip()

        st.markdown(f"**Summary:** {summary}")
        with st.expander("🔍 Show clinical details & advice", expanded=False):
            st.markdown(details)
    else:
        st.markdown(content)


def _render_physician_dashboard(engine, session_id, patient_id):
    """Render a high-level overview for clinicians."""
    st.markdown("### 🏥 Physician Dashboard")
    
    if not patient_id:
        st.warning("No Patient ID provided. Longitudinal memory is unavailable.")
        return

    try:
        session_info = engine.get_session_info(session_id, patient_id=patient_id)
        if not session_info:
            st.info("No session data available.")
            return

        summary = session_info.get("clinician_summary")
        if not summary:
            st.info("No clinician summary generated yet.")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Loci", len(summary.get("recent_memory_loci", [])))
        with col2:
            st.metric("Problem Count", len(summary.get("active_problem_list", [])))

        if summary.get("active_problem_list"):
            st.markdown("**Active Problem List:**")
            for p in summary["active_problem_list"]:
                st.markdown(f"- {p}")
        
        if summary.get("current_symptoms"):
            st.markdown("**Reported Symptoms:**")
            st.markdown(", ".join(summary["current_symptoms"]))

        if summary.get("recent_memory_loci"):
            with st.expander("📜 Recent Longitudinal Records", expanded=False):
                for loci in summary["recent_memory_loci"][:5]:
                    date = loci.get("created_at", "Unknown Date")[:10]
                    st.markdown(f"**{date}** | {loci.get('entity_type')}: {loci.get('value')} ({loci.get('room')})")

        st.divider()
        st.download_button(
            "📄 Export Clinical Summary (JSON)",
            data=json.dumps(summary, indent=2),
            file_name=f"clinician_summary_{patient_id}_{session_id[:8]}.json",
            mime="application/json",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error loading physician dashboard: {e}")


def _render_clinician_summary_export(pipeline_data: dict):
    export_path = pipeline_data.get("clinician_summary_path")
    if not export_path:
        return

    path = Path(export_path)
    st.markdown("**Clinician Summary Export:**")
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            st.caption(f"Conversation date: {payload.get('conversation_date', '—')}")
            st.caption(f"Generated at: {payload.get('generated_at', '—')}")
            if payload.get("active_problem_list"):
                st.markdown(f"**Problem List:** {', '.join(payload['active_problem_list'])}")
            if payload.get("current_symptoms"):
                st.markdown(f"**Symptoms:** {', '.join(payload['current_symptoms'])}")
            if payload.get("current_findings"):
                st.markdown(f"**Findings:** {', '.join(payload['current_findings'])}")
            st.download_button(
                "Download clinician summary JSON",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/json",
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"Clinician summary unavailable: {e}")
    else:
        st.caption(f"Summary file not found: {path}")


def _save_uploaded_image(uploaded_file) -> str | None:
    """Save uploaded file to /tmp and return the path, or None."""
    if uploaded_file is None:
        return None
    image_path = f"/tmp/eyeclip_{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return image_path



# ─── Session Management Helpers ──────────────────────────────────────────────
def _get_or_init_session_id() -> str:
    """Get or create session_id for this conversation."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def _reset_conversation():
    """Clear chat history AND start a new clinical session."""
    st.session_state.messages = []
    st.session_state.rolling_history = []
    st.session_state.session_id = str(uuid.uuid4())  # New clinical context
    # Reset voice state
    st.session_state.pop("voice_transcription", None)
    st.session_state.pop("voice_asr_meta", None)
    st.rerun()


def _get_recent_queries_for_generation() -> list[str]:
    """Extract user queries from message history for generation context."""
    queries = []
    for msg in st.session_state.messages:
        if msg["role"] == "user" and msg.get("content"):
            queries.append(msg["content"])
    return queries[-ROLLING_HISTORY_SIZE:]  # Last N queries


# ─── Header ──────────────────────────────────────────────────────────────────
st.title("👁️ LVPEI DocBot")
st.markdown(
    '<p class="subtitle">'
    'Multimodal Self-Correcting RAG · EyeCLIP · MedGemma · Hybrid Retrieval'
    '</p>',
    unsafe_allow_html=True,
)

# ─── Sidebar (UPDATED with session controls) ─────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    num_sources = st.slider("Number of sources to retrieve", 1, 5, 3)
    show_pipeline = st.toggle("Show pipeline details", value=True)
    
    # Session state toggle
    enable_session_tracking = st.toggle(
        "Enable conversation memory", 
        value=True,
        help="When enabled, the bot remembers clinical context across turns (e.g., retina findings from prior OCT analysis)."
    )
    
    # Fast mode toggle
    fast_mode = st.toggle(
        "⚡ Fast Mode (Skip Verification)",
        value=True,
        help="Skips the grounding verification and self-correction steps. Results in much faster responses, but slightly higher risk of ungrounded claims."
    )

    st.divider()
    st.markdown("### 🧑 Patient Profile")
    st.caption("Fill in what you're comfortable sharing — helps personalize advice.")
    patient_name = st.text_input("Name", key="pf_name", placeholder="e.g. Rahul")
    col1, col2 = st.columns(2)
    with col1:
        patient_age = st.text_input("Age", key="pf_age", placeholder="e.g. 45")
    with col2:
        patient_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"], key="pf_gender")
    patient_conditions = st.text_input(
        "Known conditions", key="pf_conditions",
        placeholder="e.g. diabetes, glaucoma",
    )
    patient_location = st.text_input(
        "Location", key="pf_location",
        placeholder="e.g. Delhi, India",
    )
    patient_record_id = st.text_input(
        "Patient ID (for longitudinal memory)",
        key="pf_patient_id",
        placeholder="e.g. LVPEI-000123",
        help="If provided, the assistant can retrieve and update patient-specific memory across sessions.",
    )
    patient_memory_backend = st.radio(
        "Patient memory backend",
        ["mempalace", "sqlite"],
        index=0 if st.session_state.get("engine_patient_memory_backend", "mempalace") == "mempalace" else 1,
        horizontal=True,
        help="Switch between the MemPalace palace backend and the SQLite fallback store.",
        key="pf_patient_memory_backend",
    )

    st.divider()
    
    # Session info display (debugging)
    if enable_session_tracking:
        session_id = _get_or_init_session_id()
        with st.expander("🔐 Session Info", expanded=False):
            st.code(f"Session: {session_id[:8]}...", language="text")



            try:
                engine = get_engine(patient_memory_backend)
                # Use the new physician dashboard helper
                _render_physician_dashboard(engine, session_id, patient_record_id)
            except Exception as e:
                st.caption(f"Physician dashboard unavailable: {e}")
    
    st.divider()

    # ASR model info
    if "engine" in st.session_state:
        engine = st.session_state.engine
        if engine.asr_ready:
            info = engine.speech_recognizer.model_info
            with st.expander("🎙️ ASR Info", expanded=False):
                st.markdown(f"**Model:** `{info['model_size']}`")
                st.markdown(f"**Device:** `{info['device']}` ({info['compute_type']})")
                st.markdown(f"**Status:** ✅ Ready")
    
    st.divider()
    
    # Clear button resets session too
    if st.button("🗑️ Clear Chat & Reset Context"):
        _reset_conversation()

    st.divider()
    st.markdown("### 📖 About")
    st.markdown(
        "This system answers eye health questions using content from "
        "**Kanski's Clinical Ophthalmology** and **Khurana's "
        "Comprehensive Ophthalmology**.\n\n"
        "Pipeline: **🎙️ Voice/Text Input** → **EyeCLIP Image Analysis** → "
        "Query Refinement → Hybrid Retrieval → Cross-Encoder Re-Ranking → "
        "Answer Generation → Grounding Verification → Self-Correction"
    )


def _get_patient_profile() -> dict | None:
    """Collect non-empty profile fields into a dict."""
    profile = {
        "name": patient_name,
        "age": patient_age,
        "gender": patient_gender,
        "known_conditions": patient_conditions,
        "location": patient_location,
    }
    if not any(profile.values()):
        return None
    return profile


def _get_patient_id() -> str | None:
    pid = (patient_record_id or "").strip()
    return pid if pid else None


def _render_pipeline(pipeline_data: dict):
    """Render pipeline debug info inside an expander."""
    import html as html_mod

    cols = st.columns(4)
    with cols[0]:
        st.metric("Sources", pipeline_data.get("num_sources", "—"))
    with cols[1]:
        st.metric("Grounding", pipeline_data.get("verdict", "—"))
    with cols[2]:
        st.metric("Retries", pipeline_data.get("retries", 0))
    with cols[3]:
        st.metric("Time", f"{pipeline_data.get('time', 0):.1f}s")

    # Show ASR info if query came from voice
    if pipeline_data.get("asr_time"):
        st.markdown(
            f'<span class="pipeline-step step-asr">🎙️ ASR: {pipeline_data["asr_time"]:.2f}s</span> '
            f'<span class="pipeline-step" style="background:rgba(255,255,255,0.05);color:#9ca3af;">'
            f'RTF={pipeline_data.get("asr_rtf", 0):.2f}</span>',
            unsafe_allow_html=True,
        )

    if pipeline_data.get("visual_findings"):
        st.info(f"**👁️ EyeCLIP Findings:** {pipeline_data['visual_findings']}")

    if pipeline_data.get("refined_query"):
        st.markdown(f"**Refined Query:** {pipeline_data['refined_query']}")

    if pipeline_data.get("sources"):
        st.markdown("**Retrieved Sources:**")
        for src in pipeline_data["sources"]:
            safe_text = html_mod.escape(src.get("content", ""))
            if len(safe_text) > 1500:
                safe_text = safe_text[:1500] + "..."
            safe_text = safe_text.replace('\n', '<br>')

            st.markdown(
                f'<div class="src-card">'
                f'<span class="src-title">{src["source"]}</span> · '
                f'<span class="src-path">{src["section_path"]}</span>'
                f'<div class="src-tooltip">{safe_text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if pipeline_data.get("flagged_claims"):
        st.warning("Flagged claims (corrected):")
        for claim in pipeline_data["flagged_claims"]:
            st.markdown(f"- {claim}")

    _render_clinician_summary_export(pipeline_data)


# ─── Chat State ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rolling_history" not in st.session_state:
    st.session_state.rolling_history = []
if "voice_key" not in st.session_state:
    st.session_state.voice_key = 0
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "last_transcribed_audio_hash" not in st.session_state:
    st.session_state.last_transcribed_audio_hash = None

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "assistant" else "🙋"):
        if msg["role"] == "user" and msg.get("image_path"):
            if os.path.exists(msg["image_path"]):
                st.image(msg["image_path"], caption="📷 Uploaded Image", width=280)
        # Show voice badge if message came from voice
        if msg["role"] == "user" and msg.get("from_voice"):
            st.markdown(
                '<span class="pipeline-step step-asr">🎙️ Voice Input</span>',
                unsafe_allow_html=True,
            )
            st.markdown(msg["content"], unsafe_allow_html=True)
        else:
            _render_split_response(msg["content"])
        
        if msg["role"] == "assistant" and "pipeline" in msg and show_pipeline:
            with st.expander("🔍 Pipeline Details", expanded=False):
                _render_pipeline(msg["pipeline"])


# ─── Voice Input Section ──────────────────────────────────────────────────────
st.markdown(
    '<div class="voice-label"><span class="pulse"></span> Voice Input — speak your query</div>',
    unsafe_allow_html=True,
)

voice_col1, voice_col2 = st.columns([3, 1])

with voice_col1:
    audio_value = st.audio_input(
        "🎙️ Record your eye health question",
        key=f"voice_recorder_{st.session_state.voice_key}",
        label_visibility="collapsed",
    )

with voice_col2:
    asr_available = "engine" in st.session_state and st.session_state.engine.asr_ready
    if asr_available:
        st.markdown(
            '<div style="padding-top:10px;"><span class="pipeline-step step-pass" '
            'style="font-size:0.72rem;">✅ ASR Ready</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="padding-top:10px;"><span class="pipeline-step" '
            'style="background:#374151;color:#9ca3af;font-size:0.72rem;">⏳ ASR Loading</span></div>',
            unsafe_allow_html=True,
        )

# Handle voice recording → transcription
if audio_value is not None:
    engine = get_engine()

    if not engine.asr_ready:
        st.warning("🎙️ Speech recognition model is still loading. Please try again in a moment.")
    else:
        import hashlib
        audio_bytes = audio_value.read()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        audio_value.seek(0)

        # Only transcribe if this is a NEW recording that hasn't been processed yet
        if audio_bytes and len(audio_bytes) > 100 and audio_hash != st.session_state.get("last_transcribed_audio_hash"):
            # Transcribe
            with st.spinner("🎙️ Transcribing your voice..."):
                result = engine.transcribe_audio(audio_bytes)

            if result and result.text:
                st.session_state["voice_transcription"] = result.text
                st.session_state["voice_asr_meta"] = {
                    "asr_time": result.processing_time_seconds,
                    "asr_rtf": result.real_time_factor,
                    "asr_duration": result.duration_seconds,
                    "asr_lang": result.language,
                    "asr_lang_prob": result.language_probability,
                }
                st.session_state["last_transcribed_audio_hash"] = audio_hash
            else:
                st.warning("Could not transcribe audio. Please try again or type your query below.")

# Show transcription preview and send button
if "voice_transcription" in st.session_state and st.session_state["voice_transcription"]:
    transcription = st.session_state["voice_transcription"]
    asr_meta = st.session_state.get("voice_asr_meta", {})

    st.markdown(
        f'<div class="transcription-preview">'
        f'<strong>📝 Transcribed:</strong> {transcription}'
        f'<div class="transcription-meta">'
        f'⏱ {asr_meta.get("asr_time", 0):.2f}s · '
        f'RTF {asr_meta.get("asr_rtf", 0):.2f} · '
        f'Lang: {asr_meta.get("asr_lang", "?")} '
        f'({asr_meta.get("asr_lang_prob", 0):.0%})'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Editable text area for corrections
    edited_text = st.text_area(
        "✏️ Edit transcription if needed:",
        value=transcription,
        height=80,
        key="voice_edit_area",
        label_visibility="visible",
    )

    send_col1, send_col2 = st.columns([1, 1])
    with send_col1:
        send_voice = st.button("✅ Send as Query", key="send_voice_btn", type="primary", use_container_width=True)
    with send_col2:
        discard_voice = st.button("❌ Discard", key="discard_voice_btn", use_container_width=True)

    if discard_voice:
        st.session_state.pop("voice_transcription", None)
        st.session_state.pop("voice_asr_meta", None)
        st.rerun()

    if send_voice and edited_text and edited_text.strip():
        # Store the query to process below (same path as typed input)
        st.session_state["_pending_voice_query"] = edited_text.strip()
        st.session_state["_pending_voice_meta"] = asr_meta
        st.session_state.pop("voice_transcription", None)
        st.session_state.pop("voice_asr_meta", None)
        st.session_state.voice_key += 1  # Reset widget
        st.rerun()

st.divider()

# ─── Chat Input + Inline Image Upload ─────────────────────────────────────────
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

uploaded_file = st.file_uploader(
    "📷 Attach an eye image (optional)",
    type=["png", "jpg", "jpeg"],
    key=f"eye_img_{st.session_state.upload_key}",
    label_visibility="visible",
)

# Determine the query source: voice (pending) or typed
pending_voice_query = st.session_state.pop("_pending_voice_query", None)
pending_voice_meta = st.session_state.pop("_pending_voice_meta", None)

prompt = None
from_voice = False
if pending_voice_query:
    prompt = pending_voice_query
    from_voice = True
else:
    prompt = st.chat_input("Ask an eye health question...")
    if prompt:
        # User chose to type, clear any pending voice stuff
        st.session_state.pop("voice_transcription", None)
        st.session_state.pop("voice_asr_meta", None)

# Eager load the engine into the global cache
# This ensures that when the background dummy-ping hits the app, this gets run.
# Because it relies on @st.cache_resource, it only ever loads once.
get_engine()

if prompt:
    # Save image if uploaded
    image_path = _save_uploaded_image(uploaded_file)

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "image_path": image_path,
        "from_voice": from_voice,
    })
    with st.chat_message("user", avatar="🙋"):
        if image_path:
            st.image(image_path, caption="📷 Uploaded Image", width=280)
        if from_voice:
            st.markdown(
                '<span class="pipeline-step step-asr">🎙️ Voice Input</span>',
                unsafe_allow_html=True,
            )
        st.markdown(prompt)

    # Load engine
    engine = get_engine(patient_memory_backend)
    
    # Get session_id if tracking enabled
    session_id = _get_or_init_session_id() if enable_session_tracking else None
    
    # Get recent queries for generation context (not retrieval!)
    recent_queries = _get_recent_queries_for_generation()

    # Run pipeline with status updates
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        status_container = st.empty()
        start_time = time.time()
        status_container.markdown(
            '<span class="pipeline-step step-refine">🧠 Running unified engine pipeline...</span>',
            unsafe_allow_html=True,
        )

        result = engine.ask(
            raw_query=prompt,
            image_path=image_path,
            k=num_sources,
            verbose=False,
            session_id=session_id,
            patient_id=_get_patient_id(),
            recent_history=recent_queries if enable_session_tracking else None,
            patient_profile=_get_patient_profile(),
            fast_mode=fast_mode,
            use_session_state=enable_session_tracking,
            return_trace=True,
        )

        if len(result) == 4:
            answer, visual_findings, returned_session_id, pipeline_data = result
            if enable_session_tracking and returned_session_id:
                st.session_state.session_id = returned_session_id
        elif len(result) == 3:
            answer, visual_findings, pipeline_data = result
        else:
            answer, visual_findings = result
            pipeline_data = {}

        if not isinstance(pipeline_data, dict):
            pipeline_data = {}
        if "time" not in pipeline_data:
            pipeline_data["time"] = time.time() - start_time
        if visual_findings and not pipeline_data.get("visual_findings"):
            pipeline_data["visual_findings"] = visual_findings

        # Inject ASR metadata into pipeline data if from voice
        if from_voice and pending_voice_meta:
            pipeline_data.update(pending_voice_meta)

        # Render split response
        _render_split_response(answer)
        

        elapsed = pipeline_data.get("time", time.time() - start_time)

        # Show final status
        asr_badge = ""
        if from_voice:
            asr_time = pipeline_data.get("asr_time", 0)
            asr_badge = (
                f'<span class="pipeline-step step-asr">🎙️ ASR {asr_time:.2f}s</span> '
            )

        if pipeline_data.get("verdict") == "SAFETY TRIAGE bypass":
            status_container.markdown(
                f'{asr_badge}'
                '<span class="pipeline-step step-correct">🚨 Safety Triage Bypass</span> '
                f'<span class="pipeline-step" style="background:rgba(255,255,255,0.05);color:#9ca3af;">'
                f'⏱ {elapsed:.1f}s</span>',
                unsafe_allow_html=True,
            )
        elif pipeline_data.get("verdict") == "PASS":
            status_container.markdown(
                f'{asr_badge}'
                '<span class="pipeline-step step-pass">✅ Grounded & Verified</span> '
                f'<span class="pipeline-step" style="background:rgba(255,255,255,0.05);color:#9ca3af;">'
                f'⏱ {elapsed:.1f}s</span>',
                unsafe_allow_html=True,
            )
        elif pipeline_data.get("verdict") == "FAST MODE (Unverified)":
            status_container.markdown(
                f'{asr_badge}'
                '<span class="pipeline-step step-pass" style="background:#4b5563;color:#d1d5db;">⚡ Fast Mode (Unverified)</span> '
                f'<span class="pipeline-step" style="background:rgba(255,255,255,0.05);color:#9ca3af;">'
                f'⏱ {elapsed:.1f}s</span>',
                unsafe_allow_html=True,
            )
        elif pipeline_data.get("verdict") == "PARTIAL":
            status_container.markdown(
                f'{asr_badge}'
                '<span class="pipeline-step step-rerank">ℹ️ Partially Verified</span> '
                f'<span class="pipeline-step" style="background:rgba(255,255,255,0.05);color:#9ca3af;">'
                f'⏱ {elapsed:.1f}s</span>',
                unsafe_allow_html=True,
            )
        else:
            status_container.markdown(
                f'{asr_badge}'
                '<span class="pipeline-step step-correct">⚠️ Unverified</span> '
                f'<span class="pipeline-step" style="background:rgba(255,255,255,0.05);color:#9ca3af;">'
                f'⏱ {elapsed:.1f}s</span>',
                unsafe_allow_html=True,
            )


        # Show pipeline details
        if show_pipeline:
            with st.expander("🔍 Pipeline Details", expanded=False):
                _render_pipeline(pipeline_data)

        _render_clinician_summary_export(pipeline_data)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "pipeline": pipeline_data,
    })

    # Update rolling history with raw user intent to avoid recursive query-drift.
    history_entry = prompt
    if visual_findings:
        history_entry = f"{prompt} [Image findings: {visual_findings}]"
    st.session_state.rolling_history.append(history_entry)
    st.session_state.rolling_history = st.session_state.rolling_history[-ROLLING_HISTORY_SIZE:]

    # Reset uploaders
    st.session_state.upload_key += 1
    st.session_state.voice_key += 1
    st.rerun()

# ─── Eager Background Warm-Up ────────────────────────────────────────────────
# Streamlit does not evaluate `main.py` until the first HTTP request. To achieve
# a true "background warm load" right after `streamlit run`, we spawn a thread
# that sends a dummy local request exactly once. This triggers Streamlit to load
# the page, hit get_engine(), and fully warm the global @st.cache_resource GPU cache.
import threading
import urllib.request

def _trigger_warmup():
    time.sleep(2) # Give Tornado time to bind port
    try:
        urllib.request.urlopen("http://localhost:8501")
    except:
        pass

if "WARM_START_INIT" not in os.environ:
    os.environ["WARM_START_INIT"] = "1"
    threading.Thread(target=_trigger_warmup, daemon=True).start()