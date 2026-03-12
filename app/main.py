"""
app.py — Streamlit Frontend for the Multimodal Self-Correcting Ophthalmology RAG
UPDATED: Session-aware conversation handling with clinical context persistence
─────────────────────────────────────────────────────────────────────────────────

Launch:
    streamlit run app.py --server.port 8501

Supports:
  - Text-only queries (same as before)
  - Multimodal queries (text + eye image → EyeCLIP analysis → augmented RAG)
  - NEW: Multi-turn conversations with clinical context persistence

The QueryEngine is loaded ONCE and cached in the Streamlit session.
"""

import sys
import os
import argparse

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

from src.triage import check_red_flags

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

    /* Hide default Streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── Load Engine (Session State Singleton) ───────────────────────────────────
def get_engine():
    """Load QueryEngine once and store it in session state."""
    if "engine" not in st.session_state:
        with st.spinner("Loading RAG engine (models, indexes)... This takes ~60s on first load."):
            from src.engine import QueryEngine
            st.session_state.engine = QueryEngine()
    return st.session_state.engine


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
    
    # NEW: Session state toggle
    enable_session_tracking = st.toggle(
        "Enable conversation memory", 
        value=True,
        help="When enabled, the bot remembers clinical context across turns (e.g., retina findings from prior OCT analysis)."
    )
    
    # NEW: Fast mode toggle
    fast_mode = st.toggle(
        "⚡ Fast Mode (Skip Verification)",
        value=False,
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

    st.divider()
    
    # NEW: Session info display (debugging)
    if enable_session_tracking:
        session_id = _get_or_init_session_id()
        with st.expander("🔐 Session Info", expanded=False):
            st.code(f"Session: {session_id[:8]}...", language="text")
            
            # Show clinical context if engine is loaded
            if "engine" in st.session_state:
                try:
                    engine = st.session_state.engine
                    session_info = engine.get_session_info(session_id)
                    if session_info:
                        st.markdown(f"**Turns:** {session_info['turns']}")
                        if session_info.get('anatomy'):
                            st.markdown(f"**Anatomy:** {session_info['anatomy']}")
                        if session_info.get('condition'):
                            st.markdown(f"**Condition:** {session_info['condition']}")
                        if session_info.get('context'):
                            st.markdown(f"**Retrieval Context:** `{session_info['context']}`")
                except Exception as e:
                    st.caption(f"Session info unavailable: {e}")
    
    st.divider()
    
    # UPDATED: Clear button resets session too
    if st.button("🗑️ Clear Chat & Reset Context"):
        _reset_conversation()

    st.divider()
    st.markdown("### 📖 About")
    st.markdown(
        "This system answers eye health questions using content from "
        "**Kanski's Clinical Ophthalmology** and **Khurana's "
        "Comprehensive Ophthalmology**.\n\n"
        "Pipeline: **EyeCLIP Image Analysis** → Query Refinement → "
        "Hybrid Retrieval → Cross-Encoder Re-Ranking → "
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


def _save_uploaded_image(uploaded_file) -> str | None:
    """Save uploaded file to /tmp and return the path, or None."""
    if uploaded_file is None:
        return None
    image_path = f"/tmp/eyeclip_{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return image_path


# ─── Chat State ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rolling_history" not in st.session_state:
    st.session_state.rolling_history = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "assistant" else "🙋"):
        if msg["role"] == "user" and msg.get("image_path"):
            if os.path.exists(msg["image_path"]):
                st.image(msg["image_path"], caption="📷 Uploaded Image", width=280)
        st.markdown(msg["content"], unsafe_allow_html=True)
        
        if msg["role"] == "assistant" and "pipeline" in msg and show_pipeline:
            with st.expander("🔍 Pipeline Details", expanded=False):
                _render_pipeline(msg["pipeline"])


# ─── Chat Input + Inline Image Upload ─────────────────────────────────────────
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

uploaded_file = st.file_uploader(
    "📷 Attach an eye image (optional)",
    type=["png", "jpg", "jpeg"],
    key=f"eye_img_{st.session_state.upload_key}",
    label_visibility="visible",
)

if prompt := st.chat_input("Ask an eye health question..."):
    # ── Step -1: Immediate Emergency Triage ─────
    emergency_response = check_red_flags(prompt)
    if emergency_response:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({
            "role": "assistant",
            "content": emergency_response,
            "pipeline": {"verdict": "EMERGENCY bypass", "time": 0.0},
        })
        st.session_state.upload_key += 1
        st.rerun()

    # Save image if uploaded
    image_path = _save_uploaded_image(uploaded_file)

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "image_path": image_path,
    })
    with st.chat_message("user", avatar="🙋"):
        if image_path:
            st.image(image_path, caption="📷 Uploaded Image", width=280)
        st.markdown(prompt)

    # Load engine
    engine = get_engine()
    
    # Get session_id if tracking enabled
    session_id = _get_or_init_session_id() if enable_session_tracking else None
    
    # Get recent queries for generation context (not retrieval!)
    recent_queries = _get_recent_queries_for_generation()

    # Run pipeline with status updates
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        pipeline_data = {}
        status_container = st.empty()
        start_time = time.time()

        # ── Step 0: EyeCLIP Image Analysis ────────────
        visual_findings = None
        if image_path and engine.vision_agent is not None:
            status_container.markdown(
                '<span class="pipeline-step step-eyeclip">👁️ Analyzing image with EyeCLIP...</span>',
                unsafe_allow_html=True,
            )
            visual_findings = engine.analyze_image(image_path)
            if visual_findings:
                pipeline_data["visual_findings"] = visual_findings

        # ── Step 1: Query Refinement or Rewriting ──────
        session = engine._get_or_create_session(session_id) if enable_session_tracking and session_id else None
        
        is_followup = (
            session is not None and 
            session.total_turns > 0 and 
            session.primary_condition is not None
        )
        
        if is_followup:
            status_container.markdown(
                '<span class="pipeline-step step-refine">🔄 Context-Aware Rewriting...</span>',
                unsafe_allow_html=True,
            )
            refined = engine.generator.rewrite_query_for_retrieval(
                current_query=prompt,
                session_state=session,
                visual_findings=visual_findings,
                image_path=image_path,
            )
        else:
            status_container.markdown(
                '<span class="pipeline-step step-refine">🔄 Refining query with MedGemma...</span>',
                unsafe_allow_html=True,
            )
            refined = engine.refine_query(
                prompt,
                recent_history=recent_queries if enable_session_tracking else None,
                image_path=image_path,
                visual_findings=visual_findings,
            )
            
        pipeline_data["refined_query"] = refined

        # ── Step 2: Hybrid Retrieval + Re-ranking ──────
        status_container.markdown(
            '<span class="pipeline-step step-retrieve">🔎 Searching textbooks...</span> '
            '<span class="pipeline-step step-rerank">📊 Re-ranking results...</span>',
            unsafe_allow_html=True,
        )

        retrieval_query = refined
        if visual_findings and not is_followup:
            from src.vision.eyeclip_agent import EyeClipAgent
            retrieval_terms = EyeClipAgent.get_retrieval_terms(visual_findings)
            if retrieval_terms:
                retrieval_query = f"{retrieval_terms} {refined}"

        child_hits = engine.hybrid_retriever.invoke(retrieval_query)[:num_sources * 4]
        reranked = engine.rerank(retrieval_query, child_hits, top_k=num_sources * 2)

        seen, context_docs = set(), []
        for child in reranked:
            p_id = child.metadata.get("parent_id")
            if p_id and p_id not in seen:
                parent_doc = engine.parent_store.get(p_id)
                if parent_doc:
                    context_docs.append(parent_doc)
                    seen.add(p_id)
            if len(context_docs) >= num_sources:
                break

        pipeline_data["num_sources"] = len(context_docs)
        pipeline_data["sources"] = [
            {
                "source": doc.metadata.get("source", "?"),
                "section_path": doc.metadata.get("section_path", "?"),
                "content": doc.page_content,
            }
            for doc in context_docs
        ]

        if not context_docs:
            answer = "I'm sorry, I couldn't find relevant information to answer your question. Please consult an eye care professional."
            pipeline_data["verdict"] = "N/A"
        else:
            # ── Step 3: Answer Generation ──────────────
            status_container.markdown(
                '<span class="pipeline-step step-generate">💬 Generating answer...</span>',
                unsafe_allow_html=True,
            )
            
            # NEW: Pass session_id to generate_answer via engine.ask() pattern
            answer = engine.generate_answer(
                prompt, 
                context_docs,
                session_state=engine._get_or_create_session(session_id) if enable_session_tracking and session_id else None,
                patient_profile=_get_patient_profile(),
                recent_history=recent_queries if enable_session_tracking else None,
                visual_findings=visual_findings,
                image_path=image_path,
            )

            # ── Step 4: Grounding Verification ─────────
            if not fast_mode:
                status_container.markdown(
                    '<span class="pipeline-step step-verify">🔬 Verifying facts...</span>',
                    unsafe_allow_html=True,
                )
                context_block = engine._build_context_block(context_docs)
                grounding = engine.verify_grounding(answer, context_block, verbose=False)
                pipeline_data["verdict"] = grounding["verdict"]
                pipeline_data["retries"] = 0

                # ── Step 5: Self-correction if needed ──────
                if grounding["verdict"] == "FAIL":
                    pipeline_data["flagged_claims"] = grounding["flagged_claims"]
                    status_container.markdown(
                        '<span class="pipeline-step step-correct">⚠️ Self-correcting hallucinations...</span>',
                        unsafe_allow_html=True,
                    )
                    flagged = "\n".join(f"- {c}" for c in grounding["flagged_claims"])
                    if not flagged:
                        flagged = "- Unspecified claims not supported by sources"
                    answer = engine.generate_answer(
                        prompt, 
                        context_docs,
                        correction_context=flagged,
                        session_state=engine._get_or_create_session(session_id) if enable_session_tracking and session_id else None,
                        patient_profile=_get_patient_profile(),
                        recent_history=recent_queries if enable_session_tracking else None,
                        visual_findings=visual_findings,
                        image_path=image_path,
                    )
                    grounding = engine.verify_grounding(answer, context_block, verbose=False)
                    pipeline_data["verdict"] = grounding["verdict"]
                    pipeline_data["retries"] = 1
            else:
                # Fast mode skips verification completely
                pipeline_data["verdict"] = "FAST MODE (Unverified)"
                pipeline_data["retries"] = 0

        # ── Step 6: Extract Entities & Update Session State ──────────
        if enable_session_tracking and session_id:
            session = engine._get_or_create_session(session_id)
            current_turn = session.total_turns + 1
            
            entities = engine.generator.extract_entities_from_answer(
                answer=answer,
                visual_findings=visual_findings,
                turn_id=current_turn,
            )
            
            session.update_from_entities(entities, current_turn)
            engine._persist_session(session)

        elapsed = time.time() - start_time
        pipeline_data["time"] = elapsed

        # Show final status
        if pipeline_data["verdict"] == "PASS":
            status_container.markdown(
                '<span class="pipeline-step step-pass">✅ Grounded & Verified</span> '
                f'<span class="pipeline-step" style="background:rgba(255,255,255,0.05);color:#9ca3af;">'
                f'⏱ {elapsed:.1f}s</span>',
                unsafe_allow_html=True,
            )
        elif pipeline_data["verdict"] == "FAST MODE (Unverified)":
            status_container.markdown(
                '<span class="pipeline-step step-pass" style="background:#4b5563;color:#d1d5db;">⚡ Fast Mode (Unverified)</span> '
                f'<span class="pipeline-step" style="background:rgba(255,255,255,0.05);color:#9ca3af;">'
                f'⏱ {elapsed:.1f}s</span>',
                unsafe_allow_html=True,
            )
        else:
            status_container.markdown(
                '<span class="pipeline-step step-correct">⚠️ Partially Verified</span> '
                f'<span class="pipeline-step" style="background:rgba(255,255,255,0.05);color:#9ca3af;">'
                f'⏱ {elapsed:.1f}s</span>',
                unsafe_allow_html=True,
            )

        # Display answer
        st.markdown(answer)

        # Show pipeline details
        if show_pipeline:
            with st.expander("🔍 Pipeline Details", expanded=False):
                _render_pipeline(pipeline_data)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "pipeline": pipeline_data,
    })

    # Update rolling history — store refined query + visual findings
    history_entry = refined
    if visual_findings:
        history_entry = f"{refined} [Image findings: {visual_findings}]"
    st.session_state.rolling_history.append(history_entry)
    st.session_state.rolling_history = st.session_state.rolling_history[-ROLLING_HISTORY_SIZE:]

    # Reset uploader
    st.session_state.upload_key += 1
    st.rerun()