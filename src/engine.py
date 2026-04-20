"""
query_engine.py — Orchestrator for the Self-Correcting Multimodal RAG Pipeline
UPDATED: Intelligent session state management with EyeCLIP-integrated entity extraction
────────────────────────────────────────────────────────────────────────────────
Uses `retriever.py` for search/ranking, `generator.py` for MedGemma inference,
and `eyeclip_agent.py` for zero-shot ophthalmic image classification.

NEW: Session-aware conversation handling with clinical context persistence.

Usage:
  # Backward compatible (no session tracking):
  engine = QueryEngine()
  answer, findings = engine.ask("my eye hurts in bright light")
  
  # Session-aware (recommended for multi-turn):
  answer1, session_id = engine.ask("What is this spot?", image_path="oct.png")
  answer2, session_id = engine.ask("What to do now?", session_id=session_id)
"""

import os
import uuid
import time
import re
from pathlib import Path
from typing import Optional, Tuple, List, Union, Any

# ─── GPU Configuration ────────────────────────────────────────────────────────
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"

from src.retriever import RetinaRetriever
from src.generator import MedGemmaGenerator
from src.vision.eyeclip_agent import EyeClipAgent
from src.triage import check_red_flags

# NEW: Import session state components
from src.state.clinical_session_state import ClinicalSessionState
from src.state.clinical_entity_extractor import ClinicalEntity, EntityType

# ASR (speech-to-text)
from src.speech.speech_recognizer import SpeechRecognizer, TranscriptionResult

MAX_CORRECTION_RETRIES = 1
SESSION_DIR = Path("./data/sessions")
SESSION_DIR.mkdir(parents=True, exist_ok=True)


class QueryEngine:
    """Orchestrates Retrieval, Generation, Grounding Verification, and Vision.
    
    NEW: Supports session-aware conversations with clinical context persistence
    and EyeCLIP-integrated entity extraction.
    """

    def __init__(self, enable_session_state: bool = True, config_path: str = "config.json"):
        self.retriever = RetinaRetriever()
        self.generator = MedGemmaGenerator()
        self.enable_session_state = enable_session_state
        
        # Load optional config
        self.config = {}
        if os.path.exists(config_path):
            import json
            try:
                with open(config_path) as f:
                    self.config = json.load(f)
                print(f"[QueryEngine] Loaded config from {config_path}")
            except Exception as e:
                print(f"[QueryEngine] ⚠ Failed to load config {config_path}: {e}")
        
        # In-memory session cache (persists across calls, cleared on restart)
        self._active_sessions: dict[str, ClinicalSessionState] = {}
        
        # Vision agent is optional
        try:
            self.vision_agent = EyeClipAgent()
            print("[QueryEngine] ✓ EyeCLIP vision agent loaded.")
        except Exception as e:
            print(f"[QueryEngine] ⚠ EyeCLIP failed to load: {e}")
            self.vision_agent = None

        # ASR (speech-to-text) agent is optional
        try:
            self.speech_recognizer = SpeechRecognizer()
            print("[QueryEngine] ✓ Speech recognizer loaded.")
        except Exception as e:
            print(f"[QueryEngine] ⚠ Speech recognizer failed to load: {e}")
            self.speech_recognizer = None

        # Start a background thread to run dummy inferences and cache GPU memory
        self._warmup_models()

    def _warmup_models(self):
        """Run a dummy forward pass on a background thread to cache CUDA memory / execution graphs."""
        import threading
        def warmup_task():
            try:
                print("[QueryEngine] ⚙️ Running background model warmup to reduce initial latency...")
                # Dummy Generation pass (shortest possible token generation)
                self.generator._generate([{"role": "user", "content": "hi"}], max_new_tokens=1)
                print("[QueryEngine] ✓ Background model warmup complete.")
            except Exception as e:
                print(f"[QueryEngine] ⚠ Background warmup failed: {e}")
                
        threading.Thread(target=warmup_task, daemon=True).start()

    # ── Pass-through accessors (backward compat) ─────────────────────────────
    def refine_query(self, raw_query: str, recent_history: Optional[list] = None, 
                     image_path: Optional[str] = None, visual_findings: Optional[str] = None) -> str:
        return self.generator.refine_query(raw_query, recent_history, 
                                          image_path=image_path, 
                                          visual_findings=visual_findings)

    @property
    def hybrid_retriever(self):
        return self.retriever.hybrid_retriever

    @property
    def parent_store(self):
        return self.retriever.parent_store

    def rerank(self, query, docs, top_k):
        return self.retriever._rerank(query, docs, top_k)

    def generate_answer(self, *args, **kwargs):
        return self.generator.generate_answer(*args, **kwargs)

    def _build_context_block(self, context_docs):
        return self.generator.build_context_block(context_docs)

    def verify_grounding(self, *args, **kwargs):
        return self.generator.verify_grounding(*args, **kwargs)

    # ── Session State Management ─────────────────────────────────────────────
    def _get_or_create_session(self, session_id: Optional[str]) -> ClinicalSessionState:
        """Get existing session or create new one."""
        if not self.enable_session_state:
            return ClinicalSessionState(session_id="dummy")
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Check in-memory cache first
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]
        
        # Try to load from disk
        session_path = SESSION_DIR / f"{session_id}.pkl"
        if session_path.exists():
            try:
                state = ClinicalSessionState.load(str(session_path))
                self._active_sessions[session_id] = state
                print(f"[Session] Loaded session {session_id[:8]}...")
                return state
            except Exception as e:
                print(f"[Session] ⚠ Failed to load {session_id}: {e}. Creating new.")
        
        # Create fresh session
        new_state = ClinicalSessionState(session_id=session_id)
        self._active_sessions[session_id] = new_state
        print(f"[Session] Created new session {session_id[:8]}...")
        return new_state

    def _persist_session(self, session: ClinicalSessionState):
        """Save session state to disk."""
        if not self.enable_session_state:
            return
        try:
            session_path = SESSION_DIR / f"{session.session_id}.pkl"
            session.save(str(session_path))
        except Exception as e:
            print(f"[Session] ⚠ Failed to persist session: {e}")

    # ── Speech-to-Text ────────────────────────────────────────────────────────
    def transcribe_audio(self, audio_input) -> TranscriptionResult | None:
        """
        Transcribe audio to text using faster-whisper.

        Args:
            audio_input: Raw audio bytes, io.BytesIO, or file path string.

        Returns:
            TranscriptionResult with .text, .duration_seconds, etc.
            None if ASR is unavailable.
        """
        if self.speech_recognizer is None or not self.speech_recognizer.is_ready:
            print("[QueryEngine] ⚠ Speech recognizer not available")
            return None
        try:
            return self.speech_recognizer.transcribe(audio_input)
        except Exception as e:
            print(f"[QueryEngine] ⚠ Transcription failed: {e}")
            return None

    @property
    def asr_ready(self) -> bool:
        """Whether the ASR model is loaded and ready."""
        return (
            self.speech_recognizer is not None
            and self.speech_recognizer.is_ready
        )

    # ── Image Analysis (unchanged) ───────────────────────────────────────────
    def analyze_image(self, image_path: str) -> Optional[str]:
        if self.vision_agent is None or not self.vision_agent.is_ready:
            return None

        try:
            modality_hint = self.generator.detect_modality_vlm(image_path)
        except Exception as e:
            print(f"[QueryEngine] ⚠ Modality detection failed: {e}")
            modality_hint = None

        return self.vision_agent.analyze_image(image_path, modality_hint=modality_hint)

    def _is_context_light_followup(self, query: str) -> bool:
        """Detect short/vague follow-ups that lack enough standalone clinical signal."""
        text = (query or "").strip().lower()
        if not text:
            return True

        tokens = re.findall(r"[a-z][a-z'-]*", text)
        if len(tokens) <= 3:
            return True

        medical_cues = {
            "eye", "cornea", "retina", "iris", "lens", "macula", "optic", "nerve",
            "pain", "redness", "watering", "discharge", "itching", "blurry", "vision",
            "floaters", "flashes", "photophobia", "spot", "ulcer", "keratitis",
            "conjunctivitis", "uveitis", "glaucoma", "cataract", "drusen", "edema",
            "exudates", "tearing", "swelling", "infection", "foreign", "body",
        }

        vague_tokens = {
            "what", "now", "next", "this", "that", "it", "same", "serious", "do",
            "should", "can", "how", "about", "then", "today", "again", "okay",
        }

        has_medical = any(tok in medical_cues for tok in tokens)
        vague_ratio = sum(1 for tok in tokens if tok in vague_tokens) / max(len(tokens), 1)

        return (not has_medical and len(tokens) <= 10) or vague_ratio >= 0.5

    def _last_substantive_user_query(
        self,
        recent_queries: Optional[List[str]],
        current_query: str,
    ) -> Optional[str]:
        """Return the most recent prior user query with clinical signal."""
        if not recent_queries:
            return None

        for q in reversed(recent_queries):
            if not q or q == current_query:
                continue
            if not self._is_context_light_followup(q):
                return q
        return None

    def _build_clinical_rerank_query(
        self,
        *,
        prompt: str,
        refined_query: str,
        session: Optional[ClinicalSessionState],
        recent_queries: Optional[List[str]],
    ) -> str:
        """
        Compose a rerank query robust for short follow-ups:
        current utterance + stable session terms + last substantive user query + compact refined terms.
        """
        signals: List[str] = [prompt]

        if session is not None and session.has_context(threshold=0.3, include_provisional=True):
            session_terms = session.to_query_terms(include_provisional=True)
            if session_terms:
                # Put session memory early so it is not clipped by downstream max-term limits.
                signals.insert(0, session_terms)

        if self._is_context_light_followup(prompt):
            last_substantive = self._last_substantive_user_query(recent_queries, prompt)
            if last_substantive:
                signals.append(last_substantive)

        compact_refined = self.generator.normalize_retrieval_query(refined_query or "", max_terms=12)
        if compact_refined:
            signals.append(compact_refined)

        merged = self._merge_query_signals(signals, max_terms=28)
        return merged if merged else prompt

    def _merge_query_signals(self, signals: List[str], max_terms: int = 18) -> str:
        """Merge multiple query signals without starving later context terms."""
        token_lists: List[List[str]] = []
        for signal in signals:
            if not signal:
                continue
            normalized = self.generator.normalize_retrieval_query(signal, max_terms=max_terms * 2)
            if normalized:
                token_lists.append(normalized.split())

        merged: List[str] = []
        seen = set()
        idx = 0
        while len(merged) < max_terms:
            progressed = False
            for tokens in token_lists:
                if idx >= len(tokens):
                    continue
                tok = tokens[idx].strip().lower()
                if tok and tok not in seen:
                    seen.add(tok)
                    merged.append(tok)
                    if len(merged) >= max_terms:
                        break
                progressed = True
            if not progressed:
                break
            idx += 1

        return " ".join(merged)

    # ── Full Pipeline (UPDATED with session awareness + EyeCLIP integration) ─
    def ask(
        self,
        raw_query: str,
        image_path: Optional[str] = None,
        k: int = 3,
        verbose: bool = True,
        session_id: Optional[str] = None,
        recent_history: Optional[List[str]] = None,
        patient_profile: Optional[dict] = None,
        fast_mode: bool = False,
        use_session_state: Optional[bool] = None,
        return_trace: bool = False,
    ) -> Any:
        """
        Complete self-correcting RAG pipeline with optional session awareness.
        
        Args:
            raw_query: Patient's question
            image_path: Optional path to ophthalmic image
            k: Number of documents to retrieve
            verbose: Print debug info
            session_id: Optional session identifier for conversation continuity
            recent_history: List of prior user queries (for generation context)
            patient_profile: Optional dict with patient demographics/conditions
        
        Returns:
            If session tracking enabled and session_id was None:
                (answer_text, visual_findings_or_None, new_session_id)
            Otherwise:
                (answer_text, visual_findings_or_None)
        """
        request_session_id = session_id
        session_enabled = self.enable_session_state if use_session_state is None else bool(use_session_state)

        trace: dict[str, Any] = {
            "num_sources": 0,
            "sources": [],
            "retries": 0,
            "flagged_claims": [],
            "time": 0.0,
        }
        start_time = time.time()

        def _format_return(answer_text: str, visual: Optional[str], active_session: Optional[ClinicalSessionState]) -> Any:
            # Always include session_id when session tracking is enabled so the
            # caller can update its stored ID on every turn, not just the first.
            if session_enabled and active_session is not None:
                base = (answer_text, visual, active_session.session_id)
            else:
                base = (answer_text, visual)
            if return_trace:
                return (*base, trace)
            return base

        # ── Step -1: Emergency Triage ────────────────────────────────────────
        emergency_response = check_red_flags(raw_query)
        if emergency_response:
            if verbose:
                print(f"[QueryEngine] 🚨 Red Flag Triggered: {raw_query}")
            trace["verdict"] = "SAFETY TRIAGE bypass"
            trace["time"] = 0.0
            return _format_return(emergency_response, None, None)

        # ── Session Setup ───────────────────────────────────────────────────
        session: Optional[ClinicalSessionState] = None
        if session_enabled:
            session = self._get_or_create_session(session_id)
            current_turn = session.total_turns + 1
        else:
            current_turn = 1
        
        # Check if session should be reset
        if session_enabled and session is not None and session.should_reset():
            if verbose:
                print(f"[QueryEngine] Resetting session due to inactivity/topic drift")
            session = session.reset_for_new_topic()
            session_id = session.session_id
            self._active_sessions[session_id] = session

        # ── Step 0: Vision Analysis (optional) ───────────────────────────────
        visual_findings = None
        if image_path:
            if verbose:
                print("[QueryEngine] 👁️ Analyzing uploaded image with EyeCLIP...")
            visual_findings = self.analyze_image(image_path)
            if visual_findings and verbose:
                print(f"  Findings: {visual_findings}")
            
            # If the session already has image-derived context (anatomy/conditions/findings),
            # a new image means a new clinical topic. Reset image-derived context so the
            # new EyeCLIP findings aren't blocked by reinforced old context.
            if session_enabled and session is not None and session.has_context():
                session.reset_for_new_image()
                if verbose:
                    print("[QueryEngine] 🔄 Cleared old image context for new image")

        # ── Step 1: Query Refinement/Rewriting ───────────────────────────────
        if verbose:
            print(f"[QueryEngine] Turn {current_turn}: Refining query...")
        
        # Determine if this is a follow-up with pinned clinical context
        is_followup = (
            session_enabled and
            session is not None and
            current_turn > 1 and 
            session.has_context(include_provisional=True)
        )
        
        if is_followup:
            # Use context-aware rewriting for follow-ups
            retrieval_query = self.generator.rewrite_query_for_retrieval(
                current_query=raw_query,
                session_state=session,
                visual_findings=visual_findings,
                image_path=image_path,
                recent_history=recent_history,
            )
            if verbose:
                print(f"  [Rewritten] {retrieval_query}")
        else:
            # Standard refinement for first turn or no pinned context
            refined = self.generator.refine_query(
                raw_query, 
                recent_history=recent_history,
                visual_findings=visual_findings,
                image_path=image_path,
            )
            
            # Augment with EyeCLIP terms if available
            retrieval_query = refined
            if visual_findings:
                from src.vision.eyeclip_agent import EyeClipAgent
                retrieval_terms = EyeClipAgent.get_retrieval_terms(visual_findings)
                if retrieval_terms:
                    retrieval_query = f"{retrieval_terms} {refined}"
            
            if verbose:
                print(f"  [Refined] {retrieval_query}")

        # Always enrich retrieval query with high-confidence persistent session terms.
        if session_enabled and session is not None and session.has_context(threshold=0.3, include_provisional=True):
            session_terms = session.to_query_terms(include_provisional=True)
            if session_terms:
                retrieval_query = self._merge_query_signals([session_terms, retrieval_query], max_terms=18)
                if verbose:
                    print(f"  [Refined+Session] {retrieval_query}")

        # Final guardrail: enforce compact deduplicated retrieval keywords.
        normalized_query = self.generator.normalize_retrieval_query(retrieval_query, max_terms=18)
        if normalized_query:
            retrieval_query = normalized_query
            if verbose:
                print(f"  [Normalized Query] {retrieval_query}")

        retrieval_query = self.generator.apply_symptom_sign_mapping_to_query(
            raw_query=raw_query,
            candidate_query=retrieval_query,
            max_terms=18,
        )
        if verbose:
            print(f"  [Mapped Query] {retrieval_query}")
        trace["refined_query"] = retrieval_query

        # ── Step 2: Retrieve Documents ───────────────────────────────────────
        child_hits_refined = self.retriever.hybrid_retriever.invoke(retrieval_query)[: max(k * 4, 8)]

        raw_augmented_query = raw_query
        raw_signal_parts: List[str] = [raw_query]
        if self._is_context_light_followup(raw_query):
            last_substantive = self._last_substantive_user_query(recent_history, raw_query)
            if last_substantive:
                raw_signal_parts.append(last_substantive)

        if session_enabled and session is not None and session.has_context(threshold=0.3, include_provisional=True):
            session_terms = session.to_query_terms(include_provisional=True)
            if session_terms:
                raw_signal_parts.insert(0, session_terms)

        raw_augmented_query = self._merge_query_signals(raw_signal_parts, max_terms=18)

        normalized_raw_query = self.generator.normalize_retrieval_query(raw_augmented_query, max_terms=18)
        raw_retrieval_query = normalized_raw_query if normalized_raw_query else raw_augmented_query
        raw_retrieval_query = self.generator.apply_symptom_sign_mapping_to_query(
            raw_query=raw_query,
            candidate_query=raw_retrieval_query,
            max_terms=18,
        )
        if verbose:
            print(f"[QueryEngine] Raw-anchor retrieval query: {raw_retrieval_query}")

        child_hits_raw = self.retriever.hybrid_retriever.invoke(raw_retrieval_query)[: max(k * 3, 6)]

        merged_child_hits = []
        seen_child_keys = set()
        for child in child_hits_refined + child_hits_raw:
            key = (
                child.metadata.get("parent_id"),
                child.metadata.get("source"),
                child.metadata.get("section_path"),
                child.page_content[:120],
            )
            if key in seen_child_keys:
                continue
            seen_child_keys.add(key)
            merged_child_hits.append(child)

        rerank_query = self._build_clinical_rerank_query(
            prompt=raw_query,
            refined_query=retrieval_query,
            session=session,
            recent_queries=recent_history,
        )
        rerank_query = self.generator.apply_symptom_sign_mapping_to_query(
            raw_query=raw_query,
            candidate_query=rerank_query,
            max_terms=28,
        )
        reranked_children = self.retriever._rerank(rerank_query, merged_child_hits, top_k=max(k * 2, 4))

        if verbose and reranked_children:
            print("[QueryEngine] Top reranked child hits:")
            for i, child in enumerate(reranked_children[: min(len(reranked_children), max(k, 3))], 1):
                src = child.metadata.get("source", "N/A")
                sp = child.metadata.get("section_path", "N/A")
                score = child.metadata.get("rerank_score")
                score_txt = f"{float(score):.4f}" if isinstance(score, (int, float)) else "N/A"
                print(f"  [{i}] score={score_txt} [{src}] {sp}")

        seen_parents, context_docs = set(), []
        for child in reranked_children:
            p_id = child.metadata.get("parent_id")
            if p_id and p_id not in seen_parents:
                parent_doc = self.retriever.parent_store.get(p_id)
                if parent_doc:
                    context_docs.append(parent_doc)
                    seen_parents.add(p_id)
            if len(context_docs) >= k:
                break

        # ── Step 2.5: Zero-Recall Fallback ───────────────────────────────────
        if not context_docs:
            if verbose:
                print(f"[QueryEngine] ⚠️ Zero docs retrieved for '{retrieval_query}'. Triggering fallback expansion...")
            # Generate a broader query using synonyms/hypernyms
            broad_query = self.generator.refine_query(
                f"The specific term '{retrieval_query}' returned no results. Generate a broader ophthalmic synonym or category for this condition.",
                recent_history=None
            )
            if verbose:
                print(f"  [Fallback Query] {broad_query}")
            
            # Retry search with broader query
            context_docs = self.retriever.search(broad_query, k=k, verbose=verbose)

        if not context_docs:
            fallback = (
                "I'm sorry, I couldn't find relevant information in my medical texts "
                "to answer your question. Please consult an eye care professional."
            )
            if self.enable_session_state:
                if session_enabled and session is not None:
                    session.total_turns = current_turn
                    session.last_active_turn = current_turn
                    self._persist_session(session)
            
            trace["verdict"] = "N/A"
            return _format_return(fallback, visual_findings, session)

        trace["num_sources"] = len(context_docs)
        trace["sources"] = [
            {
                "source": doc.metadata.get("source", "?"),
                "section_path": doc.metadata.get("section_path", "?"),
                "content": doc.page_content,
            }
            for doc in context_docs
        ]

        context_block = self.generator.build_context_block(context_docs)

        # ── Step 3: Generate Answer ──────────────────────────────────────────
        if verbose:
            print("[QueryEngine] Generating patient-friendly answer...")
        
        answer = self.generator.generate_answer(
            raw_query=raw_query,
            context_docs=context_docs,
            session_state=session if session_enabled else None,
            correction_context=None,
            patient_profile=patient_profile,
            recent_history=recent_history,
            visual_findings=visual_findings,
            image_path=image_path,
        )

        # ── Step 4: Verify Grounding ─────────────────────────────────────────
        if verbose:
            print("\n[QueryEngine] Verifying answer grounding...")
        
        if fast_mode:
            grounding = {"verdict": "FAST MODE (Unverified)", "flagged_claims": [], "retries": 0}
        else:
            # Combine textual context with visual findings so claims about
            # the image aren't penalized.
            verification_context = context_block
            if visual_findings:
                verification_context += f"\n\n[Visual Findings (Ground Truth)]\n{visual_findings}"
                
            grounding = self._verify_grounding_medcpt(
                answer,
                verification_context,
                verbose=verbose,
            )
            grounding["retries"] = 0

        # ── Step 5: Self-Correction Loop ─────────────────────────────────────
        # Only trigger on FAIL (contradicted claims). PARTIAL is acceptable —
        # it means most claims are grounded with a few unverifiable statements,
        # which is normal for paraphrased medical advice.
        retries = 0
        while (not fast_mode) and grounding["verdict"] == "FAIL" and retries < MAX_CORRECTION_RETRIES:
            retries += 1
            if verbose:
                print(f"\n[QueryEngine] ⚠️  Grounding FAIL — self-correcting (attempt {retries})...")

            # Extract flagged claims from new structure (claim records have 'claim' key)
            unsupported = grounding.get("flagged_claims", [])
            if not unsupported:
                unsupported_records = grounding.get("unsupported_claims", [])
                if unsupported_records:
                    unsupported = [
                        r["claim"] if isinstance(r, dict) else str(r)
                        for r in unsupported_records
                    ]

            flagged = "\n".join(f"- {c}" for c in unsupported)
            if not flagged:
                flagged = "- Unspecified claims not supported by sources"

            answer = self.generator.generate_answer(
                raw_query,
                context_docs,
                correction_context=flagged,
                session_state=session if session_enabled else None,
                patient_profile=patient_profile,
                recent_history=recent_history,
                visual_findings=visual_findings,
                image_path=image_path,
            )

            if verbose:
                print("[QueryEngine] Re-verifying corrected answer...")
            grounding = self._verify_grounding_medcpt(
                answer,
                verification_context,
                verbose=verbose,
            )
            grounding["retries"] = retries

        # Extract flagged claims for trace (handle both old and new formats)
        flagged_for_trace = grounding.get("flagged_claims", [])
        if not flagged_for_trace:
            unsupported_records = grounding.get("unsupported_claims", [])
            if unsupported_records:
                flagged_for_trace = [
                    r["claim"] if isinstance(r, dict) else str(r)
                    for r in unsupported_records
                ]

        trace["verdict"] = grounding.get("verdict", "N/A")
        trace["retries"] = retries
        trace["flagged_claims"] = flagged_for_trace

        # ── Step 5.5: Apply Abstention Disclaimer ────────────────────────────
        # If grounding is PARTIAL or FAIL after self-correction, prepend a
        # safety disclaimer so the user knows the answer isn't fully verified.
        if not fast_mode:
            answer = self.generator.apply_abstention_disclaimer(answer, grounding)

        # ── Step 6: Extract Entities & Update Session State ──────────────────
        if session_enabled and session is not None:
            entities = self.generator.extract_entities_from_turn(
                query_text=raw_query,
                answer_text=answer,
                visual_findings=visual_findings,
                turn_id=current_turn,
            )

            text_for_extraction = f"Patient Question: {raw_query}"
            if visual_findings:
                text_for_extraction += f"\nEyeCLIP Findings: {visual_findings}"
            # Update session with merged entities (EyeCLIP + text)
            session.update_from_entities(entities, current_turn, text=text_for_extraction)
            
            # Persist to disk
            self._persist_session(session)
            
            if verbose:
                ctx = session.to_query_context()
                if ctx:
                    print(f"[Session] Updated context: {ctx}")

        trace["visual_findings"] = visual_findings
        trace["time"] = time.time() - start_time

        # ── Return ───────────────────────────────────────────────────────────
        if verbose:
            if grounding["verdict"] == "PASS":
                status = "✅ GROUNDED"
            elif grounding["verdict"] == "PARTIAL":
                status = "⚠️ PARTIALLY GROUNDED (disclaimer added)"
            else:
                status = "❌ UNGROUNDED (disclaimer added)"
            print(f"\n[QueryEngine] Final status: {status}")

        return _format_return(answer, visual_findings, session)

    # ── MedCPT-Based Grounding Verification ──────────────────────────────────
    def _verify_grounding_medcpt(
        self,
        answer: str,
        context_block: str,
        verbose: bool = True,
    ) -> dict:
        """
        Verify answer grounding using the MedCPT Cross-Encoder (already loaded
        for reranking) instead of DeBERTa NLI.

        Why MedCPT > DeBERTa NLI for medical grounding:
          - MedCPT: 255M PubMed query-article pairs → understands medical
            paraphrasing, synonyms, and clinical reasoning natively.
          - DeBERTa NLI: generic NLI on MNLI/SNLI → flags nearly every
            medical paraphrase as "neutral" (unverified).

        Approach:
          1. Sentence-tokenize the answer into atomic claims.
          2. Chunk context into overlapping windows (512-token limit).
          3. Score each (claim, window) pair with MedCPT.
          4. A claim is "supported" if its best window score exceeds a
             relevance threshold — high relevance from the source context
             that generated the answer ≈ grounded.

        Verdict tiers:
          PASS    — all claims supported
          PARTIAL — some unverifiable but none contradicted / <50% unsupported
          FAIL    — majority of claims have very low relevance (≤ low_threshold)
        """
        import torch

        # ── Quick exits ──────────────────────────────────────────────────────
        refusal_patterns = [
            "outside my ophthalmology knowledge",
            "outside my knowledge base",
            "cannot answer this",
            "i am sorry",
            "i'm sorry",
            "couldn't find relevant information",
        ]
        if len(answer) < 150 and any(p in answer.lower() for p in refusal_patterns):
            if verbose:
                print("[Grounding MedCPT] Standard refusal — bypass.")
            return {
                "verdict": "PASS",
                "flagged_claims": [],
                "unsupported_claims": [],
                "reasoning": "Standard refusal bypass.",
            }

        if not context_block.strip():
            return {
                "verdict": "FAIL",
                "flagged_claims": [],
                "unsupported_claims": [],
                "reasoning": "Empty context.",
            }

        # ── Sentence-tokenize the answer into atomic claims ──────────────────
        atomic_claims: list[str] = []
        seen: set[str] = set()

        # First split by newlines (generator output has markdown structure),
        # then by sentence boundaries within each line.
        boilerplate_skip = {
            "consult your doctor", "see a specialist", "seek professional",
            "healthcare provider", "seek medical attention", "please visit",
            "go to the emergency", "automated screening", "preliminary findings",
            "only a professional", "evaluation by a qualified",
            "perform a comprehensive eye exam", "healthy lifestyle",
            "i am an ai", "i'm an ai", "disclaimer",
        }

        for line in answer.split("\n"):
            line = line.strip()
            if not line or line.endswith(":"):
                continue
            low_line = line.lower()
            if any(kw in low_line for kw in boilerplate_skip):
                continue

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', line)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 25:
                    continue
                # Skip bullet-only items (e.g. "- Diabetes")
                clean = re.sub(r'^[-•*]\s*', '', sent)
                if len(clean.split()) <= 4:
                    continue
                key = clean.lower()
                if key not in seen:
                    seen.add(key)
                    atomic_claims.append(clean)

        if not atomic_claims:
            if verbose:
                print("[Grounding MedCPT] No verifiable claims extracted → PASS")
            return {
                "verdict": "PASS",
                "flagged_claims": [],
                "unsupported_claims": [],
                "reasoning": "No verifiable claims extracted.",
            }

        # ── Chunk context into overlapping windows ───────────────────────────
        tokenizer = self.retriever.reranker_tokenizer
        MAX_CTX_TOKENS = 400  # Leave room for claim tokens in the 512 budget
        OVERLAP = 80

        ctx_ids = tokenizer.encode(context_block, add_special_tokens=False)

        if len(ctx_ids) <= MAX_CTX_TOKENS:
            context_windows = [context_block]
        else:
            context_windows = []
            start = 0
            while start < len(ctx_ids):
                end = min(start + MAX_CTX_TOKENS, len(ctx_ids))
                window_text = tokenizer.decode(ctx_ids[start:end], skip_special_tokens=True)
                context_windows.append(window_text)
                if end >= len(ctx_ids):
                    break
                start += MAX_CTX_TOKENS - OVERLAP

        # ── Score each claim against all windows ─────────────────────────────
        SUPPORT_THRESHOLD = 0.35   # Above this = supported (context is relevant)
        LOW_THRESHOLD = 0.10       # Below this = clearly unsupported

        supported_claims: list[dict] = []
        unsupported_claims: list[dict] = []
        weak_claims: list[dict] = []  # Between low and support thresholds

        model = self.retriever.reranker_model
        device = self.retriever.reranker_device

        for claim in atomic_claims:
            pairs = [[claim, window] for window in context_windows]

            with torch.no_grad():
                encoded = tokenizer(
                    pairs,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(device)
                logits = model(**encoded).logits

                # Collapse to scalar relevance scores (same logic as _rerank)
                if logits.dim() == 1:
                    relevance = logits
                elif logits.size(-1) == 1:
                    relevance = logits.squeeze(dim=-1)
                else:
                    relevance = torch.softmax(logits, dim=-1)[..., -1]

                if relevance.dim() == 0:
                    relevance = relevance.unsqueeze(0)

                best_score = relevance.max().item()

            record = {
                "claim": claim,
                "relevance_score": round(best_score, 4),
            }

            if best_score >= SUPPORT_THRESHOLD:
                record["status"] = "supported"
                supported_claims.append(record)
            elif best_score <= LOW_THRESHOLD:
                record["status"] = "unsupported"
                unsupported_claims.append(record)
            else:
                record["status"] = "weak"
                weak_claims.append(record)

        # ── Determine verdict ────────────────────────────────────────────────
        total = len(supported_claims) + len(unsupported_claims) + len(weak_claims)
        unsupported_ratio = len(unsupported_claims) / max(total, 1)

        if unsupported_ratio > 0.5:
            verdict = "FAIL"
            reasoning = (
                f"Majority of claims unsupported: {len(unsupported_claims)}/{total} "
                f"below relevance threshold ({len(context_windows)} context windows)."
            )
        elif unsupported_claims or weak_claims:
            verdict = "PARTIAL"
            reasoning = (
                f"{len(supported_claims)}/{total} claims supported, "
                f"{len(weak_claims)} weakly matched, "
                f"{len(unsupported_claims)} unsupported "
                f"({len(context_windows)} context windows)."
            )
        else:
            verdict = "PASS"
            reasoning = (
                f"All {len(supported_claims)} claims supported by context "
                f"({len(context_windows)} context windows)."
            )

        # Flagged = only the actually unsupported ones (not weak)
        flagged = [r["claim"] for r in unsupported_claims]

        if verbose:
            print(f"[Grounding MedCPT] Verdict: {verdict}")
            print(f"  Supported: {len(supported_claims)}, Weak: {len(weak_claims)}, "
                  f"Unsupported: {len(unsupported_claims)} / {total} total claims")
            for r in unsupported_claims:
                print(f"  ⚠️  {r['relevance_score']:.4f} | {r['claim'][:80]}...")
            if weak_claims and verbose:
                for r in weak_claims[:3]:
                    print(f"  ℹ️  {r['relevance_score']:.4f} | {r['claim'][:80]}...")

        return {
            "verdict": verdict,
            "flagged_claims": flagged,
            "unsupported_claims": unsupported_claims + weak_claims,
            "reasoning": reasoning,
            "supported_count": len(supported_claims),
            "unsupported_count": len(unsupported_claims),
            "weak_count": len(weak_claims),
        }

    # ── Utility Methods ──────────────────────────────────────────────────────
    def clear_session_cache(self):
        """Clear in-memory session cache (does not delete persisted files)."""
        self._active_sessions.clear()
        print("[QueryEngine] Session cache cleared.")

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get summary of session state for debugging."""
        if not self.enable_session_state:
            return None
        session = self._active_sessions.get(session_id)
        if not session:
            session_path = SESSION_DIR / f"{session_id}.pkl"
            if session_path.exists():
                try:
                    session = ClinicalSessionState.load(str(session_path))
                except Exception:
                    return None
        if session:
            retrieval_context = session.to_query_context(include_provisional=True)
            return {
                "session_id": session.session_id,
                "turns": session.total_turns,
                "anatomy": session.anatomy_of_interest.value if session.anatomy_of_interest else None,
                "condition": session.primary_condition.value if session.primary_condition else None,
                "context": retrieval_context,
                "query_terms": session.to_query_terms(include_provisional=True),
                "session_json": session.to_dict(),
            }
        return None


# ─── CLI Quick Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = QueryEngine()

    test_queries = [
        "my eye hurts when I look at bright lights",
        "Who is the president of Angola?",
    ]

    print("\n" + "=" * 60)
    print("SELF-CORRECTING RAG — Full Pipeline Test")
    print("=" * 60)

    for q in test_queries:
        print(f"\n{'─' * 60}")
        result = engine.ask(q, k=3)
        answer = result[0] if len(result) == 2 else result[0]
        print(f"\n💬 FINAL ANSWER:\n{answer}\n")