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
from pathlib import Path
from typing import Optional, Tuple, List, Union

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

MAX_CORRECTION_RETRIES = 1
SESSION_DIR = Path("./data/sessions")
SESSION_DIR.mkdir(parents=True, exist_ok=True)


class QueryEngine:
    """Orchestrates Retrieval, Generation, Grounding Verification, and Vision.
    
    NEW: Supports session-aware conversations with clinical context persistence
    and EyeCLIP-integrated entity extraction.
    """

    def __init__(self, enable_session_state: bool = True):
        self.retriever = RetinaRetriever()
        self.generator = MedGemmaGenerator()
        self.enable_session_state = enable_session_state
        
        # In-memory session cache (persists across calls, cleared on restart)
        self._active_sessions: dict[str, ClinicalSessionState] = {}
        
        # Vision agent is optional
        try:
            self.vision_agent = EyeClipAgent()
            print("[QueryEngine] ✓ EyeCLIP vision agent loaded.")
        except Exception as e:
            print(f"[QueryEngine] ⚠ EyeCLIP failed to load: {e}")
            self.vision_agent = None

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
    ) -> Union[Tuple[str, Optional[str]], Tuple[str, Optional[str], str]]:
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
        # ── Step -1: Emergency Triage ────────────────────────────────────────
        emergency_response = check_red_flags(raw_query)
        if emergency_response:
            if verbose:
                print(f"[QueryEngine] 🚨 Red Flag Triggered: {raw_query}")
            return (emergency_response, None)

        # ── Session Setup ───────────────────────────────────────────────────
        session = self._get_or_create_session(session_id)
        current_turn = session.total_turns + 1 if self.enable_session_state else 1
        
        # Check if session should be reset
        if self.enable_session_state and session.should_reset():
            if verbose:
                print(f"[QueryEngine] Resetting session due to inactivity/topic drift")
            session = session.reset_for_new_topic()
            session_id = session.session_id

        # ── Step 0: Vision Analysis (optional) ───────────────────────────────
        visual_findings = None
        if image_path:
            if verbose:
                print("[QueryEngine] 👁️ Analyzing uploaded image with EyeCLIP...")
            visual_findings = self.analyze_image(image_path)
            if visual_findings and verbose:
                print(f"  Findings: {visual_findings}")

        # ── Step 1: Query Refinement/Rewriting ───────────────────────────────
        if verbose:
            print(f"[QueryEngine] Turn {current_turn}: Refining query...")
        
        # Determine if this is a follow-up with pinned clinical context
        is_followup = (
            self.enable_session_state and 
            current_turn > 1 and 
            session.primary_condition is not None
        )
        
        if is_followup:
            # Use context-aware rewriting for follow-ups
            retrieval_query = self.generator.rewrite_query_for_retrieval(
                current_query=raw_query,
                session_state=session,
                visual_findings=visual_findings,
                image_path=image_path,
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

        # ── Step 2: Retrieve Documents ───────────────────────────────────────
        context_docs = self.retriever.search(retrieval_query, k=k, verbose=verbose)

        if not context_docs:
            fallback = (
                "I'm sorry, I couldn't find relevant information to answer "
                "your question. Please consult an eye care professional."
            )
            if self.enable_session_state:
                session.total_turns = current_turn
                session.last_active_turn = current_turn
                self._persist_session(session)
            return (fallback, visual_findings)

        context_block = self.generator.build_context_block(context_docs)

       # ── Step 3: Generate Answer ──────────────────────────────────────────
        if verbose:
            print("[QueryEngine] Generating patient-friendly answer...")
        
        answer = self.generator.generate_answer(
            raw_query=raw_query,
            context_docs=context_docs,
            session_state=session if self.enable_session_state else None,
            correction_context=None,
            patient_profile=patient_profile,
            recent_history=recent_history,
            visual_findings=visual_findings,
            image_path=image_path,
        )

        # ── Step 4: Verify Grounding ─────────────────────────────────────────
        if verbose:
            print("\n[QueryEngine] Verifying answer grounding...")
        
        # Detect query anatomy for verification
        query_anatomy = self.generator._detect_anatomy(raw_query) if self.enable_session_state else None
        
        grounding = self.generator.verify_grounding(
            answer, 
            context_block, 
            query_anatomy=query_anatomy,
            verbose=verbose
        )
        grounding["retries"] = 0

        # ── Step 5: Self-Correction Loop ─────────────────────────────────────
        retries = 0
        while grounding["verdict"] == "FAIL" and retries < MAX_CORRECTION_RETRIES:
            retries += 1
            if verbose:
                print(f"\n[QueryEngine] ⚠️  Grounding FAILED — self-correcting (attempt {retries})...")

            flagged = "\n".join(f"- {c}" for c in grounding["flagged_claims"])
            if not flagged:
                flagged = "- Unspecified claims not supported by sources"

            answer = self.generator.generate_answer(
                raw_query,
                context_docs,
                correction_context=flagged,
                session_state=session if self.enable_session_state else None,
                patient_profile=patient_profile,
                recent_history=recent_history,
                visual_findings=visual_findings,
                image_path=image_path,
            )

            if verbose:
                print("[QueryEngine] Re-verifying corrected answer...")
            grounding = self.generator.verify_grounding(
                answer, 
                context_block, 
                query_anatomy=query_anatomy,
                verbose=verbose
            )
            grounding["retries"] = retries

        # ── Step 6: Extract Entities & Update Session State ──────────────────
        if self.enable_session_state:
            # Extract entities from final answer + EyeCLIP findings
            entities = self.generator.extract_entities_from_answer(
                answer=answer,
                visual_findings=visual_findings,
                turn_id=current_turn,
            )
            
            # Update session with merged entities (EyeCLIP + text)
            session.update_from_entities(entities, current_turn, text=answer)
            
            # Persist to disk
            self._persist_session(session)
            
            if verbose:
                ctx = session.to_query_context()
                if ctx:
                    print(f"[Session] Updated context: {ctx}")

        # ── Return ───────────────────────────────────────────────────────────
        if verbose:
            status = "✅ GROUNDED" if grounding["verdict"] == "PASS" else "⚠️ PARTIALLY GROUNDED"
            print(f"\n[QueryEngine] Final status: {status}")

        # Handle return value based on session tracking
        if self.enable_session_state and session_id is None:
            # Return new session_id for caller to persist
            return (answer, visual_findings, session.session_id)
        else:
            # Backward compatible: 2-tuple return
            return (answer, visual_findings)

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
                except:
                    return None
        if session:
            return {
                "session_id": session.session_id,
                "turns": session.total_turns,
                "anatomy": session.anatomy_of_interest.value if session.anatomy_of_interest else None,
                "condition": session.primary_condition.value if session.primary_condition else None,
                "context": session.to_query_context(),
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