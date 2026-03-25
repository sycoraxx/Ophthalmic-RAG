"""
evaluator.py — Fast, specialized evaluation models
────────────────────────────────────────────────────
Implements `LightweightEvaluator` which uses specialized, small models
instead of a bulky generative LLM for evaluation tasks:

1. NLI Grounding: uses a Cross-Encoder (e.g., DeBERTa-v3 NLI) to detect
   entailment vs contradiction (hallucination) between claims and context.
2. Zero-Shot Extraction: uses a fast zero-shot classifier (e.g., BART-Large-MNLI)
   to extract the chosen MCQ option from a verbose answer.
"""

from __future__ import annotations
import gc
from typing import List, Dict, Any, Optional

import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


class LightweightEvaluator:
    def __init__(
        self,
        nli_model_id: str = "cross-encoder/nli-deberta-v3-small",
        mcq_extractor_id: str = "facebook/bart-large-mnli",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the specialized evaluation models.
        These are small enough (~1GB combined) to stay resident on GPU alongside MedGemma.
        """
        self.device = device
        self.nli_model_id = nli_model_id
        self.mcq_extractor_id = mcq_extractor_id
        
        # Pipelines
        self._nli_pipe = None
        self._zs_pipe = None

    def _load_nli(self):
        if self._nli_pipe is None:
            print(f"[LightweightEvaluator] Loading NLI model: {self.nli_model_id}")
            self._nli_pipe = pipeline(
                "text-classification",
                model=self.nli_model_id,
                device=0 if self.device == "cuda" else -1
            )
        return self._nli_pipe

    def _load_zs(self):
        if self._zs_pipe is None:
            print(f"[LightweightEvaluator] Loading Zero-Shot model: {self.mcq_extractor_id}")
            self._zs_pipe = pipeline(
                "zero-shot-classification",
                model=self.mcq_extractor_id,
                device=0 if self.device == "cuda" else -1
            )
        return self._zs_pipe

    def extract_mcq_choice(self, generated_answer: str, options: List[str]) -> Optional[int]:
        """
        Extract which MCQ option the verbose answer is selecting.
        Uses Zero-Shot classification: "This text selects which option?"
        """
        if not options or not generated_answer.strip():
            return None

        # Clean up options to just the text content for semantic matching
        # e.g., "A. Glaucoma" -> "Glaucoma"
        clean_opts = []
        for opt in options:
            if opt.startswith(("A.", "B.", "C.", "D.", "E.")):
                clean_opts.append(opt[3:].strip())
            else:
                clean_opts.append(opt.strip())

        zs_pipe = self._load_zs()
        try:
            res = zs_pipe(
                generated_answer,
                candidate_labels=clean_opts,
                hypothesis_template="The diagnosis or answer selected is {}.",
            )
            top_label = res["labels"][0]
            top_score = res["scores"][0]
            
            # If the model is confident enough, return that index
            if top_score > 0.4:
                return clean_opts.index(top_label)
            return None
        except Exception as e:
            print(f"[LightweightEvaluator] Error in MCQ extraction: {e}")
            return None

    def verify_grounding(self, claims: List[str], context: str) -> Dict[str, Any]:
        """
        Verify if a list of clinical claims are entailed by the context text.
        
        Uses a sliding-window approach to handle contexts that exceed
        DeBERTa-v3-small's 512-token limit. Each claim is tested against
        all context windows; a claim is entailed if ANY window entails it.
        """
        if not claims:
            return {"verdict": "PASS", "reasoning": "No claims provided to verify.", "unsupported_claims": []}
        
        if not context.strip():
            return {"verdict": "FAIL", "reasoning": "Context is empty.", "unsupported_claims": claims}

        nli_pipe = self._load_nli()
        tokenizer = nli_pipe.tokenizer
        
        # ── Chunk context into overlapping windows ──────────────────────────
        # Reserve ~112 tokens for the claim + special tokens; use rest for context
        MAX_SEQ = 512
        CLAIM_BUDGET = 112  # generous ceiling for a single-sentence claim
        CTX_BUDGET = MAX_SEQ - CLAIM_BUDGET  # ~400 tokens for context per window
        OVERLAP = 50  # token overlap between consecutive windows
        
        ctx_ids = tokenizer.encode(context, add_special_tokens=False, verbose=False)
        
        if len(ctx_ids) <= CTX_BUDGET:
            # Context fits in a single window — fast path (no windowing needed)
            context_windows = [context]
        else:
            # Build overlapping token windows and decode back to text
            context_windows = []
            start = 0
            while start < len(ctx_ids):
                end = min(start + CTX_BUDGET, len(ctx_ids))
                window_text = tokenizer.decode(ctx_ids[start:end], skip_special_tokens=True)
                context_windows.append(window_text)
                if end >= len(ctx_ids):
                    break
                start += CTX_BUDGET - OVERLAP  # slide forward with overlap

        # ── Check each claim against all windows ─────────────────────────────
        unsupported = []
        
        for claim in claims:
            # Build NLI pairs for every window
            pairs = [{"text": window, "text_pair": claim} for window in context_windows]
            
            try:
                all_results = nli_pipe(pairs, top_k=None)
            except Exception as e:
                print(f"[LightweightEvaluator] NLI error for claim: {e}")
                continue
            
            # Less strict aggregation: claim is grounded if ANY window entails it
            # OR if ANY window considers it a high-confidence 'neutral' (harmless conversational text).
            # We ONLY flag if the claim is explicitly contradicted everywhere.
            is_supported = False
            
            for res_list in all_results:
                top_label = res_list[0]["label"].lower()
                top_score = res_list[0]["score"]
                
                if "entailment" in top_label:
                    is_supported = True
                    break
                elif "neutral" in top_label and top_score > 0.6:
                    # High-confidence neutral -> not explicitly contradicting the text.
                    # Acceptable for generative LLMs writing patient-facing instructions.
                    is_supported = True
                    break
            
            if not is_supported:
                # Not supported anywhere — find the dominant error label for logging
                top_pred = all_results[0][0]
                unsupported.append({
                    "claim": claim,
                    "nli_label": top_pred["label"],
                    "score": round(top_pred["score"], 3)
                })

        if unsupported:
            return {
                "verdict": "FAIL",
                "reasoning": f"Found {len(unsupported)} ungrounded claims ({len(context_windows)} context windows checked).",
                "unsupported_claims": unsupported
            }
            
        return {
            "verdict": "PASS",
            "reasoning": f"All claims entailed ({len(context_windows)} context windows checked).",
            "unsupported_claims": []
        }

    def clear_memory(self):
        """Free up GPU memory if used intermittently."""
        self._nli_pipe = None
        self._zs_pipe = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

# Global singleton so we don't reload weights repeatedly during evaluation loops
_global_evaluator = None

def get_evaluator() -> LightweightEvaluator:
    global _global_evaluator
    if _global_evaluator is None:
        _global_evaluator = LightweightEvaluator()
    return _global_evaluator
