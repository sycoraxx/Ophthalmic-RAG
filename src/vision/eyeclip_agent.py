"""
eyeclip_agent.py — EyeCLIP Vision Perception Agent (v4)
─────────────────────────────────────────────────────────
Zero-shot ophthalmic image classifier using EyeCLIP (fine-tuned CLIP for
11 ophthalmic imaging modalities).

Architecture (v4 — Visual Anchor Router + Isolated Softmax + Dual Expert):
  Stage 0: Visual Anchor Router
    • Encode text descriptions of how each modality looks (not disease names)
    • Score query image against these visual anchors to detect modality
    • Supports all 11 EyeCLIP native modalities

  Stage 1: Expert Selection
    • Posterior (OCT/CFP/FFA/FAF/ICGA/OUS/RetCam) → Fine-Tuned EyeCLIP
    • Anterior (slit lamp/specular/corneal photography) → Fine-Tuned EyeCLIP
    • External (external eye) → FT EyeCLIP first, Vanilla fallback if low conf

  Stage 2: Isolated Prefix Softmax (with temperature sharpening)
    • Run softmax ONLY on labels matching detected modality prefix
    • Prevents cross-modality softmax suppression (e.g. FFA labels suppressing OCT)
    • Temperature sharpening avoids logits dilution in larger groups

  MedGemma always processes raw images in parallel (separate pipeline).
"""

import os
import torch
from PIL import Image
from . import eyeclip

# ─── Paths ────────────────────────────────────────────────────────────────────
EYECLIP_WEIGHTS  = os.path.join(".", "models", "checkpoints", "eyeclip_visual_new.pt")
CACHED_FEATURES  = os.path.join(".", "data", "cache", "eyeclip_text_features.pt")
VANILLA_FEATURES = os.path.join(".", "data", "cache", "vanilla_text_features.pt")
CLIP_BACKBONE    = "ViT-B/32"

# ─── Classification Thresholds ────────────────────────────────────────────────
PROBABLE_THRESHOLD      = 0.15   # Above this → "probable" finding
POSSIBLE_THRESHOLD      = 0.05   # Between this and PROBABLE → "possible"
LOW_CONF_THRESHOLD      = 0.10   # If top-1 below this, add uncertainty disclaimer
DEFAULT_TOP_K           = 5      # Top-k conditions to return
EXTERNAL_FT_MIN_CONF    = 0.25   # If FT fails on external, fall back to Vanilla
SOFTMAX_TEMPERATURE     = 0.5    # <1 = sharpen logits to prevent dilution

# ─── Modality Probes ──────────────────────────────────────────────────────────
# Short canonical text probes per modality, matching EyeCLIP's training distribution.
# The model was trained with prompts like "optical coherence tomography, drusen" so
# single-modality probes have the best alignment in the FT embedding space.
# Detection uses argmax of raw cosine similarity — NOT softmax — to avoid dilution.
MODALITY_PROBES: dict[str, str] = {
    "OCT":                 "optical coherence tomography",
    "CFP":                 "color fundus photography",
    "FFA":                 "fundus fluorescein angiography",
    "ICGA":                "indocyanine green angiography",
    "FAF":                 "fundus autofluorescence",
    "OUS":                 "ocular ultrasound B-scan",
    "RetCam":              "RetCam pediatric retinal imaging",
    "slit lamp":           "slit lamp anterior segment photography",
    "specular":            "specular microscopy corneal endothelium",
    "corneal photography": "corneal topography Pentacam",
    "external":            "external eye photography eyelid face",
}

# Map each modality prefix to its broad segment type
MODALITY_SEGMENT = {
    "OCT":                  "posterior",
    "CFP":                  "posterior",
    "FFA":                  "posterior",
    "ICGA":                 "posterior",
    "FAF":                  "posterior",
    "OUS":                  "posterior",
    "RetCam":               "posterior",
    "slit lamp":            "anterior",
    "specular":             "anterior",
    "corneal photography":  "anterior",
    "external":             "external",
}

# Human-readable display names
MODALITY_DISPLAY = {
    "OCT":                  "OCT (Optical Coherence Tomography)",
    "CFP":                  "CFP (Color Fundus Photography)",
    "FFA":                  "FFA (Fundus Fluorescein Angiography)",
    "ICGA":                 "ICGA (Indocyanine Green Angiography)",
    "FAF":                  "FAF (Fundus Autofluorescence)",
    "OUS":                  "OUS (Ocular Ultrasound)",
    "RetCam":               "RetCam (Pediatric Retinal Imaging)",
    "slit lamp":            "Slit Lamp Photography",
    "specular":             "Specular Microscopy",
    "corneal photography":  "Corneal Photography / Topography",
    "external":             "External Eye Photography",
}


class EyeClipAgent:
    """
    v4: Visual Anchor Router + Isolated Prefix Softmax + Dual Expert fallback.
    Supports all 11 EyeCLIP native modalities including External Eye Photography.
    MedGemma runs in parallel on raw images (implemented in generator.py).
    """

    def __init__(
        self,
        weights_path: str = EYECLIP_WEIGHTS,
        cache_path: str = CACHED_FEATURES,
        vanilla_cache: str = VANILLA_FEATURES,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models(weights_path)
        self._load_text_features(cache_path, vanilla_cache)

    # ── Model Loading ─────────────────────────────────────────────────────────
    def _load_models(self, weights_path: str):
        """Load Fine-Tuned EyeCLIP (primary) and Vanilla CLIP (external fallback)."""
        print(f"[EyeCLIP] Loading Vanilla {CLIP_BACKBONE} backbone on {self.device}...")
        self.model_vanilla, self.preprocess = eyeclip.load(CLIP_BACKBONE, device=self.device, jit=False)
        self.model_vanilla.float().eval()

        print(f"[EyeCLIP] Loading Fine-Tuned {CLIP_BACKBONE} backbone on {self.device}...")
        self.model_ft, _ = eyeclip.load(CLIP_BACKBONE, device=self.device, jit=False)

        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

            import copy
            test_model = copy.deepcopy(self.model_ft)
            test_model.load_state_dict(state_dict, strict=False)
            test_model.float().eval()

            # Health-check: fine-tuned weights should differ meaningfully from vanilla
            test_prompts = ["a photograph of a red eye", "a photo of a cat"]
            test_tokens = eyeclip.tokenize(test_prompts).to(self.device)
            with torch.no_grad():
                feats = test_model.encode_text(test_tokens)
                feats /= feats.norm(dim=-1, keepdim=True)
                diff = (feats[0] - feats[1]).abs().sum().item()
            del test_model

            if diff > 1.0:
                self.model_ft.load_state_dict(state_dict, strict=False)
                print(f"[EyeCLIP] ✓ Loaded fine-tuned weights (health check passed, diff={diff:.2f}).")
            else:
                print("[EyeCLIP] ⚠ Checkpoint causes mode collapse. Using vanilla weights in FT slot.")
        else:
            print("[EyeCLIP] Using base model for FT slot (no weights found).")

        self.model_ft.float().eval()

    # ── Text Feature Loading ──────────────────────────────────────────────────
    def _load_text_features(self, ft_cache_path: str, vanilla_cache_path: str):
        """Load both fine-tuned and vanilla text embedding caches."""
        self.ft_text_features = None
        self.vanilla_text_features = None
        self.labels: list[str] = []

        has_ft = os.path.exists(ft_cache_path)
        has_vanilla = os.path.exists(vanilla_cache_path)

        if has_ft and has_vanilla:
            print(f"[EyeCLIP] Loading FT cache from {ft_cache_path}")
            ft_payload = torch.load(ft_cache_path, map_location=self.device)
            self.labels = ft_payload["labels"]
            self.ft_text_features = ft_payload["text_features"].to(self.device).float()

            print(f"[EyeCLIP] Loading Vanilla cache from {vanilla_cache_path}")
            vanilla_payload = torch.load(vanilla_cache_path, map_location=self.device)
            self.vanilla_text_features = vanilla_payload["text_features"].to(self.device).float()

            print(f"[EyeCLIP] ✓ Loaded {len(self.labels)} labels (FT + Vanilla).")
        else:
            print("[EyeCLIP] ⚠ Caches not found. Run embed_labels.py first.")
            self.labels = []

    # ── Stage 0: Modality Routing ─────────────────────────────────────────────
    # Modality detection is now handled exclusively upstream by MedGemma VLM
    # (via engine.py -> generator.detect_modality_vlm) which passes a modality_hint.

    # ── Stage 2: Isolated Prefix Softmax ──────────────────────────────────────
    def _classify_prefix(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        prefix: str,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """
        Run softmax ONLY on labels matching `prefix`.
        Temperature sharpening reduces logits dilution from large label groups.
        Returns: list of (clean_condition_name, probability)
        """
        indices = [i for i, lbl in enumerate(self.labels) if lbl.startswith(prefix)]
        if not indices:
            return []

        idx_tensor = torch.tensor(indices, device=image_features.device)
        group_feats = text_features[idx_tensor]

        # Temperature-sharpened logits (smaller T → sharper distribution)
        logits = (100.0 * image_features @ group_feats.T) / SOFTMAX_TEMPERATURE
        group_probs = logits[0].softmax(dim=-1)

        k = min(top_k, len(indices))
        top_probs, top_local_indices = group_probs.topk(k)

        results = []
        for prob, local_idx in zip(top_probs, top_local_indices):
            global_idx = indices[local_idx.item()]
            raw_label = self.labels[global_idx]
            # Strip prefix for clean display: "OCT, drusen" → "drusen"
            clean_cond = raw_label.replace(prefix + ", ", "").replace(prefix + ",", "").strip()
            results.append((clean_cond, prob.item()))

        return results

    # ── Main Entry Point ──────────────────────────────────────────────────────
    def analyze_image(
        self,
        image_path: str,
        top_k: int = DEFAULT_TOP_K,
        modality_hint: str | None = None,
    ) -> str:
        """
        Analyze an ophthalmic image and return structured clinical text findings.

        Args:
            image_path:     Path to image file.
            top_k:          Number of top conditions to return.
            modality_hint:  Optional modality string (e.g. 'OCT', 'slit lamp').
                            If provided, skips the visual anchor router entirely.
                            Should be produced by MedGemma's detect_modality_vlm().
        """
        if not self.labels:
            return "EyeCLIP: No label vocabulary loaded. Please run embed_labels.py."
        if not os.path.exists(image_path):
            return f"EyeCLIP: Image not found at {image_path}"

        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # ── Stage 0: MedGemma VLM Modality Routing ───────────────────
                if modality_hint and modality_hint in MODALITY_SEGMENT:
                    modality = modality_hint
                    router_source = "MedGemma VLM"
                else:
                    modality = "CFP"  # Safe default if VLM fails
                    router_source = "Default Fallback"

                segment = MODALITY_SEGMENT.get(modality, "posterior")

                # ── Stage 1: Encode image with appropriate expert ─────────────
                img_f_ft = self.model_ft.encode_image(image_tensor)
                img_f_ft /= img_f_ft.norm(dim=-1, keepdim=True)

                # ── Stage 2: Isolated prefix softmax ─────────────────────────
                results = self._classify_prefix(
                    img_f_ft, self.ft_text_features, modality, top_k
                )

                # ── External fallback: if FT confidence is low, try Vanilla ───
                if segment == "external" and results:
                    top1_prob = results[0][1]
                    if top1_prob < EXTERNAL_FT_MIN_CONF and self.vanilla_text_features is not None:
                        img_f_v = self.model_vanilla.encode_image(image_tensor)
                        img_f_v /= img_f_v.norm(dim=-1, keepdim=True)
                        results_v = self._classify_prefix(
                            img_f_v, self.vanilla_text_features, modality, top_k
                        )
                        if results_v and results_v[0][1] > top1_prob:
                            results = results_v
                            router_source += " + Vanilla CLIP (external fallback)"

            # ── Build structured output ───────────────────────────────────────
            modality_display = MODALITY_DISPLAY.get(modality, modality)
            segment_display = {
                "posterior": "posterior segment (fundus / retinal imaging)",
                "anterior":  "anterior segment (cornea / lens / iris)",
                "external":  "external eye / adnexa",
            }.get(segment, segment)

            sep = "━" * 46
            lines = [
                "EyeCLIP Image Analysis (v4)",
                sep,
                f"Detected Image Type:  {modality_display}",
                f"Segment:              {segment_display}",
                f"Router:               {router_source}",
                "",
                "Top Findings:",
            ]

            probable, possible = [], []
            for condition, prob in results:
                pct = prob * 100
                if prob >= PROBABLE_THRESHOLD:
                    probable.append(f"  ● Probable: {condition} ({pct:.1f}%)")
                elif prob >= POSSIBLE_THRESHOLD:
                    possible.append(f"  ○ Possible: {condition} ({pct:.1f}%)")

            if probable:
                lines.extend(probable)
            if possible:
                lines.extend(possible)
            if not probable and not possible:
                lines.append("  No conditions detected above reporting threshold.")

            # Normal baseline: probability of the top-ranked result being "normal"
            normal_labels = ["normal", "normal macula", "normal fundus",
                             "normal external eye", "normal anterior segment"]
            normal_score = 0.0
            for cond, prob in results:
                if any(n in cond.lower() for n in normal_labels):
                    normal_score = max(normal_score, prob)

            lines.append("")
            lines.append(f"Normal Baseline:  {normal_score * 100:.1f}%"
                         f"  ({'normal' if normal_score > 0.3 else 'pathology likely present'})")

            top1_prob = results[0][1] if results else 0.0
            if top1_prob < LOW_CONF_THRESHOLD:
                lines.append("")
                lines.append("⚠ Low confidence: model is uncertain. Clinical judgement must take priority.")

            lines.append(sep)
            lines.append("⚕ These are automated screening findings only.")
            lines.append("  They must be confirmed by a qualified ophthalmologist.")

            return "\n".join(lines)

        except Exception as e:
            return f"EyeCLIP analysis failed: {e}"

    # ── Structured Finding Extraction ────────────────────────────────────────
    @staticmethod
    def extract_key_findings(findings_text: str | None) -> dict | None:
        """
        Parse the structured EyeCLIP text output into a compact dict.

        Returns:
            {
                "modality": "OCT",
                "conditions": [
                    {"name": "drusen", "confidence": "probable", "score": 0.452},
                    {"name": "macular edema", "confidence": "possible", "score": 0.08},
                ]
            }
            or None if parsing fails / no findings.
        """
        if not findings_text:
            return None

        import re

        result = {"modality": None, "conditions": []}

        # Extract modality from "Detected Image Type: OCT (...)"
        mod_match = re.search(r"Detected Image Type:\s*(.+?)(?:\n|$)", findings_text)
        if mod_match:
            raw_mod = mod_match.group(1).strip()
            # Extract abbreviation before parenthetical: "OCT (Optical ...)" → "OCT"
            abbrev_match = re.match(r"^(\S+)", raw_mod)
            if abbrev_match:
                result["modality"] = abbrev_match.group(1)

        # Extract conditions: "● Probable: drusen (45.2%)" / "○ Possible: macular edema (8.0%)"
        condition_pattern = re.compile(
            r"[●○]\s*(Probable|Possible):\s*(.+?)\s*\((\d+\.?\d*)%\)"
        )
        for match in condition_pattern.finditer(findings_text):
            confidence = match.group(1).lower()
            name = match.group(2).strip()
            score = float(match.group(3)) / 100.0
            result["conditions"].append({
                "name": name,
                "confidence": confidence,
                "score": score,
            })

        if not result["conditions"]:
            return None

        return result

    @staticmethod
    def get_retrieval_terms(findings_text: str | None) -> str:
        """
        Extract concise clinical keywords from EyeCLIP findings for retrieval.

        Only includes up to 2 high-confidence ('probable') conditions.
        Returns e.g. "OCT drusen macular edema" (Strings contain no percentages).
        """
        parsed = EyeClipAgent.extract_key_findings(findings_text)
        if not parsed:
            return ""

        terms = []

        # Include modality
        if parsed["modality"]:
            terms.append(parsed["modality"])

        # Include up to 2 probable (Top Findings) without their score percentages
        probable_count = 0
        for cond in parsed["conditions"]:
            if cond["confidence"] == "probable":
                terms.append(cond["name"])
                probable_count += 1
                if probable_count >= 2:
                    break

        return " ".join(terms)

    @property
    def is_ready(self) -> bool:
        """True if the agent has model + text features loaded."""
        return (
            self.model_ft is not None
            and self.ft_text_features is not None
            and len(self.labels) > 0
        )
