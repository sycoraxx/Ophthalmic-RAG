# src/state/clinical_entity_extractor.py

from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import re
from pathlib import Path

class EntityType(Enum):
    ANATOMY = "anatomy"
    CONDITION = "condition"
    SYMPTOM = "symptom"
    PROCEDURE = "procedure"
    IMAGING = "imaging"
    MEDICATION = "medication"
    FINDING = "finding"  # NEW: For objective imaging findings (e.g., "drusen", "hemorrhage")

class OphthalmicRegion(Enum):
    ANTERIOR = "anterior_segment"
    POSTERIOR = "posterior_segment"
    ADNEXA = "adnexa_and_orbit"
    SYSTEMIC = "systemic_ocular"
    UNSPECIFIED = "unspecified"

class IndianClinicalPriority(Enum):
    LEVEL_A = "emergency"  # Action within 4-72 hours
    LEVEL_B = "urgent"     # Action within 4 weeks
    LEVEL_C = "elective"   # Action > 6 weeks
    ROUTINE = "routine"

@dataclass
class ClinicalEntity:
    """Represents an extracted clinical entity with confidence and provenance."""
    text: str
    entity_type: EntityType
    confidence: float  # 0.0 - 1.0
    source: str  # "answer", "eyeclip", "user_query", "merged"
    turn_id: int
    normalized: Optional[str] = None  # SNOMED/UMLS normalized form
    modality: Optional[str] = None  # Imaging modality if applicable (e.g., "OCT", "CFP")
    spatial_location: Optional[str] = None  # e.g., "macula", "peripheral retina"
    region: OphthalmicRegion = OphthalmicRegion.UNSPECIFIED
    priority: IndianClinicalPriority = IndianClinicalPriority.ROUTINE
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "confidence": self.confidence,
            "source": self.source,
            "turn_id": self.turn_id,
            "normalized": self.normalized,
            "modality": self.modality,
            "spatial_location": self.spatial_location,
            "region": self.region.value,
            "priority": self.priority.value,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ClinicalEntity":
        return cls(
            text=data["text"],
            entity_type=EntityType(data["entity_type"]),
            confidence=data["confidence"],
            source=data["source"],
            turn_id=data["turn_id"],
            normalized=data.get("normalized"),
            modality=data.get("modality"),
            spatial_location=data.get("spatial_location"),
            region=OphthalmicRegion(data.get("region", "unspecified")),
            priority=IndianClinicalPriority(data.get("priority", "routine")),
        )
    
    def merge_with(self, other: "ClinicalEntity") -> "ClinicalEntity":
        """Merge two entities representing the same concept (weighted confidence)."""
        # Use higher confidence source as base
        base = self if self.confidence >= other.confidence else other
        other_src = other if self.confidence >= other.confidence else self
        
        # Weighted average confidence (favor visual findings for objective signs)
        if base.source == "eyeclip" and other_src.source == "answer":
            # Visual findings get 70% weight for objective signs
            merged_conf = 0.7 * base.confidence + 0.3 * other_src.confidence
        else:
            merged_conf = 0.6 * base.confidence + 0.4 * other_src.confidence
        
        return ClinicalEntity(
            text=base.text,
            entity_type=base.entity_type,
            confidence=min(merged_conf, 1.0),
            source="merged",
            turn_id=max(base.turn_id, other_src.turn_id),
            normalized=base.normalized or other_src.normalized,
            modality=base.modality or other_src.modality,
            spatial_location=base.spatial_location or other_src.spatial_location,
            region=base.region if base.region != OphthalmicRegion.UNSPECIFIED else other_src.region,
            priority=base.priority if base.priority != IndianClinicalPriority.ROUTINE else other_src.priority,
        )


class ClinicalEntityExtractor:
    """
    Extracts clinical entities from text AND EyeCLIP visual findings.
    
    EyeCLIP integration:
    - High-confidence conditions from image analysis are injected as entities
    - Modality detection (OCT/CFP/etc.) mapped to IMAGING entities
    - Spatial findings (e.g., "macular drusen") parsed for location context
    - Conflicts between text and visual findings resolved via confidence weighting
    """
    
    # EyeCLIP condition name → standardized clinical terminology
    EYECLIP_CONDITION_MAP = {
        # Macular/Retinal
        "drusen": {"normalized": "drusen", "type": EntityType.FINDING, "anatomy": "retina"},
        "age-related macular degeneration": {"normalized": "age-related macular degeneration", "type": EntityType.CONDITION, "anatomy": "macula"},
        "amd": {"normalized": "age-related macular degeneration", "type": EntityType.CONDITION, "anatomy": "macula"},
        "diabetic retinopathy": {"normalized": "diabetic retinopathy", "type": EntityType.CONDITION, "anatomy": "retina"},
        "macular edema": {"normalized": "macular edema", "type": EntityType.FINDING, "anatomy": "macula"},
        "retinal hemorrhage": {"normalized": "retinal hemorrhage", "type": EntityType.FINDING, "anatomy": "retina"},
        "retinal detachment": {"normalized": "retinal detachment", "type": EntityType.CONDITION, "anatomy": "retina"},
        "epiretinal membrane": {"normalized": "epiretinal membrane", "type": EntityType.FINDING, "anatomy": "retina"},
        "macular hole": {"normalized": "macular hole", "type": EntityType.FINDING, "anatomy": "macula"},
        
        # Anterior segment
        "cataract": {"normalized": "cataract", "type": EntityType.CONDITION, "anatomy": "lens"},
        "corneal opacity": {"normalized": "corneal opacity", "type": EntityType.FINDING, "anatomy": "cornea"},
        "pterygium": {"normalized": "pterygium", "type": EntityType.CONDITION, "anatomy": "conjunctiva"},
        
        # Optic nerve
        "glaucoma": {"normalized": "glaucoma", "type": EntityType.CONDITION, "anatomy": "optic nerve"},
        "optic disc edema": {"normalized": "optic disc edema", "type": EntityType.FINDING, "anatomy": "optic nerve"},
        
        # Inflammatory
        "uveitis": {"normalized": "uveitis", "type": EntityType.CONDITION, "anatomy": "uvea"},
        "conjunctivitis": {"normalized": "conjunctivitis", "type": EntityType.CONDITION, "anatomy": "conjunctiva"},
    }
    
    # EyeCLIP modality → standardized imaging entity
    EYECLIP_MODALITY_MAP = {
        "OCT": "optical coherence tomography",
        "CFP": "color fundus photography",
        "FFA": "fluorescein angiography",
        "ICGA": "indocyanine green angiography",
        "FAF": "fundus autofluorescence",
        "slit lamp": "slit lamp biomicroscopy",
        "specular": "specular microscopy",
        "RetCam": "widefield retinal imaging",
    }

    # Anatomical region mapping (Indian clinical standards)
    REGION_MAPPINGS = {
        OphthalmicRegion.ANTERIOR: {
            "cornea", "lens", "conjunctiva", "iris", "anterior chamber", 
            "ciliary body", "sclera", "pupil", "angle", "aqueous"
        },
        OphthalmicRegion.POSTERIOR: {
            "retina", "macula", "vitreous", "optic nerve", "choroid", 
            "posterior pole", "fovea", "optic disc", "vasculature"
        },
        OphthalmicRegion.ADNEXA: {
            "eyelid", "lacrimal", "orbit", "extraocular muscle", "nasolacrimal"
        }
    }

    # Clinical priority mapping (Indian clinical standards)
    PRIORITY_MAPPINGS = {
        IndianClinicalPriority.LEVEL_A: {
            "acute angle closure", "trauma", "chemical burn", "perforation",
            "sudden vision loss", "central retinal artery occlusion", "endophthalmitis"
        },
        IndianClinicalPriority.LEVEL_B: {
            "diabetic retinopathy", "retinal detachment", "uveitis",
            "corneal ulcer", "macular edema"
        },
        IndianClinicalPriority.LEVEL_C: {
            "cataract", "pterygium", "refractive error", "dry eye"
        }
    }
    
    # Spatial keywords for location parsing
    SPATIAL_KEYWORDS = {
        "macula": ["macula", "macular", "fovea", "foveal", "central retina"],
        "peripheral_retina": ["peripheral", "mid-peripheral", "far peripheral", "ora serrata"],
        "optic_disc": ["optic disc", "optic nerve head", "ONH", "cup", "rim"],
        "anterior_chamber": ["anterior chamber", "AC", "angle", "trabecular"],
        "lens": ["lens", "crystalline", "anterior capsule", "posterior capsule"],
        "cornea": ["cornea", "corneal", "epithelium", "endothelium", "stroma"],
    }

    SYMPTOM_PATTERNS = {
        "blurry vision": [r"\bblurr?y vision\b", r"\bblurred vision\b", r"\bvision blur\b"],
        "floaters": [r"\bfloaters?\b"],
        "flashes": [r"\bflashes?\b", r"\bphotopsia\b"],
        "photophobia": [r"\bphotophobia\b", r"\blight sensitivity\b"],
        "eye pain": [r"\beye pain\b", r"\bocular pain\b", r"\bpainful eye\b"],
        "redness": [r"\beye redness\b", r"\bred eye\b", r"\bocular redness\b", r"\bredness\b"],
        "watering": [r"\bwatering\b", r"\bwatery eyes?\b", r"\btearing\b", r"\bexcessive tearing\b", r"\bepiphora\b"],
        "dryness": [
            r"\bdry eye\b",
            r"\bdry eyes\b",
            r"\beye dryness\b",
            r"\bocular dryness\b",
            r"\b(eye|eyes)\s+(feel\s+)?dry\b",
            r"\beyes?\s+are\s+dry\b",
        ],
        "itching": [
            r"\bitchy eyes?\b",
            r"\beye itch\w*\b",
            r"\bitching\b",
            r"\bocular itch\w*\b",
            r"\bitchy\b",
        ],
        "burning": [r"\bburning\b", r"\bburning sensation\b", r"\beye burn\w*\b", r"\bocular burn\w*\b"],
        "foreign body sensation": [r"\bforeign body sensation\b", r"\bgritty\b", r"\bsandy sensation\b"],
        "discharge": [r"\beye discharge\b", r"\bdischarge\b", r"\bpus\b", r"\bsticky eyes?\b"],
        "swelling": [r"\beyelid swelling\b", r"\bswollen eyelid\b", r"\beye swelling\b", r"\bswelling\b"],
        "double vision": [r"\bdouble vision\b", r"\bdiplopia\b"],
        "halos": [r"\bhalos?\b", r"\bhaloes?\b"],
        "metamorphopsia": [r"\bmetamorphopsia\b", r"\bdistorted vision\b"],
        "scotoma": [r"\bscotoma\b", r"\bblind spot\b"],
        "white spot": [r"\bwhite\s+(spot|patch|dot|mark|opacity|lesion)s?\b"],
    }

    ANATOMY_PATTERNS = {
        "cornea": [
            r"\bcornea\b",
            r"\bcorneal\b",
            r"\bfront\s+of\s+(the\s+)?eye\b",
            r"\bblack\s+part\s+of\s+(the\s+)?eye\b",
            r"\bspot\s+on\s+(the\s+)?black\s+part\s+of\s+(the\s+)?eye\b",
        ],
        "conjunctiva": [r"\bconjunctiva\b", r"\bwhite\s+of\s+(the\s+)?eye\b"],
    }

    FINDING_PATTERNS = {
        "drusen": [r"\bdrusen\b"],
        "macular edema": [r"\bmacular edema\b", r"\bcystoid macular edema\b"],
        "retinal hemorrhage": [r"\bretinal hemorrhage\b", r"\bhemorrhages?\b", r"\bhaemorrhages?\b"],
        "hard exudates": [r"\bhard exudates?\b", r"\bexudates?\b"],
        "roth spots": [r"\broth'?s?\s+spots?\b", r"\broth\s+spots?\b"],
        "cotton wool spots": [r"\bcotton\s+wool\s+spots?\b"],
        "neovascularization": [r"\bneovasculari[sz]ation\b", r"\bnew vessels\b"],
        "optic disc edema": [r"\boptic disc edema\b", r"\bpapilledema\b"],
    }

    FINDING_TO_SYMPTOM_BRIDGE = {
        "roth spot": "roth spots",
        "roth spots": "roth spots",
        "roth's spot": "roth spots",
        "roth's spots": "roth spots",
        "white spot": "white spot",
        "white spots": "white spots",
        "cotton wool spots": "cotton wool spots",
    }

    SYMPTOM_HINT_KEYWORDS = {
        "pain", "redness", "itch", "watering", "watery", "tearing", "dry", "blur",
        "blurry", "vision", "photophobia", "floaters", "flashes", "diplopia", "discharge",
        "burning", "foreign body", "irritation", "sensitivity", "headache", "halos"
    }

    FINDING_HINT_KEYWORDS = {
        "spot", "spots", "opacity", "opacities", "hemorrhage", "haemorrhage", "exudate",
        "edema", "oedema", "lesion", "ulcer", "infiltrate", "deposit", "detachment",
        "drusen", "neovascular", "scar", "atrophy", "membrane", "hole", "hypopyon",
        "papilledema", "cotton wool", "roth"
    }

    MEDICATION_PATTERNS = {
        "anti-VEGF": [r"\banti\s*-?\s*vegf\b"],
        "bevacizumab": [r"\bbevacizumab\b", r"\bavastin\b"],
        "ranibizumab": [r"\branibizumab\b", r"\blucentis\b"],
        "aflibercept": [r"\baflibercept\b", r"\beylea\b"],
        "faricimab": [r"\bfaricimab\b", r"\bvabysmo\b"],
        "timolol": [r"\btimolol\b"],
        "latanoprost": [r"\blatanoprost\b"],
        "brimonidine": [r"\bbrimonidine\b"],
        "dorzolamide": [r"\bdorzolamide\b"],
        "acetazolamide": [r"\bacetazolamide\b"],
        "prednisolone": [r"\bprednisolone\b"],
        "dexamethasone": [r"\bdexamethasone\b"],
        "loteprednol": [r"\bloteprednol\b"],
        "artificial tears": [r"\bartificial tears\b", r"\blubricating drops?\b"],
        "cyclosporine eye drops": [r"\bcyclosporine\b"],
        "moxifloxacin": [r"\bmoxifloxacin\b"],
    }

    PROCEDURE_PATTERNS = {
        "intravitreal injection": [r"\bintravitreal injections?\b", r"\bintravitreal anti\s*-?\s*vegf\b"],
        "phacoemulsification": [r"\bphaco(emulsification)?\b"],
        "cataract surgery": [r"\bcataract surgery\b"],
        "vitrectomy": [r"\bvitrectomy\b"],
        "laser photocoagulation": [r"\blaser photocoagulation\b", r"\bPRP\b", r"\bpanretinal photocoagulation\b"],
        "trabeculectomy": [r"\btrabeculectomy\b"],
        "YAG capsulotomy": [r"\bYAG capsulotomy\b", r"\bposterior capsulotomy\b"],
    }

    # Dedicated ophthalmic lexicon seed (runtime fallback when no custom file is provided).
    # This avoids direct dependency on general disease corpora.
    DEFAULT_OPHTHALMIC_LEXICON: Dict[str, List[str]] = {
        "condition": [
            "age-related macular degeneration",
            "diabetic retinopathy",
            "central serous chorioretinopathy",
            "retinal vein occlusion",
            "retinal artery occlusion",
            "retinitis pigmentosa",
            "retinopathy of prematurity",
            "coats disease",
            "best vitelliform macular dystrophy",
            "stargardt disease",
            "vogt-koyanagi-harada disease",
            "behcet uveitis",
            "fuchs endothelial corneal dystrophy",
            "keratoconus",
            "pseudoexfoliation glaucoma",
            "primary open angle glaucoma",
            "primary angle closure glaucoma",
            "neovascular glaucoma",
            "optic neuritis",
            "papillophlebitis",
        ],
        "finding": [
            "roth spots",
            "cotton wool spots",
            "hard exudates",
            "soft exudates",
            "drusen",
            "dot blot hemorrhage",
            "flame hemorrhage",
            "cherry red spot",
            "macular star",
            "hollenhorst plaque",
            "keratic precipitates",
            "cells and flare",
            "anterior chamber reaction",
            "snowbanking",
            "iris bombe",
            "hypopyon",
            "hyphema",
            "vitritis",
            "disc pallor",
            "optic disc edema",
            "cup disc ratio asymmetry",
            "kayser fleischer ring",
            "arlt line",
            "bitot spots",
            "corneal infiltrate",
            "epiretinal membrane",
            "macular hole",
            "retinal detachment",
        ],
        "symptom": [
            "redness",
            "watering",
            "foreign body sensation",
            "photophobia",
            "blurry vision",
            "sudden vision loss",
            "distorted vision",
            "double vision",
            "floaters",
            "flashes",
            "halos",
            "night blindness",
            "central scotoma",
            "eye pain",
            "headache",
            "white spot",
            "white spots",
            "roth spots",
        ],
    }
    
    def __init__(self, medgemma_generator):
        self.generator = medgemma_generator
        self._entity_templates = self._load_entity_templates()
        self._ophthalmic_lexicon = self._load_ophthalmic_lexicon()
        self._ner_nlp = None
        self._ner_backend = "none"
        self._medical_ner_enabled = os.getenv("LVP_MEDICAL_NER", "1").strip() not in {"0", "false", "False"}
        if self._medical_ner_enabled:
            self._init_medical_ner()
    
    def _load_entity_templates(self) -> Dict[str, List[str]]:
        """Pre-defined entity templates for few-shot prompting and fallback."""
        return {
            "anatomy": [
                "retina", "macula", "fovea", "optic nerve", "optic disc",
                "lens", "cornea", "iris", "ciliary body", "choroid",
                "vitreous", "sclera", "conjunctiva", "eyelid", "retinal pigment epithelium", "RPE"
            ],
            "condition": list(set(v["normalized"] for v in self.EYECLIP_CONDITION_MAP.values() 
                                 if v["type"] == EntityType.CONDITION)),
            "finding": list(set(v["normalized"] for v in self.EYECLIP_CONDITION_MAP.values() 
                               if v["type"] == EntityType.FINDING)),
            "symptom": [
                "blurry vision", "floaters", "flashes", "photophobia",
                "eye pain", "redness", "dryness", "distortion",
                "metamorphopsia", "scotoma", "double vision", "halos",
                "white spot", "roth spots", "cotton wool spots"
            ],
            "imaging": list(self.EYECLIP_MODALITY_MAP.values()),
            "medication": sorted(self.MEDICATION_PATTERNS.keys()),
            "procedure": sorted(self.PROCEDURE_PATTERNS.keys()),
        }

    def _load_ophthalmic_lexicon(self) -> Dict[str, List[str]]:
        """Load dedicated ophthalmic lexicon from repo-owned JSON file.

        Runtime intentionally avoids dependency on broad pre-existing disease corpora.
        """
        project_root = Path(__file__).resolve().parents[2]
        max_terms = int(os.getenv("LVP_OPHTHALMIC_LEXICON_MAX_TERMS", "5000"))
        lexicon_path = os.getenv(
            "LVP_OPHTHALMIC_LEXICON_PATH",
            str(project_root / "data" / "knowledge_base" / "ophthalmic_lexicon.json"),
        )

        lexicon: Dict[str, Set[str]] = {
            "condition": set(self.DEFAULT_OPHTHALMIC_LEXICON.get("condition", [])),
            "finding": set(self.DEFAULT_OPHTHALMIC_LEXICON.get("finding", [])),
            "symptom": set(self.DEFAULT_OPHTHALMIC_LEXICON.get("symptom", [])),
        }

        def normalize_term(raw: str) -> Optional[str]:
            term = re.sub(r"\s+", " ", (raw or "").strip().lower())
            term = re.sub(r"^[\-•*\d\.)\(\s]+", "", term)
            term = term.strip(" ,;:")
            if not term or len(term) < 3 or len(term) > 80:
                return None
            return term

        def add_term(category: str, raw: str):
            norm = normalize_term(raw)
            if not norm:
                return
            if category not in lexicon:
                return
            lexicon[category].add(norm)

        path_obj = Path(lexicon_path)
        if path_obj.exists():
            try:
                payload = json.loads(path_obj.read_text(encoding="utf-8"))
                for category in ("condition", "finding", "symptom"):
                    values = payload.get(category, []) if isinstance(payload, dict) else []
                    if isinstance(values, list):
                        for item in values:
                            if isinstance(item, str):
                                add_term(category, item)
            except Exception as exc:
                print(f"[EntityExtractor] Failed to load ophthalmic lexicon at {path_obj}: {exc}")

        def ordered(values: Set[str]) -> List[str]:
            items = sorted(values, key=lambda x: (len(x.split()) > 8, len(x), x))
            return items[:max_terms]

        return {
            "condition": ordered(lexicon["condition"]),
            "finding": ordered(lexicon["finding"]),
            "symptom": ordered(lexicon["symptom"]),
        }
    
    def extract_entities(
        self,
        text: str,
        visual_findings: Optional[str] = None,
        turn_id: int = 0,
        source: str = "answer",
    ) -> List[ClinicalEntity]:
        """
        Extract clinical entities from text AND EyeCLIP visual findings.
        
        Args:
            text: The text to extract from (answer, query, or findings)
            visual_findings: EyeCLIP output string (optional)
            turn_id: Conversation turn number for tracking
        
        Returns:
            List of ClinicalEntity objects with confidence scores and provenance
        """
        entities: List[ClinicalEntity] = []
        
        # ── Step 1: Extract from EyeCLIP visual findings (HIGH CONFIDENCE) ──
        if visual_findings:
            eyeclip_entities = self._extract_from_eyeclip(visual_findings, turn_id)
            entities.extend(eyeclip_entities)

        # ── Step 2: Medical NER extraction (medspaCy/spaCy, if available) ─
        ner_entities = self._extract_from_medical_ner(text, turn_id, source=source)

        # ── Step 2: Deterministic medical pattern extraction ───────────────
        rule_entities = self._extract_from_text_rules(text, turn_id, source=source)
        
        # ── Step 3: Extract from text using LLM ─────────────────────────────
        text_entities = self._extract_from_text_llm(
            text,
            turn_id,
            visual_findings,
            source=source,
        )
        
        # ── Step 4: Merge entities with confidence weighting ────────────────
        merged_entities = self._merge_entities(entities, ner_entities + rule_entities + text_entities)
        
        # ── Step 5: Filter low-confidence and negated entities ──────────────
        filtered = [
            e for e in merged_entities 
            if e.confidence >= 0.5 and not self._is_negated(text, e.text)
        ]

        # Bridge clinically salient findings to symptom memory terms where useful.
        bridged = self._inject_symptom_bridges(filtered, turn_id)
        
        # ── Step 6: Tag with Locality and Priority metadata ─────────────
        final_entities = self._tag_locality_and_priority(bridged)
        
        return final_entities

    def _init_medical_ner(self):
        """Initialize optional medical NER backend with graceful fallback.

        Priority:
        1) medspaCy pipeline (if installed)
        2) spaCy model (SciSpaCy/standard model, if available)
        3) spaCy blank pipeline + ophthalmic EntityRuler (light fallback)
        """
        # Attempt medspaCy first
        try:
            import medspacy  # type: ignore

            # PyRuSH can emit very verbose debug logs by default.
            try:
                from loguru import logger as loguru_logger  # type: ignore
                loguru_logger.disable("PyRuSH")
            except Exception:
                pass

            self._ner_nlp = medspacy.load()
            self._ner_backend = "medspacy"
            self._ensure_entity_ruler(self._ner_nlp)
            return
        except Exception:
            pass

        # Fallback to spaCy family
        try:
            import spacy  # type: ignore

            # Try common biomedical / general English models in order
            candidate_models = [
                "en_core_sci_lg",
                "en_core_sci_md",
                "en_core_sci_sm",
                "en_core_web_trf",
                "en_core_web_sm",
            ]

            nlp = None
            for model_name in candidate_models:
                try:
                    nlp = spacy.load(model_name)
                    self._ner_backend = f"spacy:{model_name}"
                    break
                except Exception:
                    continue

            if nlp is None:
                # Last resort: blank pipeline + EntityRuler dictionary terms
                nlp = spacy.blank("en")
                if "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
                self._ner_backend = "spacy:blank+ruler"

            self._ner_nlp = nlp
            self._ensure_entity_ruler(self._ner_nlp)
        except Exception:
            self._ner_nlp = None
            self._ner_backend = "none"

    def _ensure_entity_ruler(self, nlp):
        """Inject ophthalmic lexicon rules so NER captures domain terms reliably."""
        try:
            if "entity_ruler" in nlp.pipe_names:
                ruler = nlp.get_pipe("entity_ruler")
            else:
                if "ner" in nlp.pipe_names:
                    ruler = nlp.add_pipe("entity_ruler", before="ner")
                else:
                    ruler = nlp.add_pipe("entity_ruler")

            patterns = []

            def add_patterns(label: str, values: List[str]):
                for value in values:
                    v = value.strip()
                    if not v:
                        continue
                    patterns.append({"label": label, "pattern": v})

            add_patterns("ANATOMY", self._entity_templates["anatomy"])
            add_patterns("CONDITION", self._entity_templates["condition"])
            add_patterns("FINDING", self._entity_templates["finding"])
            add_patterns("SYMPTOM", self._entity_templates["symptom"])

            # Dedicated ophthalmic lexicon terms (repo-owned JSON)
            add_patterns("CONDITION", self._ophthalmic_lexicon.get("condition", []))
            add_patterns("FINDING", self._ophthalmic_lexicon.get("finding", []))
            add_patterns("SYMPTOM", self._ophthalmic_lexicon.get("symptom", []))

            add_patterns("IMAGING", self._entity_templates["imaging"])
            add_patterns("MEDICATION", list(self.MEDICATION_PATTERNS.keys()))
            add_patterns("PROCEDURE", list(self.PROCEDURE_PATTERNS.keys()))

            if patterns:
                ruler.add_patterns(patterns)
        except Exception:
            # Keep extraction pipeline resilient even if ruler setup fails.
            return

    def _map_ner_label(self, label: str, entity_text: str) -> Optional[EntityType]:
        """Map NER labels from medspaCy/spaCy models to internal EntityType."""
        label_norm = (label or "").upper()

        label_map = {
            "ANATOMY": EntityType.ANATOMY,
            "ANATOMICAL_SITE": EntityType.ANATOMY,
            "BODY_PART": EntityType.ANATOMY,
            "CONDITION": EntityType.CONDITION,
            "DISEASE": EntityType.CONDITION,
            "DISORDER": EntityType.CONDITION,
            "DIAGNOSIS": EntityType.CONDITION,
            "PROBLEM": EntityType.CONDITION,
            "FINDING": EntityType.FINDING,
            "SIGN": EntityType.FINDING,
            "SYMPTOM": EntityType.SYMPTOM,
            "SIGN_SYMPTOM": EntityType.SYMPTOM,
            "IMAGING": EntityType.IMAGING,
            "TEST": EntityType.IMAGING,
            "MEDICATION": EntityType.MEDICATION,
            "DRUG": EntityType.MEDICATION,
            "CHEMICAL": EntityType.MEDICATION,
            "PROCEDURE": EntityType.PROCEDURE,
            "TREATMENT": EntityType.PROCEDURE,
            "THERAPY": EntityType.PROCEDURE,
        }

        if label_norm in label_map:
            return label_map[label_norm]

        # Heuristic fallback for free-form labels
        txt = entity_text.lower().strip()
        if any(re.search(p, txt, re.I) for pats in self.MEDICATION_PATTERNS.values() for p in pats):
            return EntityType.MEDICATION
        if any(re.search(p, txt, re.I) for pats in self.PROCEDURE_PATTERNS.values() for p in pats):
            return EntityType.PROCEDURE
        if txt in self._entity_templates["anatomy"]:
            return EntityType.ANATOMY
        if txt in self._entity_templates["condition"]:
            return EntityType.CONDITION
        if txt in self._entity_templates["finding"]:
            return EntityType.FINDING
        if txt in self._entity_templates["symptom"]:
            return EntityType.SYMPTOM

        return None

    def _extract_from_medical_ner(
        self,
        text: str,
        turn_id: int,
        source: str = "answer",
    ) -> List[ClinicalEntity]:
        """Extract entities from optional medical NER backend (medspaCy/spaCy)."""
        if not self._ner_nlp or not text.strip():
            return []

        try:
            doc = self._ner_nlp(text)
        except Exception:
            return []

        extracted: Dict[tuple, ClinicalEntity] = {}
        base_conf = 0.86 if self._ner_backend.startswith("medspacy") else 0.8

        for ent in getattr(doc, "ents", []):
            ent_text = ent.text.strip()
            if len(ent_text) < 3:
                continue

            entity_type = self._map_ner_label(ent.label_, ent_text)
            if entity_type is None:
                continue

            normalized = self._normalize_entity(ent_text, entity_type)
            key = (normalized, entity_type)
            confidence = base_conf

            # Short ontology terms are often ambiguous; mildly downweight.
            if len(ent_text.split()) == 1 and len(ent_text) <= 4:
                confidence -= 0.06

            existing = extracted.get(key)
            if existing is None or confidence > existing.confidence:
                extracted[key] = ClinicalEntity(
                    text=ent_text,
                    entity_type=entity_type,
                    confidence=max(0.5, min(confidence, 0.95)),
                    source=f"{source}_ner",
                    turn_id=turn_id,
                    normalized=normalized,
                )

        return list(extracted.values())

    def _extract_from_text_rules(
        self,
        text: str,
        turn_id: int,
        source: str = "answer",
    ) -> List[ClinicalEntity]:
        """Deterministic extraction for high-value entities to stabilize session state."""
        entities: Dict[tuple, ClinicalEntity] = {}
        text_lower = text.lower()

        def add_entity(normalized_text: str, entity_type: EntityType, confidence: float):
            key = (normalized_text.lower(), entity_type)
            if key in entities:
                entities[key].confidence = max(entities[key].confidence, confidence)
                return
            entities[key] = ClinicalEntity(
                text=normalized_text,
                entity_type=entity_type,
                confidence=confidence,
                source=source,
                turn_id=turn_id,
                normalized=self._normalize_entity(normalized_text, entity_type),
            )

        # Symptoms / findings / medications / procedures from explicit patterns
        for normalized, patterns in self.SYMPTOM_PATTERNS.items():
            if any(re.search(pat, text_lower, re.I) for pat in patterns):
                add_entity(normalized, EntityType.SYMPTOM, 0.8)

        for normalized, patterns in self.FINDING_PATTERNS.items():
            if any(re.search(pat, text_lower, re.I) for pat in patterns):
                add_entity(normalized, EntityType.FINDING, 0.8)

        # Bridge selected findings into symptom-like memory terms for patient-described signs.
        for finding_term, symptom_term in self.FINDING_TO_SYMPTOM_BRIDGE.items():
            if finding_term in text_lower:
                add_entity(symptom_term, EntityType.SYMPTOM, 0.76)

        for normalized, patterns in self.MEDICATION_PATTERNS.items():
            if any(re.search(pat, text_lower, re.I) for pat in patterns):
                add_entity(normalized, EntityType.MEDICATION, 0.82)

        for normalized, patterns in self.PROCEDURE_PATTERNS.items():
            if any(re.search(pat, text_lower, re.I) for pat in patterns):
                add_entity(normalized, EntityType.PROCEDURE, 0.82)

        # Conditions/anatomy/imaging from templates
        for keyword in self._entity_templates["condition"]:
            if re.search(rf"\b{re.escape(keyword.lower())}\b", text_lower):
                add_entity(keyword, EntityType.CONDITION, 0.75)

        for normalized, patterns in self.ANATOMY_PATTERNS.items():
            if any(re.search(pat, text_lower, re.I) for pat in patterns):
                add_entity(normalized, EntityType.ANATOMY, 0.74)

        for keyword in self._entity_templates["anatomy"]:
            if re.search(rf"\b{re.escape(keyword.lower())}\b", text_lower):
                add_entity(keyword, EntityType.ANATOMY, 0.72)

        for keyword in self._entity_templates["imaging"]:
            if re.search(rf"\b{re.escape(keyword.lower())}\b", text_lower):
                add_entity(keyword, EntityType.IMAGING, 0.8)

        return list(entities.values())

    def _inject_symptom_bridges(self, entities: List[ClinicalEntity], turn_id: int) -> List[ClinicalEntity]:
        """Add symptom-side aliases for key visible findings (e.g., Roth spots)."""
        merged = list(entities)
        existing_keys = {
            ((e.normalized or e.text.lower().strip()), e.entity_type)
            for e in merged
        }

        for entity in entities:
            if entity.entity_type != EntityType.FINDING:
                continue

            normalized = (entity.normalized or entity.text).lower().strip()
            symptom_alias = None

            for finding_key, symptom_value in self.FINDING_TO_SYMPTOM_BRIDGE.items():
                if finding_key in normalized:
                    symptom_alias = symptom_value
                    break

            if not symptom_alias:
                continue

            symptom_norm = self._normalize_entity(symptom_alias, EntityType.SYMPTOM)
            key = (symptom_norm, EntityType.SYMPTOM)
            if key in existing_keys:
                continue

            merged.append(
                ClinicalEntity(
                    text=symptom_alias,
                    entity_type=EntityType.SYMPTOM,
                    confidence=max(0.78, min(0.88, entity.confidence * 0.95)),
                    source=f"{entity.source}_bridge",
                    turn_id=turn_id,
                    normalized=symptom_norm,
                    region=entity.region,
                    priority=entity.priority,
                )
            )
            existing_keys.add(key)

        return merged
    
    def _tag_locality_and_priority(self, entities: List[ClinicalEntity]) -> List[ClinicalEntity]:
        """Tag entities with anatomical region and clinical priority."""
        for entity in entities:
            text = entity.text.lower()
            
            # Identify Region
            for region, keywords in self.REGION_MAPPINGS.items():
                if any(kw in text for kw in keywords):
                    entity.region = region
                    break
            
            # If still unspecified, check parent anatomy in EYECLIP_CONDITION_MAP
            if entity.region == OphthalmicRegion.UNSPECIFIED:
                mapped = self.EYECLIP_CONDITION_MAP.get(text)
                if mapped and "anatomy" in mapped:
                    parent_anatomy = mapped["anatomy"]
                    for region, keywords in self.REGION_MAPPINGS.items():
                        if any(kw in parent_anatomy for kw in keywords):
                            entity.region = region
                            break

            # Identify Priority
            for priority, keywords in self.PRIORITY_MAPPINGS.items():
                if any(kw in text for kw in keywords):
                    entity.priority = priority
                    break
        
        return entities
    
    def _extract_from_eyeclip(
        self, 
        visual_findings: str, 
        turn_id: int
    ) -> List[ClinicalEntity]:
        """
        Parse EyeCLIP output into structured ClinicalEntity objects.
        
        EyeCLIP format example:
        "Detected Image Type: OCT (Optical Coherence Tomography)
         Top Findings: 
         ● Probable: drusen (69.0%) 
         ● Probable: central serous chorioretinopathy (24.3%)
         Normal Baseline: 0.0%"
        """
        entities = []
        
        # Parse modality
        modality_match = re.search(r'Detected Image Type:\s*([^\n\r]+)', visual_findings, re.I)
        detected_modality = None
        if modality_match:
            modality_label = modality_match.group(1).strip()
            normalized_modality = modality_label.lower()
            for key, mapped in self.EYECLIP_MODALITY_MAP.items():
                if key.lower() in modality_label.lower() or mapped.lower() in modality_label.lower():
                    normalized_modality = mapped
                    break
            detected_modality = normalized_modality
            entities.append(ClinicalEntity(
                text=normalized_modality,
                entity_type=EntityType.IMAGING,
                confidence=0.95,  # High confidence for modality detection
                source="eyeclip",
                turn_id=turn_id,
            ))
        
        # Parse conditions/findings with confidence scores
        # Pattern: "● Probable: drusen (69.0%)" or "● Possible: hemorrhage (31.2%)"
        finding_pattern = r'[●○\-*]\s*(Probable|Possible|Detected):\s*([^(\n]+?)\s*\(([\d.]+)%\)'
        for match in re.finditer(finding_pattern, visual_findings, re.I):
            confidence_label = match.group(1).lower()
            condition_text = match.group(2).strip().lower()
            confidence_score = float(match.group(3)) / 100.0
            
            # Map confidence label to numeric adjustment
            # NOTE: EyeCLIP percentages are inherently low (5-20% typical) but still accurate.
            # Adjustments are kept minimal to avoid filtering out valid findings.
            label_adjustment = {
                "probable": 0.10,  # Slight boost for higher-confidence label
                "possible": -0.02, # Minimal penalty — the raw % already reflects uncertainty
                "detected": 0.0,   # Neutral
            }
            adjusted_confidence = min(confidence_score + label_adjustment.get(confidence_label, 0), 1.0)
            
            # Skip only noise-level findings (below 3%)
            if adjusted_confidence < 0.03:
                continue
            
            # Look up standardized mapping
            mapping = self.EYECLIP_CONDITION_MAP.get(condition_text)
            if mapping:
                entity = ClinicalEntity(
                    text=mapping["normalized"],
                    entity_type=mapping["type"],
                    confidence=adjusted_confidence,
                    source="eyeclip",
                    turn_id=turn_id,
                    normalized=mapping["normalized"],
                    modality=detected_modality,
                    spatial_location=mapping.get("anatomy"),  # Use anatomy as spatial hint
                )
                entities.append(entity)
            else:
                # Unmapped condition: create generic entity
                entity = ClinicalEntity(
                    text=condition_text,
                    entity_type=EntityType.FINDING,  # Default to objective finding
                    confidence=adjusted_confidence,
                    source="eyeclip",
                    turn_id=turn_id,
                    modality=detected_modality,
                )
                entities.append(entity)
        
        # Parse spatial location hints from finding descriptions
        # e.g., "macular drusen" → spatial_location="macula"
        for entity in entities:
            if entity.entity_type in [EntityType.FINDING, EntityType.CONDITION]:
                for location, keywords in self.SPATIAL_KEYWORDS.items():
                    if any(kw in entity.text.lower() for kw in keywords):
                        entity.spatial_location = location
                        break
        
        return entities
    
    def _extract_from_text_llm(
        self, 
        text: str, 
        turn_id: int,
        visual_findings: Optional[str] = None,
        source: str = "answer",
    ) -> List[ClinicalEntity]:
        """Extract entities from text using MedGemma with structured output."""
        
        # Build EyeCLIP context hint for the LLM (helps resolve conflicts)
        eyeclip_context = ""
        if visual_findings:
            # Extract high-confidence conditions to inject as context
            high_conf = []
            for match in re.finditer(r'●\s*Probable:\s*([^(]+)', visual_findings, re.I):
                condition = match.group(1).strip().lower()
                if condition in self.EYECLIP_CONDITION_MAP:
                    high_conf.append(self.EYECLIP_CONDITION_MAP[condition]["normalized"])
            if high_conf:
                eyeclip_context = f"\n\nNOTE: Automated image analysis detected: {', '.join(high_conf)}. " \
                                 f"If the text contradicts these findings, note the discrepancy."
        
        system_prompt = (
            "You are a clinical entity extraction system for ophthalmology.\n"
            "Extract medical entities from the text and output ONLY valid JSON.\n\n"
            "ENTITY TYPES:\n"
            "- anatomy: Eye structures (retina, lens, cornea, optic nerve, etc.)\n"
            "- condition: Diseases/diagnoses (AMD, drusen, glaucoma, cataract, etc.)\n"
            "- finding: Objective signs observed on exam/imaging (hemorrhage, edema, drusen, etc.)\n"
            "- symptom: Patient complaints (blurry vision, floaters, pain, etc.)\n"
            "- imaging: Tests/modalities (OCT, fundus photo, FFA, etc.)\n"
            "- medication: Drugs/treatments (anti-VEGF, steroids, etc.)\n\n"
            "- procedure: Procedures/interventions (intravitreal injection, vitrectomy, laser, surgery)\n\n"
            "RULES:\n"
            "1. Output ONLY a JSON array. No explanations, no markdown.\n"
            "2. Each entity must have: text, entity_type, confidence (0.0-1.0)\n"
            "3. Confidence guidelines:\n"
            "   - 0.9-1.0: Explicitly stated diagnosis/findings\n"
            "   - 0.7-0.9: Strongly implied or probable\n"
            "   - 0.5-0.7: Mentioned as possibility/differential\n"
            "   - <0.5: Uncertain or negated (exclude these)\n"
            "4. Normalize entity text to standard medical terminology.\n"
            "5. Exclude negated entities (e.g., 'no cataract' → don't extract cataract)\n"
            "6. Merge synonyms (e.g., 'AMD' → 'age-related macular degeneration')\n"
            "7. If IMAGE FINDINGS are provided in the NOTE above, prioritize them for "
            "   objective signs (findings), but still extract symptoms/conditions from text.\n\n"
            "OUTPUT FORMAT:\n"
            "[\n"
            '  {"text": "drusen", "entity_type": "finding", "confidence": 0.95},\n'
            '  {"text": "retina", "entity_type": "anatomy", "confidence": 0.9}\n'
            "]"
        )
        
        user_message = f"Extract clinical entities from this text:{eyeclip_context}\n\n{text}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        try:
            response = self.generator._generate(
                messages,
                max_new_tokens=500,
                temperature=0.0,  # Deterministic for extraction
                do_sample=False,
            )
            
            # Parse JSON using regex to find the array block
            response = response.strip()
            
            # Find JSON payload robustly: prefer first '[' to last ']'.
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end + 1]
            else:
                # If no clear array, maybe it output a single object
                match = re.search(r'\{.*?\}', response, re.DOTALL)
                if match:
                    json_str = f"[{match.group(0)}]"
                else:
                    raise ValueError(f"Could not find JSON structure in output:\n{response}")
            
            try:
                entities_data = json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback to ast.literal_eval if it generated malformed python dicts instead of strict JSON
                import ast
                try:
                    entities_data = ast.literal_eval(json_str)
                except Exception as e:
                    raise ValueError(f"Failed to parse extracted text as JSON or Dict: {e}\n{json_str}")
                
            if not isinstance(entities_data, list):
                entities_data = [entities_data]
            
            # Convert to ClinicalEntity objects
            entities = []
            for item in entities_data:
                if not isinstance(item, dict):
                    continue
                if "text" not in item:
                    continue
                if item.get("confidence", 0) < 0.5:
                    continue
                
                # Map invalid/hallucinated types from LLM
                raw_type = item.get("entity_type", "finding").lower()
                type_map = {
                    "treatment": EntityType.MEDICATION,
                    "drug": EntityType.MEDICATION,
                    "surgery": EntityType.PROCEDURE,
                    "test": EntityType.IMAGING,
                    "sign": EntityType.FINDING,
                }
                
                try:
                    if raw_type in type_map:
                        entity_type = type_map[raw_type]
                    else:
                        entity_type = EntityType(raw_type)
                except ValueError:
                    # Fallback for completely unknown strings
                    entity_type = EntityType.FINDING
                
                entity = ClinicalEntity(
                    text=item["text"],
                    entity_type=entity_type,
                    confidence=item["confidence"],
                    source=source,
                    turn_id=turn_id,
                    normalized=self._normalize_entity(item["text"], entity_type),
                )
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            print(f"[EntityExtractor] ⚠ LLM extraction failed: {e}")
            # Fallback to template matching
            return self._fallback_template_extraction(text, turn_id)
    
    def _merge_entities(
        self, 
        eyeclip_entities: List[ClinicalEntity], 
        text_entities: List[ClinicalEntity]
    ) -> List[ClinicalEntity]:
        """
        Merge entities from EyeCLIP and text extraction with confidence weighting.
        
        Rules:
        - Same normalized text + same type → merge with weighted confidence
        - EyeCLIP findings get priority for objective signs (hemorrhage, drusen, etc.)
        - Text gets priority for symptoms and patient-reported history
        - Conflicting anatomy assignments → use EyeCLIP if confidence > 0.7
        """
        # Index entities by (normalized_text, entity_type) for merging
        entity_index: Dict[tuple, ClinicalEntity] = {}
        
        # Add EyeCLIP entities first (higher priority for objective findings)
        for entity in eyeclip_entities:
            key = (entity.normalized or entity.text.lower(), entity.entity_type)
            entity_index[key] = entity
        
        # Merge text entities
        for entity in text_entities:
            key = (entity.normalized or entity.text.lower(), entity.entity_type)
            
            if key in entity_index:
                # Merge with existing EyeCLIP entity
                existing = entity_index[key]
                
                # Special handling: symptoms from text override EyeCLIP (images don't report symptoms)
                if entity.entity_type == EntityType.SYMPTOM:
                    entity_index[key] = entity  # Text wins for symptoms
                # Objective findings: EyeCLIP wins if high confidence
                elif entity.entity_type == EntityType.FINDING and existing.confidence > 0.7:
                    # Keep EyeCLIP entity but boost confidence slightly if text agrees
                    if existing.confidence < 1.0:
                        existing.confidence = min(existing.confidence + 0.05, 1.0)
                else:
                    # General merge with weighted average
                    entity_index[key] = existing.merge_with(entity)
            else:
                # New entity from text
                entity_index[key] = entity
        
        return list(entity_index.values())
    
    def _is_negated(self, text: str, entity_text: str) -> bool:
        """Detect if an entity is negated in the source text."""
        text_lower = text.lower()
        entity_lower = entity_text.lower()
        
        negation_patterns = [
            f"no {entity_lower}",
            f"without {entity_lower}",
            f"absent {entity_lower}",
            f"negative for {entity_lower}",
            f"rules out {entity_lower}",
            f"exclude {entity_lower}",
            f"no evidence of {entity_lower}",
            f"not consistent with {entity_lower}",
        ]
        
        return any(pattern in text_lower for pattern in negation_patterns)
    
    def _normalize_entity(self, text: str, entity_type: EntityType) -> str:
        """Normalize entity text to standard medical terminology."""
        normalizations = {
            "AMD": "age-related macular degeneration",
            "ARMD": "age-related macular degeneration",
            "DR": "diabetic retinopathy",
            "PDR": "proliferative diabetic retinopathy",
            "NPDR": "non-proliferative diabetic retinopathy",
            "RPE": "retinal pigment epithelium",
            "IOP": "intraocular pressure",
            "VEGF": "vascular endothelial growth factor",
            "CSC": "central serous chorioretinopathy",
            "CSR": "central serous retinopathy",
        }
        
        # Check EyeCLIP mapping first
        if entity_type in [EntityType.CONDITION, EntityType.FINDING]:
            for key, mapping in self.EYECLIP_CONDITION_MAP.items():
                if text.lower() == key or text.lower() in key:
                    return mapping["normalized"]
        
        return normalizations.get(text.upper(), text.lower().strip())
    
    def _fallback_template_extraction(
        self, text: str, turn_id: int
    ) -> List[ClinicalEntity]:
        """Fallback keyword matching if LLM extraction fails."""
        entities = []
        text_lower = text.lower()
        
        for entity_type, keywords in self._entity_templates.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Boost confidence if keyword appears near EyeCLIP-mentioned terms
                    confidence = 0.6
                    entity = ClinicalEntity(
                        text=keyword,
                        entity_type=EntityType(entity_type),
                        confidence=confidence,
                        source="fallback",
                        turn_id=turn_id,
                    )
                    entities.append(entity)
        
        return entities