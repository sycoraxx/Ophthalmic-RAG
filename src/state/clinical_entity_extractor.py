# src/state/clinical_entity_extractor.py

from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re

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
    
    def __init__(self, medgemma_generator):
        self.generator = medgemma_generator
        self._entity_templates = self._load_entity_templates()
    
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
                "metamorphopsia", "scotoma", "double vision", "halos"
            ],
            "imaging": list(self.EYECLIP_MODALITY_MAP.values()),
        }
    
    def extract_entities(
        self,
        text: str,
        visual_findings: Optional[str] = None,
        turn_id: int = 0,
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
        entities = []
        
        # ── Step 1: Extract from EyeCLIP visual findings (HIGH CONFIDENCE) ──
        if visual_findings:
            eyeclip_entities = self._extract_from_eyeclip(visual_findings, turn_id)
            entities.extend(eyeclip_entities)
        
        # ── Step 2: Extract from text using LLM ─────────────────────────────
        text_entities = self._extract_from_text_llm(text, turn_id, visual_findings)
        
        # ── Step 3: Merge entities with confidence weighting ────────────────
        merged_entities = self._merge_entities(entities, text_entities)
        
        # ── Step 4: Filter low-confidence and negated entities ──────────────
        filtered = [
            e for e in merged_entities 
            if e.confidence >= 0.5 and not self._is_negated(text, e.text)
        ]
        
        # ── Step 5: Tag with Locality and Priority metadata ─────────────
        final_entities = self._tag_locality_and_priority(filtered)
        
        return final_entities
    
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
        modality_match = re.search(r'Detected Image Type:\s*(\w+)', visual_findings, re.I)
        detected_modality = None
        if modality_match:
            modality_key = modality_match.group(1).upper()
            normalized_modality = self.EYECLIP_MODALITY_MAP.get(modality_key, modality_key.lower())
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
        finding_pattern = r'●\s*(Probable|Possible|Detected):\s*([^(]+?)\s*\(([\d.]+)%\)'
        for match in re.finditer(finding_pattern, visual_findings, re.I):
            confidence_label = match.group(1).lower()
            condition_text = match.group(2).strip().lower()
            confidence_score = float(match.group(3)) / 100.0
            
            # Map confidence label to numeric adjustment
            label_adjustment = {
                "probable": 0.15,  # Add 15% to model confidence
                "possible": -0.10, # Subtract 10% for uncertainty
                "detected": 0.0,   # Neutral
            }
            adjusted_confidence = min(confidence_score + label_adjustment.get(confidence_label, 0), 1.0)
            
            # Skip very low confidence findings
            if adjusted_confidence < 0.2:
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
        visual_findings: Optional[str] = None
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
            
            # Find the JSON array using regex, spanning multiple lines
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if match:
                json_str = f"[{match.group(1)}]"
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
                if item.get("confidence", 0) < 0.5:
                    continue
                
                entity_type = EntityType(item.get("entity_type", "finding"))
                
                entity = ClinicalEntity(
                    text=item["text"],
                    entity_type=entity_type,
                    confidence=item["confidence"],
                    source="answer",
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