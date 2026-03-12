# src/state/clinical_session_state.py

from typing import Optional, List, Dict, Set, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json
import pickle
from .clinical_entity_extractor import ClinicalEntity, EntityType, OphthalmicRegion, IndianClinicalPriority

@dataclass
class StateConfidence:
    """Tracks confidence decay for state elements."""
    value: str
    confidence: float
    last_updated_turn: int
    decay_rate: float = 0.1  # Confidence drops by 10% per turn without reinforcement
    
    def decay(self, current_turn: int) -> float:
        """Apply time-based decay to confidence."""
        turns_since_update = current_turn - self.last_updated_turn
        decayed = self.confidence * (1 - self.decay_rate) ** turns_since_update
        return max(decayed, 0.1)  # Floor at 0.1
    
    def reinforce(self, new_confidence: float, current_turn: int):
        """Reinforce with new evidence (weighted average)."""
        self.confidence = 0.6 * self.confidence + 0.4 * new_confidence
        self.last_updated_turn = current_turn


@dataclass
class ClinicalSessionState:
    """
    Intelligent session state with confidence tracking, topic drift detection,
    and automatic expiration logic.
    """
    
    # Session metadata
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_active_turn: int = 0
    
    # Pinned clinical context (with confidence tracking)
    anatomy_of_interest: Optional[StateConfidence] = None
    primary_condition: Optional[StateConfidence] = None
    secondary_conditions: List[StateConfidence] = field(default_factory=list)
    
    # Symptoms (tracked across conversation)
    symptoms: Dict[str, StateConfidence] = field(default_factory=dict)
    
    # Imaging context
    imaging_modality: Optional[StateConfidence] = None
    
    # Track other clinical markers (with decay)
    clinical_findings: Dict[str, StateConfidence] = field(default_factory=dict)
    medications: Dict[str, StateConfidence] = field(default_factory=dict)
    procedures: Dict[str, StateConfidence] = field(default_factory=dict)
    
    # Aggregated Metadata (Indian context)
    primary_region: OphthalmicRegion = OphthalmicRegion.UNSPECIFIED
    triage_priority: IndianClinicalPriority = IndianClinicalPriority.ROUTINE
    
    # Topic tracking
    topic_history: List[str] = field(default_factory=list)
    topic_drift_detected: bool = False
    
    # Conversation metadata
    total_turns: int = 0
    last_answer_summary: Optional[str] = None
    
    # ── Serialization ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        
        # Convert StateConfidence objects to dicts
        if self.anatomy_of_interest:
            data["anatomy_of_interest"] = asdict(self.anatomy_of_interest)
        if self.primary_condition:
            data["primary_condition"] = asdict(self.primary_condition)
        data["secondary_conditions"] = [asdict(c) for c in self.secondary_conditions]
        data["symptoms"] = {k: asdict(v) for k, v in self.symptoms.items()}
        if self.imaging_modality:
            data["imaging_modality"] = asdict(self.imaging_modality)
        
        data["clinical_findings"] = {k: asdict(v) for k, v in self.clinical_findings.items()}
        data["medications"] = {k: asdict(v) for k, v in self.medications.items()}
        data["procedures"] = {k: asdict(v) for k, v in self.procedures.items()}
        
        data["primary_region"] = self.primary_region.value
        data["triage_priority"] = self.triage_priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "ClinicalSessionState":
        """Reconstruct from dict."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        if data.get("anatomy_of_interest"):
            data["anatomy_of_interest"] = StateConfidence(**data["anatomy_of_interest"])
        if data.get("primary_condition"):
            data["primary_condition"] = StateConfidence(**data["primary_condition"])
        data["secondary_conditions"] = [
            StateConfidence(**c) for c in data.get("secondary_conditions", [])
        ]
        data["symptoms"] = {
            k: StateConfidence(**v) for k, v in data.get("symptoms", {}).items()
        }
        if data.get("imaging_modality"):
            data["imaging_modality"] = StateConfidence(**data["imaging_modality"])
        
        data["clinical_findings"] = {
            k: StateConfidence(**v) for k, v in data.get("clinical_findings", {}).items()
        }
        data["medications"] = {
            k: StateConfidence(**v) for k, v in data.get("medications", {}).items()
        }
        data["procedures"] = {
            k: StateConfidence(**v) for k, v in data.get("procedures", {}).items()
        }
        
        data["primary_region"] = OphthalmicRegion(data.get("primary_region", "unspecified"))
        data["triage_priority"] = IndianClinicalPriority(data.get("triage_priority", "routine"))
        
        return cls(**data)
    
    def save(self, filepath: str):
        """Persist state to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, filepath: str) -> "ClinicalSessionState":
        """Load state from disk."""
        with open(filepath, "rb") as f:
            return cls.from_dict(pickle.load(f))
    
    # ── State Updates ─────────────────────────────────────────────────────
    def update_from_entities(self, entities: List[ClinicalEntity], current_turn: int, text: Optional[str] = None):
        """
        Update state from extracted entities with confidence merging.
        """
        self.total_turns = current_turn
        self.last_active_turn = current_turn
        
        for entity in entities:
            if entity.confidence < 0.5:
                continue
            
            if entity.entity_type == EntityType.ANATOMY:
                self._update_anatomy(entity.text, entity.confidence, current_turn)
            
            elif entity.entity_type == EntityType.CONDITION:
                self._update_condition(entity.text, entity.confidence, current_turn)
            
            elif entity.entity_type == EntityType.SYMPTOM:
                self._update_symptom(entity.text, entity.confidence, current_turn)
            
            elif entity.entity_type == EntityType.IMAGING:
                self._update_imaging(entity.text, entity.confidence, current_turn)
            
            elif entity.entity_type == EntityType.FINDING:
                self._update_finding(entity.text, entity.confidence, current_turn)
                
            elif entity.entity_type == EntityType.MEDICATION:
                self._update_medication(entity.text, entity.confidence, current_turn)
                
            elif entity.entity_type == EntityType.PROCEDURE:
                self._update_procedure(entity.text, entity.confidence, current_turn)
        
        # Apply decay to all state elements
        self._apply_decay(current_turn)
        
        # Update aggregate regions/priorities
        self._aggregate_metadata(entities, text)
        
        # Check for topic drift
        self._detect_topic_drift(current_turn)
    
    def _update_anatomy(self, text: str, confidence: float, turn: int):
        """Update anatomy with confidence-weighted merging."""
        normalized = text.lower()
        
        if self.anatomy_of_interest is None:
            self.anatomy_of_interest = StateConfidence(
                value=normalized,
                confidence=confidence,
                last_updated_turn=turn,
            )
        elif self.anatomy_of_interest.value == normalized:
            # Reinforce existing anatomy
            self.anatomy_of_interest.reinforce(confidence, turn)
        else:
            # New anatomy detected - check if confidence is higher
            current_confidence = self.anatomy_of_interest.decay(turn)
            if confidence > current_confidence + 0.2:  # Significant shift
                print(f"[State] Anatomy shift detected: {self.anatomy_of_interest.value} → {normalized}")
                self.anatomy_of_interest = StateConfidence(
                    value=normalized,
                    confidence=confidence,
                    last_updated_turn=turn,
                )
                self.topic_drift_detected = True
    
    def _update_condition(self, text: str, confidence: float, turn: int):
        """Update primary/secondary conditions."""
        normalized = text.lower()
        
        # Check if this matches primary condition
        if self.primary_condition and self.primary_condition.value == normalized:
            self.primary_condition.reinforce(confidence, turn)
            return
        
        # Check secondary conditions
        for sec in self.secondary_conditions:
            if sec.value == normalized:
                sec.reinforce(confidence, turn)
                return
        
        # New condition - add as secondary or promote to primary
        new_conf = StateConfidence(value=normalized, confidence=confidence, last_updated_turn=turn)
        
        if self.primary_condition is None or confidence > self.primary_condition.decay(turn) + 0.3:
            # Promote to primary
            if self.primary_condition:
                self.secondary_conditions.append(self.primary_condition)
            self.primary_condition = new_conf
        else:
            self.secondary_conditions.append(new_conf)
        
        # Limit secondary conditions
        self.secondary_conditions = self.secondary_conditions[-5:]
    
    def _update_symptom(self, text: str, confidence: float, turn: int):
        """Track symptoms across conversation."""
        normalized = text.lower()
        if normalized in self.symptoms:
            self.symptoms[normalized].reinforce(confidence, turn)
        else:
            self.symptoms[normalized] = StateConfidence(
                value=normalized,
                confidence=confidence,
                last_updated_turn=turn,
            )
    
    def _update_imaging(self, text: str, confidence: float, turn: int):
        """Update imaging modality context."""
        normalized = text.upper() if len(text) <= 5 else text.lower()
        
        if self.imaging_modality is None:
            self.imaging_modality = StateConfidence(
                value=normalized,
                confidence=confidence,
                last_updated_turn=turn,
            )
        elif self.imaging_modality.value == normalized:
            self.imaging_modality.reinforce(confidence, turn)

    def _update_finding(self, text: str, confidence: float, turn: int):
        """Update clinical findings."""
        normalized = text.lower()
        if normalized in self.clinical_findings:
            self.clinical_findings[normalized].reinforce(confidence, turn)
        else:
            self.clinical_findings[normalized] = StateConfidence(normalized, confidence, turn)

    def _update_medication(self, text: str, confidence: float, turn: int):
        """Update medications."""
        normalized = text.lower()
        if normalized in self.medications:
            self.medications[normalized].reinforce(confidence, turn)
        else:
            self.medications[normalized] = StateConfidence(normalized, confidence, turn)

    def _update_procedure(self, text: str, confidence: float, turn: int):
        """Update procedures."""
        normalized = text.lower()
        if normalized in self.procedures:
            self.procedures[normalized].reinforce(confidence, turn)
        else:
            self.procedures[normalized] = StateConfidence(normalized, confidence, turn)
    
    def _apply_decay(self, current_turn: int):
        """Apply confidence decay to all state elements."""
        if self.anatomy_of_interest:
            self.anatomy_of_interest.confidence = self.anatomy_of_interest.decay(current_turn)
        
        if self.primary_condition:
            self.primary_condition.confidence = self.primary_condition.decay(current_turn)
        
        for sec in self.secondary_conditions:
            sec.confidence = sec.decay(current_turn)
        
        for symptom in self.symptoms.values():
            symptom.confidence = symptom.decay(current_turn)
        
        for tracker in [self.clinical_findings, self.medications, self.procedures]:
            for item in tracker.values():
                item.confidence = item.decay(current_turn)
        
        if self.imaging_modality:
            self.imaging_modality.confidence = self.imaging_modality.decay(current_turn)

    def _aggregate_metadata(self, entities: List[ClinicalEntity], text: Optional[str] = None):
        """Aggregate granular entity metadata into session-level insights."""
        if not entities and not text:
            return

        # 1. Update Primary Region (Majority vote based on confidence)
        if entities:
            region_scores = {region: 0.0 for region in OphthalmicRegion}
            for entity in entities:
                if entity.region != OphthalmicRegion.UNSPECIFIED:
                    region_scores[entity.region] += entity.confidence
            
            best_region = max(region_scores, key=region_scores.get)
            if region_scores[best_region] > 0.5:
                self.primary_region = best_region

        # 2. Update Triage Priority (Escalation logic)
        priority_order = {
            IndianClinicalPriority.LEVEL_A: 3,
            IndianClinicalPriority.LEVEL_B: 2,
            IndianClinicalPriority.LEVEL_C: 1,
            IndianClinicalPriority.ROUTINE: 0
        }
        
        current_priority_val = priority_order.get(self.triage_priority, 0)
        for entity in entities:
            entity_priority_val = priority_order.get(entity.priority, 0)
            if entity_priority_val > current_priority_val:
                self.triage_priority = entity.priority
                current_priority_val = entity_priority_val
    
    def _detect_topic_drift(self, current_turn: int):
        """Detect if conversation has shifted to a new topic."""
        current_anatomy = self.anatomy_of_interest.value if self.anatomy_of_interest else None
        
        # Always track topic history (even on first turn)
        topic_label = (
            current_anatomy
            or (self.primary_condition.value if self.primary_condition else None)
            or "general"
        )
        self.topic_history.append(topic_label)
        self.topic_history = self.topic_history[-10:]  # Keep last 10 topics
        
        # Only detect drift after 2+ topics recorded
        if len(self.topic_history) < 2:
            return
        
        # Simple heuristic: if anatomy changes significantly, flag drift
        recent_anatomy = self.topic_history[-2]  # Previous topic
        
        if recent_anatomy and current_anatomy and recent_anatomy != current_anatomy:
            self.topic_drift_detected = True
            print(f"[State] Topic drift detected at turn {current_turn}")
    
    # ── Query Context Generation ──────────────────────────────────────────
    def to_query_context(self) -> str:
        """
        Convert state to compact query augmentation string for retrieval.
        Only includes high-confidence elements.
        """
        parts = []
        
        if self.anatomy_of_interest and self.anatomy_of_interest.confidence > 0.5:
            parts.append(f"anatomy:{self.anatomy_of_interest.value}")
        
        if self.primary_condition and self.primary_condition.confidence > 0.5:
            parts.append(f"condition:{self.primary_condition.value}")
        
        if self.secondary_conditions:
            high_conf_secondaries = [
                c.value for c in self.secondary_conditions if c.confidence > 0.5
            ]
            if high_conf_secondaries:
                parts.append(f"secondary:{', '.join(high_conf_secondaries)}")
        
        if self.imaging_modality and self.imaging_modality.confidence > 0.5:
            parts.append(f"imaging:{self.imaging_modality.value}")
        
        return " [" + " | ".join(parts) + "]" if parts else ""
    
    def to_generation_context(self) -> str:
        """
        Convert state to rich context for answer generation.
        Includes more detail than query context.
        """
        parts = []
        
        # Indian Localization Metadata
        if self.primary_region != OphthalmicRegion.UNSPECIFIED:
            parts.append(f"Anatomical Region: {self.primary_region.value}")
        
        if self.triage_priority != IndianClinicalPriority.ROUTINE:
            parts.append(f"Clinical Priority: {self.triage_priority.value.upper()}")
        
        if self.anatomy_of_interest:
            conf = f" (confidence: {self.anatomy_of_interest.confidence:.2f})"
            parts.append(f"Anatomical focus: {self.anatomy_of_interest.value}{conf}")
        
        if self.primary_condition:
            conf = f" (confidence: {self.primary_condition.confidence:.2f})"
            parts.append(f"Primary condition: {self.primary_condition.value}{conf}")
        
        if self.secondary_conditions:
            conditions = [
                f"{c.value} ({c.confidence:.2f})"
                for c in self.secondary_conditions if c.confidence > 0.4
            ]
            if conditions:
                parts.append(f"Other conditions: {', '.join(conditions)}")
        
        if self.symptoms:
            symptoms = [
                f"{s.value} ({s.confidence:.2f})"
                for s in self.symptoms.values() if s.confidence > 0.4
            ]
            if symptoms:
                parts.append(f"Reported symptoms: {', '.join(symptoms)}")
        
        if self.clinical_findings:
            findings = [
                f"{f.value} ({f.confidence:.2f})"
                for f in self.clinical_findings.values() if f.confidence > 0.4
            ]
            if findings:
                parts.append(f"Clinical findings: {', '.join(findings)}")

        if self.medications:
            meds = [f.value for f in self.medications.values() if f.confidence > 0.4]
            if meds:
                parts.append(f"Active medications: {', '.join(meds)}")

        if self.procedures:
            procs = [f.value for f in self.procedures.values() if f.confidence > 0.4]
            if procs:
                parts.append(f"Relevant procedures: {', '.join(procs)}")
        
        if self.imaging_modality:
            parts.append(f"Imaging: {self.imaging_modality.value}")
        
        if self.topic_drift_detected:
            parts.append("⚠ Topic shift detected in conversation")
        
        return "\n".join(parts) if parts else ""
    
    def should_reset(self, max_inactive_turns: int = 15) -> bool:
        """Check if state should be reset due to inactivity or low confidence."""
        if self.total_turns == 0:
            return False
        
        turns_since_active = self.total_turns - self.last_active_turn
        if turns_since_active > max_inactive_turns:
            return True
        
        # Check if all confidences have decayed too low
        if self.anatomy_of_interest and self.anatomy_of_interest.confidence < 0.2:
            if self.primary_condition and self.primary_condition.confidence < 0.2:
                return True
        
        return False
    
    def reset_for_new_topic(self, preserve_session_id: bool = True):
        """Reset state while preserving session metadata."""
        old_session_id = self.session_id if preserve_session_id else None
        old_created_at = self.created_at
        
        # Create fresh state
        new_state = ClinicalSessionState(
            session_id=old_session_id or self.session_id,
            created_at=old_created_at,
        )
        
        # Copy session metadata
        new_state.topic_history = self.topic_history[-3:]  # Keep recent topic history
        new_state.total_turns = self.total_turns
        
        return new_state