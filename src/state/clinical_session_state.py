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
    evidence_count: int = 1
    decay_rate: float = 0.1  # Confidence drops by 10% per turn without reinforcement
    
    def decay(self, current_turn: int) -> float:
        """Apply time-based decay to confidence."""
        turns_since_update = current_turn - self.last_updated_turn
        decayed = self.confidence * (1 - self.decay_rate) ** turns_since_update
        return max(decayed, 0.1)  # Floor at 0.1
    
    def reinforce(self, new_confidence: float, current_turn: int):
        """Reinforce with new evidence (weighted average)."""
        alpha = 0.25 if self.evidence_count >= 2 else 0.4
        self.confidence = (1 - alpha) * self.confidence + alpha * new_confidence
        self.last_updated_turn = current_turn
        self.evidence_count = min(self.evidence_count + 1, 8)

    def is_stable(
        self,
        threshold: float = 0.35,
        min_evidence: int = 2,
        high_confidence_override: float = 0.65,
    ) -> bool:
        """Treat an item as stable only after repeated support, unless confidence is very high."""
        if self.confidence >= high_confidence_override:
            return True
        return self.confidence >= threshold and self.evidence_count >= min_evidence


@dataclass
class PendingEvidence:
    """Tracks candidate entities before promoting them into persistent memory."""
    value: str
    confidence: float
    first_seen_turn: int
    last_seen_turn: int
    hits: int = 1

    def reinforce(self, new_confidence: float, current_turn: int):
        self.confidence = 0.7 * self.confidence + 0.3 * new_confidence
        self.last_seen_turn = current_turn
        self.hits += 1

    def is_actionable(self, min_confidence: float = 0.55, min_hits: int = 1) -> bool:
        """Whether this pending item is reliable enough for short-horizon carry-over."""
        return self.confidence >= min_confidence and self.hits >= min_hits


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

    # Pending candidates to avoid single-turn context poisoning
    pending_anatomy_shift: Optional[PendingEvidence] = None
    pending_condition_shift: Optional[PendingEvidence] = None
    pending_symptoms: Dict[str, PendingEvidence] = field(default_factory=dict)
    pending_findings: Dict[str, PendingEvidence] = field(default_factory=dict)
    pending_medications: Dict[str, PendingEvidence] = field(default_factory=dict)
    pending_procedures: Dict[str, PendingEvidence] = field(default_factory=dict)
    
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

        if data.get("pending_anatomy_shift"):
            data["pending_anatomy_shift"] = PendingEvidence(**data["pending_anatomy_shift"])
        if data.get("pending_condition_shift"):
            data["pending_condition_shift"] = PendingEvidence(**data["pending_condition_shift"])

        data["pending_symptoms"] = {
            k: PendingEvidence(**v) for k, v in data.get("pending_symptoms", {}).items()
        }
        data["pending_findings"] = {
            k: PendingEvidence(**v) for k, v in data.get("pending_findings", {}).items()
        }
        data["pending_medications"] = {
            k: PendingEvidence(**v) for k, v in data.get("pending_medications", {}).items()
        }
        data["pending_procedures"] = {
            k: PendingEvidence(**v) for k, v in data.get("pending_procedures", {}).items()
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
            # EyeCLIP entities use a lower threshold (their scores are inherently low but accurate)
            min_confidence = 0.05 if entity.source == "eyeclip" else 0.3
            if entity.confidence < min_confidence:
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
        self._prune_pending_candidates(current_turn)
        
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
                evidence_count=2 if confidence >= 0.6 else 1,
            )
        elif self.anatomy_of_interest.value == normalized:
            # Reinforce existing anatomy
            self.anatomy_of_interest.reinforce(confidence, turn)
            self.pending_anatomy_shift = None
        else:
            if self._confirm_shift(
                pending_attr="pending_anatomy_shift",
                new_value=normalized,
                new_confidence=confidence,
                current_turn=turn,
                current_item=self.anatomy_of_interest,
                min_margin=0.15,
                min_hits=2,
            ):
                print(f"[State] Anatomy shift detected: {self.anatomy_of_interest.value} → {normalized}")
                self.anatomy_of_interest = StateConfidence(
                    value=normalized,
                    confidence=confidence,
                    last_updated_turn=turn,
                    evidence_count=2,
                )
                self.topic_drift_detected = True
    
    def _update_condition(self, text: str, confidence: float, turn: int):
        """Update primary/secondary conditions."""
        normalized = text.lower()
        
        # Check if this matches primary condition
        if self.primary_condition and self.primary_condition.value == normalized:
            self.primary_condition.reinforce(confidence, turn)
            self.pending_condition_shift = None
            return
        
        # Check secondary conditions
        for sec in self.secondary_conditions:
            if sec.value == normalized:
                sec.reinforce(confidence, turn)

                # Promote a reinforced secondary only after shift confirmation.
                if self.primary_condition and self._confirm_shift(
                    pending_attr="pending_condition_shift",
                    new_value=normalized,
                    new_confidence=sec.confidence,
                    current_turn=turn,
                    current_item=self.primary_condition,
                    min_margin=0.2,
                    min_hits=2,
                ):
                    self.secondary_conditions = [s for s in self.secondary_conditions if s.value != normalized]
                    self.secondary_conditions.append(self.primary_condition)
                    self.primary_condition = sec
                return
        
        # New condition - add as secondary or promote to primary
        new_conf = StateConfidence(
            value=normalized,
            confidence=confidence,
            last_updated_turn=turn,
            evidence_count=2 if confidence >= 0.6 else 1,
        )
        
        should_promote = False
        if self.primary_condition is None:
            should_promote = True
        else:
            should_promote = self._confirm_shift(
                pending_attr="pending_condition_shift",
                new_value=normalized,
                new_confidence=confidence,
                current_turn=turn,
                current_item=self.primary_condition,
                min_margin=0.2,
                min_hits=2,
            )

        if should_promote:
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
            self._stage_or_promote(
                stable_attr="symptoms",
                pending_attr="pending_symptoms",
                value=normalized,
                confidence=confidence,
                turn=turn,
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
        else:
            # Different modality — replace (new image uploaded)
            print(f"[State] Imaging modality changed: {self.imaging_modality.value} → {normalized}")
            self.imaging_modality = StateConfidence(
                value=normalized,
                confidence=confidence,
                last_updated_turn=turn,
            )

    def _update_finding(self, text: str, confidence: float, turn: int):
        """Update clinical findings."""
        normalized = text.lower()
        if normalized in self.clinical_findings:
            self.clinical_findings[normalized].reinforce(confidence, turn)
        else:
            self._stage_or_promote(
                stable_attr="clinical_findings",
                pending_attr="pending_findings",
                value=normalized,
                confidence=confidence,
                turn=turn,
            )

    def _update_medication(self, text: str, confidence: float, turn: int):
        """Update medications."""
        normalized = text.lower()
        if normalized in self.medications:
            self.medications[normalized].reinforce(confidence, turn)
        else:
            self._stage_or_promote(
                stable_attr="medications",
                pending_attr="pending_medications",
                value=normalized,
                confidence=confidence,
                turn=turn,
            )

    def _update_procedure(self, text: str, confidence: float, turn: int):
        """Update procedures."""
        normalized = text.lower()
        if normalized in self.procedures:
            self.procedures[normalized].reinforce(confidence, turn)
        else:
            self._stage_or_promote(
                stable_attr="procedures",
                pending_attr="pending_procedures",
                value=normalized,
                confidence=confidence,
                turn=turn,
            )

    def _confirm_shift(
        self,
        pending_attr: str,
        new_value: str,
        new_confidence: float,
        current_turn: int,
        current_item: Optional[StateConfidence],
        min_margin: float,
        min_hits: int,
    ) -> bool:
        """Require repeated evidence before replacing a pinned primary context field."""
        pending = getattr(self, pending_attr)

        if pending and pending.value == new_value and current_turn - pending.last_seen_turn <= 2:
            pending.reinforce(new_confidence, current_turn)
        else:
            pending = PendingEvidence(
                value=new_value,
                confidence=new_confidence,
                first_seen_turn=current_turn,
                last_seen_turn=current_turn,
            )
            setattr(self, pending_attr, pending)

        current_conf = current_item.decay(current_turn) if current_item else 0.0

        # Allow immediate shift only for very strong contradictory evidence.
        immediate_shift = new_confidence >= current_conf + (min_margin + 0.2)
        confirmed_shift = pending.hits >= min_hits and pending.confidence >= current_conf + min_margin

        if immediate_shift or confirmed_shift:
            setattr(self, pending_attr, None)
            return True

        return False

    def _stage_or_promote(
        self,
        stable_attr: str,
        pending_attr: str,
        value: str,
        confidence: float,
        turn: int,
        immediate_threshold: float = 0.75,
        promote_threshold: float = 0.35,
        min_hits: int = 2,
    ):
        """Stage low-confidence entities and promote only after repeated support."""
        stable_tracker: Dict[str, StateConfidence] = getattr(self, stable_attr)
        pending_tracker: Dict[str, PendingEvidence] = getattr(self, pending_attr)

        if value in stable_tracker:
            stable_tracker[value].reinforce(confidence, turn)
            pending_tracker.pop(value, None)
            return

        if confidence >= immediate_threshold:
            stable_tracker[value] = StateConfidence(
                value=value,
                confidence=confidence,
                last_updated_turn=turn,
                evidence_count=2,
            )
            pending_tracker.pop(value, None)
            return

        candidate = pending_tracker.get(value)
        if candidate and turn - candidate.last_seen_turn <= 2:
            candidate.reinforce(confidence, turn)
        else:
            pending_tracker[value] = PendingEvidence(
                value=value,
                confidence=confidence,
                first_seen_turn=turn,
                last_seen_turn=turn,
            )
            return

        if candidate.hits >= min_hits and candidate.confidence >= promote_threshold:
            stable_tracker[value] = StateConfidence(
                value=value,
                confidence=candidate.confidence,
                last_updated_turn=turn,
                evidence_count=candidate.hits,
            )
            pending_tracker.pop(value, None)

    def _prune_pending_candidates(self, current_turn: int, max_age_turns: int = 3):
        """Drop stale pending candidates that were not reinforced across recent turns."""
        for attr_name in (
            "pending_symptoms",
            "pending_findings",
            "pending_medications",
            "pending_procedures",
        ):
            tracker = getattr(self, attr_name)
            setattr(
                self,
                attr_name,
                {
                    k: v
                    for k, v in tracker.items()
                    if current_turn - v.last_seen_turn <= max_age_turns
                },
            )

        if self.pending_anatomy_shift and current_turn - self.pending_anatomy_shift.last_seen_turn > max_age_turns:
            self.pending_anatomy_shift = None
        if self.pending_condition_shift and current_turn - self.pending_condition_shift.last_seen_turn > max_age_turns:
            self.pending_condition_shift = None

    def _is_stable_context_item(self, item: Optional[StateConfidence], threshold: float = 0.35) -> bool:
        if not item:
            return False
        return item.is_stable(threshold=threshold)
    
    def _apply_decay(self, current_turn: int):
        """Apply confidence decay to all state elements and prune stale entries."""
        PRUNE_THRESHOLD = 0.15  # Evict entries that have decayed below this
        
        if self.anatomy_of_interest:
            self.anatomy_of_interest.confidence = self.anatomy_of_interest.decay(current_turn)
            if self.anatomy_of_interest.confidence < PRUNE_THRESHOLD:
                self.anatomy_of_interest = None
        
        if self.primary_condition:
            self.primary_condition.confidence = self.primary_condition.decay(current_turn)
            if self.primary_condition.confidence < PRUNE_THRESHOLD:
                self.primary_condition = None
        
        # Prune secondary conditions
        for sec in self.secondary_conditions:
            sec.confidence = sec.decay(current_turn)
        self.secondary_conditions = [
            s
            for s in self.secondary_conditions
            if s.confidence >= PRUNE_THRESHOLD and (s.evidence_count >= 2 or s.confidence >= 0.5)
        ]
        
        # Prune symptoms
        for symptom in self.symptoms.values():
            symptom.confidence = symptom.decay(current_turn)
        self.symptoms = {
            k: v
            for k, v in self.symptoms.items()
            if v.confidence >= PRUNE_THRESHOLD and (v.evidence_count >= 2 or v.confidence >= 0.55)
        }
        
        # Prune findings, medications, procedures
        for attr_name in ('clinical_findings', 'medications', 'procedures'):
            tracker = getattr(self, attr_name)
            for item in tracker.values():
                item.confidence = item.decay(current_turn)
            setattr(
                self,
                attr_name,
                {
                    k: v
                    for k, v in tracker.items()
                    if v.confidence >= PRUNE_THRESHOLD and (v.evidence_count >= 2 or v.confidence >= 0.55)
                },
            )
        
        if self.imaging_modality:
            self.imaging_modality.confidence = self.imaging_modality.decay(current_turn)
            if self.imaging_modality.confidence < PRUNE_THRESHOLD:
                self.imaging_modality = None

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
            
            best_region = max(region_scores.items(), key=lambda item: item[1])[0]
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
        """Detect if conversation has shifted to a genuinely new clinical topic.
        
        Refined: also considers condition-to-anatomy relationships (e.g. cataract -> lens)
        to prevent false positives.
        """
        current_anatomy = self.anatomy_of_interest.value if self.anatomy_of_interest else None
        current_condition = self.primary_condition.value if self.primary_condition else None
        
        # Build topic label from the most specific available context
        topic_label = current_anatomy or current_condition or None
        
        # If the current turn produced NO identifiable topic (vague follow-up),
        # inherit the previous topic instead of defaulting to "general".
        if topic_label is None and self.topic_history:
            topic_label = self.topic_history[-1]
        elif topic_label is None:
            topic_label = "general"
        
        self.topic_history.append(topic_label)
        self.topic_history = self.topic_history[-10:]
        
        if len(self.topic_history) < 2:
            return
        
        previous_topic = self.topic_history[-2]
        
        # ── Anatomy-aware Drift Detection ──────────────────────────────────
        # Map common conditions to their primary anatomy to bridge the gap
        condition_to_anatomy = {
            "cataract": "lens",
            "glaucoma": "optic nerve",
            "uveitis": "uvea",
            "keratitis": "cornea",
            "conjunctivitis": "conjunctiva",
            "diabetic retinopathy": "retina",
            "amd": "macula",
            "drusen": "macula",
            "posterior subcapsular cataract": "lens",
        }
        
        prev_canon = condition_to_anatomy.get(previous_topic.lower(), previous_topic.lower())
        curr_canon = condition_to_anatomy.get(topic_label.lower(), topic_label.lower())
        
        is_both_specific = (
            previous_topic != "general"
            and topic_label != "general"
            and prev_canon != curr_canon
        )
        
        # Only flag drift when a *new, explicit* anatomy contradicts the old one (or its canonical form)
        if is_both_specific and current_anatomy and prev_canon != current_anatomy.lower():
            self.topic_drift_detected = True
            print(f"[State] Topic drift detected at turn {current_turn}: '{previous_topic}' → '{current_anatomy}'")
        elif not self.topic_drift_detected:
            # Only clear if _update_anatomy hasn't already flagged drift this cycle
            self.topic_drift_detected = False
    
    def has_context(
        self,
        threshold: float = 0.4,
        include_provisional: bool = False,
        provisional_min_confidence: float = 0.55,
    ) -> bool:
        """Check if session contains any active, meaningful clinical context."""
        if self._is_stable_context_item(self.primary_condition, threshold):
            return True
        if self._is_stable_context_item(self.anatomy_of_interest, threshold):
            return True
        if any(c.is_stable(threshold=threshold) for c in self.secondary_conditions):
            return True
        if any(f.is_stable(threshold=threshold) for f in self.clinical_findings.values()):
            return True
        if any(s.is_stable(threshold=threshold) for s in self.symptoms.values()):
            return True
        if any(m.is_stable(threshold=threshold) for m in self.medications.values()):
            return True
        if any(p.is_stable(threshold=threshold) for p in self.procedures.values()):
            return True

        if include_provisional:
            pending_trackers = (
                self.pending_symptoms,
                self.pending_findings,
                self.pending_medications,
                self.pending_procedures,
            )
            for tracker in pending_trackers:
                if any(v.is_actionable(min_confidence=provisional_min_confidence) for v in tracker.values()):
                    return True

            if self.pending_anatomy_shift and self.pending_anatomy_shift.is_actionable(
                min_confidence=provisional_min_confidence
            ):
                return True
            if self.pending_condition_shift and self.pending_condition_shift.is_actionable(
                min_confidence=provisional_min_confidence
            ):
                return True
        return False

    def _collect_provisional_terms(
        self,
        *,
        min_confidence: float = 0.55,
        max_per_bucket: int = 3,
    ) -> Dict[str, List[str]]:
        """Collect top pending terms that are safe to use for immediate follow-up carry-over."""

        def top_pending(tracker: Dict[str, PendingEvidence]) -> List[str]:
            picked = [
                p.value
                for p in sorted(
                    tracker.values(),
                    key=lambda x: (x.confidence, x.hits, x.last_seen_turn),
                    reverse=True,
                )
                if p.is_actionable(min_confidence=min_confidence)
            ]
            return picked[:max_per_bucket]

        provisional = {
            "symptoms": top_pending(self.pending_symptoms),
            "findings": top_pending(self.pending_findings),
            "medications": top_pending(self.pending_medications),
            "procedures": top_pending(self.pending_procedures),
        }

        if not self.anatomy_of_interest and self.pending_anatomy_shift and self.pending_anatomy_shift.is_actionable(
            min_confidence=min_confidence
        ):
            provisional["anatomy"] = [self.pending_anatomy_shift.value]

        if not self.primary_condition and self.pending_condition_shift and self.pending_condition_shift.is_actionable(
            min_confidence=min_confidence
        ):
            provisional["condition"] = [self.pending_condition_shift.value]

        return provisional

    # ── Query Context Generation ──────────────────────────────────────────
    def to_query_context(
        self,
        include_provisional: bool = False,
        provisional_min_confidence: float = 0.55,
    ) -> str:
        """
        Convert state to compact query augmentation string for retrieval.
        Only includes high-confidence elements.
        """
        parts = []
        
        def _merge_bucket(stable_values: List[str], provisional_values: List[str], cap: int) -> List[str]:
            seen: Set[str] = set()
            merged: List[str] = []
            for value in stable_values + provisional_values:
                key = value.strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(value)
                if len(merged) >= cap:
                    break
            return merged

        provisional = (
            self._collect_provisional_terms(min_confidence=provisional_min_confidence)
            if include_provisional
            else {}
        )

        anatomy = self.anatomy_of_interest if self._is_stable_context_item(self.anatomy_of_interest, 0.3) else None
        anatomy_values = _merge_bucket(
            [anatomy.value] if anatomy else [],
            provisional.get("anatomy", []),
            cap=1,
        )
        if anatomy_values:
            parts.append(f"anatomy:{', '.join(anatomy_values)}")

        primary_condition = self.primary_condition if self._is_stable_context_item(self.primary_condition, 0.3) else None
        condition_values = _merge_bucket(
            [primary_condition.value] if primary_condition else [],
            provisional.get("condition", []),
            cap=1,
        )
        if condition_values:
            parts.append(f"condition:{', '.join(condition_values)}")

        high_conf_secondaries = [
            c.value for c in self.secondary_conditions if c.is_stable(threshold=0.3)
        ]
        if high_conf_secondaries:
            parts.append(f"secondary:{', '.join(high_conf_secondaries[:3])}")

        high_conf_findings = [
            f.value
            for f in sorted(self.clinical_findings.values(), key=lambda x: x.confidence, reverse=True)
            if f.is_stable(threshold=0.3)
        ]
        findings_values = _merge_bucket(high_conf_findings, provisional.get("findings", []), cap=4)
        if findings_values:
            parts.append(f"findings:{', '.join(findings_values)}")

        high_conf_symptoms = [
            s.value
            for s in sorted(self.symptoms.values(), key=lambda x: x.confidence, reverse=True)
            if s.is_stable(threshold=0.3)
        ]
        symptom_values = _merge_bucket(high_conf_symptoms, provisional.get("symptoms", []), cap=4)
        if symptom_values:
            parts.append(f"symptoms:{', '.join(symptom_values)}")

        high_conf_meds = [
            m.value
            for m in sorted(self.medications.values(), key=lambda x: x.confidence, reverse=True)
            if m.is_stable(threshold=0.3)
        ]
        med_values = _merge_bucket(high_conf_meds, provisional.get("medications", []), cap=3)
        if med_values:
            parts.append(f"medications:{', '.join(med_values)}")

        high_conf_procs = [
            p.value
            for p in sorted(self.procedures.values(), key=lambda x: x.confidence, reverse=True)
            if p.is_stable(threshold=0.3)
        ]
        proc_values = _merge_bucket(high_conf_procs, provisional.get("procedures", []), cap=3)
        if proc_values:
            parts.append(f"procedures:{', '.join(proc_values)}")

        imaging = self.imaging_modality if self._is_stable_context_item(self.imaging_modality, 0.3) else None
        if imaging:
            parts.append(f"imaging:{imaging.value}")
        
        return " [" + " | ".join(parts) + "]" if parts else ""

    def to_query_terms(
        self,
        include_provisional: bool = False,
        provisional_min_confidence: float = 0.55,
    ) -> str:
        """Flatten high-confidence state into retrieval-friendly plain terms."""
        terms: List[str] = []

        anatomy = self.anatomy_of_interest if self._is_stable_context_item(self.anatomy_of_interest, 0.3) else None
        if anatomy:
            terms.append(anatomy.value)

        primary_condition = self.primary_condition if self._is_stable_context_item(self.primary_condition, 0.3) else None
        if primary_condition:
            terms.append(primary_condition.value)

        terms.extend([c.value for c in self.secondary_conditions if c.is_stable(threshold=0.35)][:3])
        terms.extend([
            f.value
            for f in sorted(self.clinical_findings.values(), key=lambda x: x.confidence, reverse=True)
            if f.is_stable(threshold=0.35)
        ][:4])
        terms.extend([
            s.value
            for s in sorted(self.symptoms.values(), key=lambda x: x.confidence, reverse=True)
            if s.is_stable(threshold=0.35)
        ][:4])
        terms.extend([
            m.value
            for m in sorted(self.medications.values(), key=lambda x: x.confidence, reverse=True)
            if m.is_stable(threshold=0.35)
        ][:3])
        terms.extend([
            p.value
            for p in sorted(self.procedures.values(), key=lambda x: x.confidence, reverse=True)
            if p.is_stable(threshold=0.35)
        ][:3])

        imaging = self.imaging_modality if self._is_stable_context_item(self.imaging_modality, 0.3) else None
        if imaging:
            terms.append(imaging.value)

        if include_provisional:
            provisional = self._collect_provisional_terms(min_confidence=provisional_min_confidence)
            for bucket in ("anatomy", "condition", "findings", "symptoms", "medications", "procedures"):
                terms.extend(provisional.get(bucket, []))

        # Order-preserving dedupe
        seen = set()
        deduped = []
        for term in terms:
            t = term.strip().lower()
            if not t or t in seen:
                continue
            seen.add(t)
            deduped.append(term)

        return " ".join(deduped)
    
    def to_generation_context(
        self,
        include_provisional: bool = True,
        provisional_min_confidence: float = 0.55,
    ) -> str:
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
                for s in self.symptoms.values() if s.is_stable(threshold=0.4)
            ]
            if symptoms:
                parts.append(f"Reported symptoms: {', '.join(symptoms)}")
        
        if self.clinical_findings:
            findings = [
                f"{f.value} ({f.confidence:.2f})"
                for f in self.clinical_findings.values() if f.is_stable(threshold=0.4)
            ]
            if findings:
                parts.append(f"Clinical findings: {', '.join(findings)}")

        if self.medications:
            meds = [f.value for f in self.medications.values() if f.is_stable(threshold=0.4)]
            if meds:
                parts.append(f"Active medications: {', '.join(meds)}")

        if self.procedures:
            procs = [f.value for f in self.procedures.values() if f.is_stable(threshold=0.4)]
            if procs:
                parts.append(f"Relevant procedures: {', '.join(procs)}")
        
        if self.imaging_modality:
            parts.append(f"Imaging: {self.imaging_modality.value}")

        if include_provisional:
            provisional = self._collect_provisional_terms(min_confidence=provisional_min_confidence)
            provisional_lines = []
            if provisional.get("anatomy") and not self.anatomy_of_interest:
                provisional_lines.append(f"Possible anatomy focus: {', '.join(provisional['anatomy'])}")
            if provisional.get("condition") and not self.primary_condition:
                provisional_lines.append(f"Possible condition clues: {', '.join(provisional['condition'])}")
            if provisional.get("findings"):
                provisional_lines.append(f"Recent findings clues: {', '.join(provisional['findings'])}")
            if provisional.get("symptoms"):
                provisional_lines.append(f"Recent symptom clues: {', '.join(provisional['symptoms'])}")
            if provisional.get("medications"):
                provisional_lines.append(f"Recent medication mentions: {', '.join(provisional['medications'])}")
            if provisional.get("procedures"):
                provisional_lines.append(f"Recent procedure mentions: {', '.join(provisional['procedures'])}")

            parts.extend(provisional_lines)
        
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
    
    def reset_for_new_image(self):
        """Reset image-derived context when a new image is uploaded.
        
        Clears anatomy, conditions, findings, and imaging modality (which are
        typically derived from EyeCLIP or the previous image analysis) while
        preserving patient-reported context (symptoms, medications, procedures).
        """
        print(f"[State] Resetting image-derived context for new image upload")
        self.anatomy_of_interest = None
        self.primary_condition = None
        self.secondary_conditions = []
        self.clinical_findings = {}
        self.imaging_modality = None
        self.primary_region = OphthalmicRegion.UNSPECIFIED
        self.triage_priority = IndianClinicalPriority.ROUTINE
        self.topic_drift_detected = False
        self.pending_anatomy_shift = None
        self.pending_condition_shift = None
        self.pending_findings = {}
    
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