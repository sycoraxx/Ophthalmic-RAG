import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import QueryEngine
from src.state.clinical_session_state import ClinicalSessionState
from src.state.clinical_entity_extractor import OphthalmicRegion, IndianClinicalPriority

def test_indian_localization():
    engine = QueryEngine()
    session_id = "test_indian_ctx"
    
    print("\n--- SCENARIO 1: Posterior Segment / Urgent (DR) ---")
    # Simulate a diabetic retinopathy case
    answer1 = "The OCT scan shows significant macular edema and retinal hemorrhages. Given your history of diabetes, this indicates proliferative diabetic retinopathy."
    visual_findings = "Detected Image Type: OCT\n● Probable: diabetic retinopathy (85.2%)\n● Probable: macular edema (72.1%)"
    
    session = engine._get_or_create_session(session_id)
    current_turn = 1
    
    entities = engine.generator.extract_entities_from_answer(
        answer=answer1,
        visual_findings=visual_findings,
        turn_id=current_turn
    )
    
    print("Extracted Entities & Metadata:")
    for e in entities:
        print(f"  - {e.text} | Region: {e.region.name} | Priority: {e.priority.name}")
        
    session.update_from_entities(entities, current_turn, text=answer1)
    
    print(f"\nAggregated Session Metadata:")
    print(f"  Primary Region: {session.primary_region.name}")
    print(f"  Triage Priority: {session.triage_priority.name}")
    print(f"  Generation Context Block:\n{session.to_generation_context()}")

    print("\n--- SCENARIO 2: Anterior Segment / Emergency (Angle Closure) ---")
    session2 = engine._get_or_create_session("test_emergency")
    answer2 = "The patient presents with sudden severe eye pain, redness, and a fixed mid-dilated pupil. This is highly suggestive of acute angle closure glaucoma."
    
    entities2 = engine.generator.extract_entities_from_answer(
        answer=answer2,
        visual_findings=None,
        turn_id=1
    )
    
    print("Extracted Entities & Metadata:")
    for e in entities2:
        print(f"  - {e.text} | Region: {e.region.name} | Priority: {e.priority.name}")
        
    session2.update_from_entities(entities2, 1, text=answer2)
    
    print(f"\nAggregated Session Metadata:")
    print(f"  Primary Region: {session2.primary_region.name}")
    print(f"  Triage Priority: {session2.triage_priority.name}")
    print(f"  Priority Level: {session2.triage_priority.value.upper()}")

    print("\n--- SCENARIO 3: Blindness Category ---")
    session3 = engine._get_or_create_session("test_blindness")
    answer3 = "The patient's best corrected visual acuity is 3/60 in the better eye, which corresponds to economic blindness as per NPCB guidelines."
    
    entities3 = engine.generator.extract_entities_from_answer(
        answer=answer3,
        visual_findings=None,
        turn_id=1
    )
    
    session3.update_from_entities(entities3, 1, text=answer3)
    print(f"  Blindness Category: {session3.blindness_category}")

if __name__ == "__main__":
    test_indian_localization()
