import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.generator import MedGemmaGenerator
from src.state.clinical_session_state import ClinicalSessionState

def main():
    print("Loading generator (for entity extractor)...")
    generator = MedGemmaGenerator()
    session = ClinicalSessionState(session_id="test_session")

    print("\n--- Testing Clinical Entity Extraction ---")
    
    query1 = "My eye is paining and it's kind of bleeding as well."
    print(f"\nTurn 1: {query1}")
    
    # Process turn
    print("Extracting entities...")
    new_entities = generator.entity_extractor.extract_entities(
        text=query1,
        visual_findings=None,
        turn_id=1
    )
    
    # Update session
    session.update_from_entities(
        entities=new_entities,
        current_turn=1,
        text=query1
    )
    
    print("\n--- Extracted Entities (Turn 1) ---")
    for e in new_entities:
        print(f"[{e.entity_type.name}] {e.text} (Conf: {e.confidence:.2f})")
        
    print("\n--- Session Context Block (Turn 1) ---")
    print(session.to_query_context(include_provisional=True))

    query2 = "What should I do?"
    print(f"\nTurn 2: {query2}")
    
    # Process turn 2 (no entities, just seeing context)
    new_entities2 = generator.entity_extractor.extract_entities(
        text=query2,
        visual_findings=None,
        turn_id=2
    )
    session.update_from_entities(
        entities=new_entities2,
        current_turn=2,
        text=query2
    )
    
    print("\n--- Session Context Block (Turn 2) ---")
    print(session.to_query_context(include_provisional=True))

if __name__ == "__main__":
    main()
