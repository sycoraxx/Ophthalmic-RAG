import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.generator import MedGemmaGenerator

def main():
    print("Loading generator...")
    generator = MedGemmaGenerator()

    queries = [
        "I have a white spot on the black part of my eye.",
        "The white part of my eye is super red and hurting in the light.",
    ]

    print("\n--- Testing refine_query ---")
    for q in queries:
        print(f"\nRaw Query: {q}")
        refined = generator.refine_query(raw_query=q)
        print(f"Refined Query: {refined}")

    print("\n--- Testing rewrite_query_for_retrieval ---")
    current = "Will it spread?"
    history = ["I have a white spot on the black part of my eye."]
    
    # Needs a dummy session state or None
    print(f"\nRecent History: {history}")
    print(f"Current Query: {current}")
    rewritten = generator.rewrite_query_for_retrieval(
        current_query=current,
        recent_history=history,
    )
    print(f"Rewritten Query: {rewritten}")

if __name__ == "__main__":
    main()
