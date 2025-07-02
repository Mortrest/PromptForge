import sys
from KGSynth.text_to_kg_llama import text_to_kg, kg_to_text
from KGSynth.kg_synonym_perturbation import synonym_perturb_kg

def kg_pipeline(context, perturb_percent=100, max_new_tokens=128):
    """
    Pipeline: context -> KG -> perturbed KG -> generated text
    Returns the generated text from the perturbed KG.
    """
    triples = text_to_kg(context, max_new_tokens=max_new_tokens)
    perturbed_kg = synonym_perturb_kg(triples, percent=perturb_percent)
    generated_text = kg_to_text(perturbed_kg, max_new_tokens=max_new_tokens)
    return generated_text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KG pipeline: context -> KG -> perturbed KG -> generated text")
    parser.add_argument('--context', type=str, required=True, help='Input context text')
    parser.add_argument('--perturb_percent', type=int, default=100, help='Percent of triples to perturb (default: 100)')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max new tokens for Llama 3 generation')
    args = parser.parse_args()

    generated_text = kg_pipeline(args.context, perturb_percent=args.perturb_percent, max_new_tokens=args.max_new_tokens)
    print("\nGenerated text from perturbed KG:")
    print(generated_text) 