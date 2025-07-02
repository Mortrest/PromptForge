from kg_synonym_perturbation import synonym_perturb_kg

# You can adjust this function to generate different prompt templates as needed
def generate_prompts_from_kg(triples):
    """
    Given a list of (subject, relation, object) triples, return a list of prompt strings for generation.
    Each prompt can use a different template or phrasing.
    """
    triples_str = "\n".join([f"({s}, {r}, {o})" for s, r, o in triples])
    prompts = [
        f"Here are some knowledge graph triples (subject, relation, object):\n{triples_str}\nGenerate a coherent text that expresses the information in these triples.",
        f"Given the following triples:\n{triples_str}\nWrite a story that includes all the facts.",
        f"Use these facts to write a paragraph:\n{triples_str}",
    ]
    return prompts

def parse_triples_from_input(triples_str):
    """
    Parse user-pasted triples in the format: (subject, relation, object) per line,
    or subject, relation, object per line (with or without parentheses).
    Returns a list of (s, r, o) tuples.
    """
    import re
    triples = []
    for line in triples_str.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Try to match (subject, relation, object)
        m = re.match(r"\(([^,]+),([^,]+),([^)]+)\)", line)
        if m:
            triples.append(tuple(s.strip() for s in m.groups()))
            continue
        # Try to match subject, relation, object (no parentheses)
        parts = [p.strip() for p in line.split(',')]
        if len(parts) == 3:
            triples.append(tuple(parts))
    return triples

def main():
    print("=== KG Pipeline Demo (Manual Llama 3 Step) ===")
    context = input("Enter your context text: ")
    print("\nCopy and paste the following prompt into Llama 3 to extract triples:")
    llama_prompt = (
        "Extract knowledge graph triples (subject, relation, object) from the following text:\n"
        f"Text: {context}\n"
        "Triples:\n"
    )
    print("\n--- Prompt for Llama 3 ---\n")
    print(llama_prompt)
    print("\n-------------------------\n")
    print("Paste the output from Llama 3 (one triple per line, e.g. (subject, relation, object)):")
    user_triples = []
    print("Enter triples (end with an empty line):")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    triples = parse_triples_from_input("\n".join(lines))
    if not triples:
        print("No valid triples found. Exiting.")
        return
    perturbed_kg = synonym_perturb_kg(triples, percent=100)
    prompts = generate_prompts_from_kg(perturbed_kg)
    print("\nCandidate prompts based on perturbed KG:")
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}:\n{prompt}")

if __name__ == "__main__":
    main() 