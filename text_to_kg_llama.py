import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

def text_to_kg(text, max_new_tokens=128, device=None):
    """
    Given a text, use Llama 3.1 8B Instruct to extract (subject, relation, object) triples.
    Returns a list of triples.
    """
    prompt = (
        "Extract knowledge graph triples (subject, relation, object) from the following text:\n"
        f"Text: {text}\n"
        "Triples:\n"
    )
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if device=='cuda' else torch.float32)
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the part after 'Triples:'
    triples_text = generated.split("Triples:")[-1]
    triple_lines = re.findall(r"\(([^)]+)\)", triples_text)
    triples = []
    for line in triple_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            triples.append(tuple(parts))
    return triples

def kg_to_text(triples, max_new_tokens=128, device=None):
    """
    Given a list of triples, use Llama 3.1 8B Instruct to generate a full text.
    """
    triples_str = "\n".join([f"({s}, {r}, {o})" for s, r, o in triples])
    prompt = (
        "Here are some knowledge graph triples (subject, relation, object):\n"
        f"{triples_str}\n"
        "Generate a coherent text that expresses the information in these triples.\nText: "
    )
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if device=='cuda' else torch.float32)
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the part after 'Text:'
    text_out = generated.split("Text:")[-1].strip()
    return text_out

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Llama 3.1 8B Instruct KG extractor and generator")
    parser.add_argument('--mode', choices=['text_to_kg', 'kg_to_text'], required=True)
    parser.add_argument('--input', type=str, required=True, help='Input text or path to triples file')
    args = parser.parse_args()

    if args.mode == 'text_to_kg':
        text = args.input
        triples = text_to_kg(text)
        print("Extracted triples:")
        for t in triples:
            print(t)
    elif args.mode == 'kg_to_text':
        # Expecting a file with one triple per line: (subject, relation, object)
        with open(args.input, 'r') as f:
            triples = []
            for line in f:
                m = re.match(r"\(([^,]+),([^,]+),([^\)]+)\)", line.strip())
                if m:
                    triples.append(tuple(s.strip() for s in m.groups()))
        text = kg_to_text(triples)
        print("Generated text:")
        print(text) 