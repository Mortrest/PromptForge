import json
import os
import sys
from text_to_kg_llama import text_to_kg

# Load the first 1000 SQuAD samples
with open("/home/afalahat/KGSynth/squad_first_1000.jsonl", "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

# Find the first unique context
seen_contexts = set()
unique_contexts = []
for sample in samples:
    context = sample["context"]
    if context not in seen_contexts:
        seen_contexts.add(context)
        unique_contexts.append(context)

# For each unique context, convert to KG and print the triples
for idx, context in enumerate(unique_contexts):
    print(f"Context #{idx+1} (truncated): {context[:100]}...")
    print(context)
    # triples = text_to_kg(context)
    # print("Extracted triples:")
    # for t in triples:
    #     print(t)
    print("-"*40)
    # Uncomment below to process only the first unique context
    # break 