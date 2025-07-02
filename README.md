# KGSynth


## Features
- Extract (subject, relation, object) triples from text using LLMs
- Perturb KGs by replacing entities with synonyms (WordNet)
- Generate text from KGs
- End-to-end pipeline: context → KG → perturbed KG → text

## Basic Usage
- Fine-tune QA model: `python train.py`
- Extract triples: `python text_to_kg_llama.py --mode text_to_kg --input "Your text"`
- Generate text from triples: `python text_to_kg_llama.py --mode kg_to_text --input triples.txt`
- Full pipeline: `python kg_pipeline.py --context "Your text"`


Install dependencies:
```sh
pip install torch transformers datasets evaluate nltk
```
