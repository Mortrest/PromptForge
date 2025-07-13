

## Pipeline Overview

```
SQuAD Contexts → KG Generation → KG Perturbation → Synthetic Contexts → QA Generation → BERT Training & Evaluation
```

## Key Features

- **Context-aware synonym replacement** using Llama 3
- **Complete evaluation pipeline** comparing baseline vs enhanced BERT models
- **Flexible configuration** with predefined experiments
- **Quality control** with fallback mechanisms
- **Performance optimization** with caching and quantization 


### Key Components

1. **Context Selection**: Select K contexts from SQuAD dataset
2. **KG Generation**: Convert contexts to knowledge graph triples using Llama 3
3. **KG Perturbation**: Create M perturbed KGs per context using Llama-based synonym replacement
4. **Synthetic Context Generation**: Convert perturbed KGs back to text using Llama 3
5. **QA Generation**: Generate QA pairs from synthetic contexts
6. **BERT Training & Evaluation**: 
   - **Baseline**: Train BERT on subset of SQuAD → Evaluate
   - **Enhanced**: Train BERT on subset of SQuAD + synthetic data → Evaluate
   - **Comparison**: Compare performance between baseline and enhanced models

## File Structure

```
KGSynth/
├── synthetic_data_pipeline.py      # Main pipeline implementation
├── run_synthetic_pipeline.py       # Execution script with CLI
├── pipeline_config.py              # Configuration management
├── text_to_kg_llama.py            # Context ↔ KG conversion
├── kg_synonym_perturbation_llama_light.py  # Llama-based synonym replacement
├── kg_synonym_perturbation.py     # Original WordNet implementation
├── requirements_llama_synonyms.txt # Dependencies
├── README_complete_pipeline.md     # This file
└── squad_first_1000.jsonl         # SQuAD dataset (not included)
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_llama_synonyms.txt

# Additional dependencies for the complete pipeline
pip install datasets scikit-learn
```

### 2. Basic Usage

```bash
# Run with default configuration
python run_synthetic_pipeline.py

# Run with predefined experiment
python run_synthetic_pipeline.py --experiment small_scale

# Run with custom parameters
python run_synthetic_pipeline.py \
    --num_contexts 10 \
    --num_perturbations 3 \
    --num_questions 2 \
    --epochs 3
```

### 3. Advanced Usage

```bash
# High-quality experiment with more conservative perturbations
python run_synthetic_pipeline.py --experiment high_quality

# Large-scale experiment
python run_synthetic_pipeline.py --experiment large_scale

# Custom model and device
python run_synthetic_pipeline.py \
    --bert_model bert-base-uncased \
    --llama_model meta-llama/Llama-2-7b-chat-hf \
    --device cuda
```

## Configuration

### Predefined Experiments

| Experiment | Contexts (K) | Perturbations (M) | Questions | Epochs | Use Case |
|------------|--------------|-------------------|-----------|--------|----------|
| `fast_experiment` | 3 | 1 | 1 | 1 | Quick testing |
| `small_scale` | 5 | 2 | 1 | 2 | Development |
| `medium_scale` | 20 | 3 | 2 | 3 | Research |
| `large_scale` | 50 | 5 | 3 | 5 | Production |
| `high_quality` | 30 | 4 | 2 | 3 | Quality-focused |

### Custom Configuration

You can modify `pipeline_config.py` to customize:

```python
PIPELINE_CONFIG = {
    'squad_file': 'squad_first_1000.jsonl',
    'bert_model_name': 'distilbert-base-uncased',
    'llama_model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'num_contexts': 10,
    'num_perturbations': 3,
    'num_questions_per_context': 2,
    'train_split_ratio': 0.8,
    'learning_rate': 2e-5,
    'num_train_epochs': 3,
    # ... more options
}
```

## Expected Output

### Pipeline Execution Summary

```
============================================================
SYNTHETIC DATA GENERATION PIPELINE
============================================================
SQuAD file: squad_first_1000.jsonl
BERT model: distilbert-base-uncased
Llama model: meta-llama/Meta-Llama-3-8B-Instruct
Number of contexts: 10
Perturbations per context: 3
Questions per context: 2
Expected synthetic data: 60
Training epochs: 3
Context-aware perturbation: True
============================================================

Generating synthetic data from 10 contexts...
Processing context 1/10
  Generated 2 QA pairs from perturbation 1
  Generated 2 QA pairs from perturbation 2
  Generated 2 QA pairs from perturbation 3
...

Training baseline model (SQuAD only)...
Training enhanced model (SQuAD + synthetic)...

============================================================
FINAL COMPARISON
============================================================
Original training data: 800 QA pairs
Synthetic data generated: 60 QA pairs
Enhanced training data: 860 QA pairs

Baseline eval loss: 2.3456
Enhanced eval loss: 2.1234
Improvement in loss: 0.2222
```

### Results File (`pipeline_results.json`)

```json
{
  "baseline": {
    "eval_loss": 2.3456,
    "eval_accuracy": 0.7234
  },
  "enhanced": {
    "eval_loss": 2.1234,
    "eval_accuracy": 0.7567
  },
  "synthetic_data_count": 60,
  "original_data_count": 800,
  "improvement": {
    "eval_loss": 0.2222,
    "eval_accuracy": 0.0333
  },
  "execution_time": 1245.67,
  "config": {...},
  "timestamp": "2024-01-15T10:30:45.123456"
}
```
