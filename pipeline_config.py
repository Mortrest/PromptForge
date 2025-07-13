
# Pipeline Configuration
PIPELINE_CONFIG = {
    # Data parameters
    'squad_file': 'squad_first_1000.jsonl',  # Path to SQuAD data file
    'train_split_ratio': 0.8,  # Ratio of data to use for training (rest for evaluation)
    
    # Model parameters
    'bert_model_name': 'distilbert-base-uncased',  # BERT model for QA training
    'llama_model_name': 'meta-llama/Meta-Llama-3-8B-Instruct',  # Llama model for KG operations
    
    # Synthetic data generation parameters
    'num_contexts': 10,  # Number of contexts to select from SQuAD (K)
    'num_perturbations': 3,  # Number of perturbations per context (M)
    'num_questions_per_context': 2,  # Number of questions to generate per synthetic context
    
    # KG perturbation parameters
    'perturbation_percentages': [30, 50, 70, 100],  # Different perturbation levels
    'context_aware_perturbation': True,  # Use context-aware synonym replacement
    
    # BERT training parameters
    'learning_rate': 2e-5,
    'num_train_epochs': 3,
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'weight_decay': 0.01,
    
    # Device configuration
    'device': None,  # None for auto-detection, or 'cuda'/'cpu'
    'use_8bit_quantization': True,  # Use 8-bit quantization for memory efficiency
    
    # Output configuration
    'save_results': True,
    'results_file': 'pipeline_results.json',
    'save_synthetic_data': True,
    'synthetic_data_file': 'synthetic_qa_data.json',
    
    # Logging configuration
    'verbose': True,
    'log_level': 'INFO',
}

# Experiment configurations for different scenarios
EXPERIMENT_CONFIGS = {
    'small_scale': {
        'num_contexts': 5,
        'num_perturbations': 2,
        'num_questions_per_context': 1,
        'num_train_epochs': 2,
    },
    
    'medium_scale': {
        'num_contexts': 20,
        'num_perturbations': 3,
        'num_questions_per_context': 2,
        'num_train_epochs': 3,
    },
    
    'large_scale': {
        'num_contexts': 50,
        'num_perturbations': 5,
        'num_questions_per_context': 3,
        'num_train_epochs': 5,
    },
    
    'high_quality': {
        'num_contexts': 30,
        'num_perturbations': 4,
        'num_questions_per_context': 2,
        'context_aware_perturbation': True,
        'perturbation_percentages': [25, 40, 60, 80],  # More conservative perturbations
    },
    
    'fast_experiment': {
        'num_contexts': 3,
        'num_perturbations': 1,
        'num_questions_per_context': 1,
        'num_train_epochs': 1,
        'per_device_train_batch_size': 4,
    }
}

# Model-specific configurations
MODEL_CONFIGS = {
    'distilbert-base-uncased': {
        'max_length': 512,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
    },
    
    'bert-base-uncased': {
        'max_length': 512,
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
    },
    
    'distilbert-qa-small': {
        'max_length': 384,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
    }
}

# Question generation prompts
QUESTION_PROMPTS = [
    "Generate a question about this text: {context}\nQuestion:",
    "Create a who/what/where/when question based on: {context}\nQuestion:",
    "Form a question that can be answered from this passage: {context}\nQuestion:",
    "What question can be asked about this information: {context}\nQuestion:",
    "Ask a question that tests understanding of: {context}\nQuestion:",
]

# Answer extraction strategies
ANSWER_EXTRACTION_STRATEGIES = {
    'simple': 'extract_simple_answer',
    'keyword_based': 'extract_keyword_answer',
    'llama_based': 'extract_llama_answer',
}

def get_config(experiment_name: str = None) -> dict:
    """
    Get configuration for the pipeline.
    
    Args:
        experiment_name: Name of predefined experiment config, or None for default
        
    Returns:
        Configuration dictionary
    """
    config = PIPELINE_CONFIG.copy()
    
    if experiment_name and experiment_name in EXPERIMENT_CONFIGS:
        config.update(EXPERIMENT_CONFIGS[experiment_name])
    
    return config

def validate_config(config: dict) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'squad_file', 'bert_model_name', 'llama_model_name',
        'num_contexts', 'num_perturbations', 'num_questions_per_context'
    ]
    
    for field in required_fields:
        if field not in config:
            print(f"Missing required field: {field}")
            return False
    
    if config['num_contexts'] <= 0:
        print("num_contexts must be positive")
        return False
    
    if config['num_perturbations'] <= 0:
        print("num_perturbations must be positive")
        return False
    
    if config['num_questions_per_context'] <= 0:
        print("num_questions_per_context must be positive")
        return False
    
    if not (0 < config['train_split_ratio'] < 1):
        print("train_split_ratio must be between 0 and 1")
        return False
    
    return True

def print_config_summary(config: dict):
    """Print a summary of the configuration."""
    print("=" * 50)
    print("PIPELINE CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"SQuAD file: {config['squad_file']}")
    print(f"BERT model: {config['bert_model_name']}")
    print(f"Llama model: {config['llama_model_name']}")
    print(f"Number of contexts: {config['num_contexts']}")
    print(f"Perturbations per context: {config['num_perturbations']}")
    print(f"Questions per context: {config['num_questions_per_context']}")
    print(f"Expected synthetic data: {config['num_contexts'] * config['num_perturbations'] * config['num_questions_per_context']}")
    print(f"Training epochs: {config['num_train_epochs']}")
    print(f"Context-aware perturbation: {config['context_aware_perturbation']}")
    print("=" * 50)

if __name__ == "__main__":
    # Test configuration
    config = get_config('small_scale')
    print_config_summary(config)
    
    if validate_config(config):
        print("Configuration is valid!")
    else:
        print("Configuration has errors!") 