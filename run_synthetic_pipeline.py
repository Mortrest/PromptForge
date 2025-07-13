
import argparse
import json
import sys
import os
import time
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synthetic_data_pipeline import SyntheticDataPipeline
from pipeline_config import get_config, validate_config, print_config_summary

def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Synthetic Data Generation Pipeline')
    
    parser.add_argument(
        '--experiment', 
        type=str, 
        default=None,
        choices=['small_scale', 'medium_scale', 'large_scale', 'high_quality', 'fast_experiment'],
        help='Predefined experiment configuration'
    )
    
    parser.add_argument(
        '--squad_file', 
        type=str, 
        default=None,
        help='Path to SQuAD data file'
    )
    
    parser.add_argument(
        '--num_contexts', 
        type=int, 
        default=None,
        help='Number of contexts to process (K)'
    )
    
    parser.add_argument(
        '--num_perturbations', 
        type=int, 
        default=None,
        help='Number of perturbations per context (M)'
    )
    
    parser.add_argument(
        '--num_questions', 
        type=int, 
        default=None,
        help='Number of questions per synthetic context'
    )
    
    parser.add_argument(
        '--bert_model', 
        type=str, 
        default=None,
        help='BERT model name for QA training'
    )
    
    parser.add_argument(
        '--llama_model', 
        type=str, 
        default=None,
        help='Llama model name for KG operations'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run models on'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=None,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--dry_run', 
        action='store_true',
        help='Run configuration validation only'
    )
    
    parser.add_argument(
        '--save_synthetic', 
        action='store_true',
        help='Save synthetic data to file'
    )
    
    return parser.parse_args()

def update_config_with_args(config: dict, args) -> dict:
    """Update configuration with command line arguments."""
    if args.squad_file:
        config['squad_file'] = args.squad_file
    
    if args.num_contexts:
        config['num_contexts'] = args.num_contexts
    
    if args.num_perturbations:
        config['num_perturbations'] = args.num_perturbations
    
    if args.num_questions:
        config['num_questions_per_context'] = args.num_questions
    
    if args.bert_model:
        config['bert_model_name'] = args.bert_model
    
    if args.llama_model:
        config['llama_model_name'] = args.llama_model
    
    if args.device:
        config['device'] = args.device
    
    if args.epochs:
        config['num_train_epochs'] = args.epochs
    
    if args.save_synthetic:
        config['save_synthetic_data'] = True
    
    return config

def main():
    """Main execution function."""
    print("=" * 70)
    print("SYNTHETIC DATA GENERATION PIPELINE")
    print("=" * 70)
    
    # Parse arguments
    args = parse_arguments()
    
    # Get configuration
    config = get_config(args.experiment)
    config = update_config_with_args(config, args)
    
    # Setup logging
    setup_logging(config.get('log_level', 'INFO'))
    logger = logging.getLogger(__name__)
    
    # Print configuration summary
    print_config_summary(config)
    
    # Validate configuration
    if not validate_config(config):
        logger.error("Configuration validation failed!")
        sys.exit(1)
    
    # Dry run mode
    if args.dry_run:
        logger.info("Dry run completed successfully!")
        return
    
    # Check if SQuAD file exists
    if not os.path.exists(config['squad_file']):
        logger.error(f"SQuAD file not found: {config['squad_file']}")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        logger.info("Initializing synthetic data pipeline...")
        pipeline = SyntheticDataPipeline(
            squad_file=config['squad_file'],
            bert_model_name=config['bert_model_name'],
            llama_model_name=config['llama_model_name'],
            device=config['device']
        )
        
        # Run complete pipeline
        logger.info("Starting pipeline execution...")
        start_time = time.time()
        
        results = pipeline.run_complete_pipeline(
            num_contexts=config['num_contexts'],
            num_perturbations=config['num_perturbations'],
            num_questions_per_context=config['num_questions_per_context'],
            train_split_ratio=config['train_split_ratio']
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Add execution metadata
        results['execution_time'] = execution_time
        results['config'] = config
        results['timestamp'] = datetime.now().isoformat()
        
        # Save results
        if config.get('save_results', True):
            results_file = config.get('results_file', 'pipeline_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")
        
        # Print final summary
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETED")
        print("=" * 70)
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Original data: {results['original_data_count']} QA pairs")
        print(f"Synthetic data: {results['synthetic_data_count']} QA pairs")
        print(f"Total enhanced data: {results['original_data_count'] + results['synthetic_data_count']} QA pairs")
        
        if 'improvement' in results:
            print(f"\nPerformance Improvement:")
            print(f"  Loss improvement: {results['improvement']['eval_loss']:.4f}")
            print(f"  Accuracy improvement: {results['improvement']['eval_accuracy']:.4f}")
        
        print(f"\nResults saved to: {config.get('results_file', 'pipeline_results.json')}")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)

def run_quick_test():
    """Run a quick test to verify the pipeline works."""
    print("Running quick test...")
    
    config = get_config('fast_experiment')
    config['squad_file'] = 'squad_first_1000.jsonl'  # Ensure this file exists
    
    if not os.path.exists(config['squad_file']):
        print(f"Error: SQuAD file not found: {config['squad_file']}")
        print("Please ensure the SQuAD data file is available.")
        return False
    
    try:
        pipeline = SyntheticDataPipeline(
            squad_file=config['squad_file'],
            bert_model_name=config['bert_model_name'],
            device='cpu'  # Use CPU for quick test
        )
        
        # Run with minimal parameters
        results = pipeline.run_complete_pipeline(
            num_contexts=2,
            num_perturbations=1,
            num_questions_per_context=1,
            train_split_ratio=0.8
        )
        
        print("Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False

if __name__ == "__main__":
    # Check if this is a test run
    if len(sys.argv) == 1:
        print("No arguments provided. Running quick test...")
        run_quick_test()
    else:
        main() 