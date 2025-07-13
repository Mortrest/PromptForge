import json
import random
import time
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import re
from typing import List, Tuple, Dict, Any

# Import our custom modules
from text_to_kg_llama import text_to_kg, kg_to_text
from kg_synonym_perturbation_llama_light import synonym_perturb_kg, get_synonym_replacer

class SyntheticDataPipeline:
    def __init__(self, 
                 squad_file: str,
                 bert_model_name: str = "distilbert-base-uncased",
                 llama_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 device: str = None):
        """
        Initialize the synthetic data pipeline.
        
        Args:
            squad_file: Path to SQuAD JSON file
            bert_model_name: BERT model for QA training
            llama_model_name: Llama model for KG operations
            device: Device to run models on
        """
        self.squad_file = squad_file
        self.bert_model_name = bert_model_name
        self.llama_model_name = llama_model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load SQuAD data
        self.squad_data = self.load_squad_data()
        
        # Initialize models
        self.init_models()
        
        # Initialize synonym replacer
        self.synonym_replacer = get_synonym_replacer()
        
    def load_squad_data(self) -> List[Dict]:
        """Load and parse SQuAD data."""
        print(f"Loading SQuAD data from {self.squad_file}...")
        with open(self.squad_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Flatten the data structure
        qa_pairs = []
        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    qa_pairs.append({
                        'context': context,
                        'question': qa['question'],
                        'answer': qa['answers'][0]['text'] if qa['answers'] else '',
                        'answer_start': qa['answers'][0]['answer_start'] if qa['answers'] else 0
                    })
        
        print(f"Loaded {len(qa_pairs)} QA pairs")
        return qa_pairs
    
    def init_models(self):
        """Initialize BERT and Llama models."""
        print("Initializing models...")
        
        # Initialize BERT tokenizer and model
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        self.bert_model = AutoModelForQuestionAnswering.from_pretrained(self.bert_model_name)
        self.bert_model.to(self.device)
        
        print("Models initialized successfully!")
    
    def context_to_kg(self, context: str) -> List[Tuple[str, str, str]]:
        """Convert context to knowledge graph triples."""
        try:
            triples = text_to_kg(context, max_new_tokens=128, device=self.device)
            return triples
        except Exception as e:
            print(f"Error converting context to KG: {e}")
            return []
    
    def kg_to_synthetic_context(self, triples: List[Tuple[str, str, str]]) -> str:
        """Convert knowledge graph triples back to synthetic context."""
        try:
            if not triples:
                return ""
            context = kg_to_text(triples, max_new_tokens=256, device=self.device)
            return context
        except Exception as e:
            print(f"Error converting KG to context: {e}")
            return ""
    
    def generate_perturbed_kgs(self, triples: List[Tuple[str, str, str]], 
                             num_perturbations: int = 3) -> List[List[Tuple[str, str, str]]]:
        """Generate multiple perturbed knowledge graphs."""
        perturbed_kgs = []
        
        for i in range(num_perturbations):
            try:
                # Use different perturbation percentages for variety
                percent = random.choice([30, 50, 70, 100])
                perturbed = synonym_perturb_kg(triples, percent=percent, context_aware=True)
                perturbed_kgs.append(perturbed)
            except Exception as e:
                print(f"Error generating perturbation {i}: {e}")
                perturbed_kgs.append(triples)  # Fallback to original
        
        return perturbed_kgs
    
    def generate_qa_from_context(self, context: str, num_questions: int = 2) -> List[Dict]:
        """Generate QA pairs from synthetic context using Llama."""
        qa_pairs = []
        
        # Create prompts for different types of questions
        question_prompts = [
            f"Generate a question about this text: {context}\nQuestion:",
            f"Create a who/what/where/when question based on: {context}\nQuestion:",
            f"Form a question that can be answered from this passage: {context}\nQuestion:"
        ]
        
        for i in range(num_questions):
            try:
                # Use different prompts for variety
                prompt = random.choice(question_prompts)
                
                # Generate question using Llama
                from transformers import AutoTokenizer, AutoModelForCausalLM
                tokenizer = AutoTokenizer.from_pretrained(self.llama_model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.llama_model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
                )
                model.to(self.device)
                
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                question = tokenizer.decode(outputs[0], skip_special_tokens=True)
                question = question[len(prompt):].strip()
                
                if question and len(question) > 5:
                    # For now, use a simple answer extraction (you can improve this)
                    answer = self.extract_simple_answer(context, question)
                    
                    qa_pairs.append({
                        'context': context,
                        'question': question,
                        'answer': answer,
                        'answer_start': context.find(answer) if answer in context else 0
                    })
                
            except Exception as e:
                print(f"Error generating QA pair {i}: {e}")
                continue
        
        return qa_pairs
    
    def extract_simple_answer(self, context: str, question: str) -> str:
        """Simple answer extraction (placeholder - can be improved)."""
        # This is a simple implementation - you might want to use a more sophisticated approach
        words = context.split()
        if len(words) >= 3:
            # Return a random 3-word phrase as answer
            start_idx = random.randint(0, len(words) - 3)
            return " ".join(words[start_idx:start_idx + 3])
        return context[:50] if context else ""
    
    def generate_synthetic_data(self, contexts: List[str], 
                              num_perturbations: int = 3,
                              num_questions_per_context: int = 2) -> List[Dict]:
        """Generate synthetic QA data from contexts."""
        synthetic_data = []
        
        print(f"Generating synthetic data from {len(contexts)} contexts...")
        
        for i, context in enumerate(contexts):
            print(f"Processing context {i+1}/{len(contexts)}")
            
            # Convert context to KG
            triples = self.context_to_kg(context)
            if not triples:
                continue
            
            # Generate perturbed KGs
            perturbed_kgs = self.generate_perturbed_kgs(triples, num_perturbations)
            
            # Convert each perturbed KG to synthetic context
            for j, perturbed_triples in enumerate(perturbed_kgs):
                synthetic_context = self.kg_to_synthetic_context(perturbed_triples)
                if not synthetic_context:
                    continue
                
                # Generate QA pairs from synthetic context
                qa_pairs = self.generate_qa_from_context(synthetic_context, num_questions_per_context)
                synthetic_data.extend(qa_pairs)
                
                print(f"  Generated {len(qa_pairs)} QA pairs from perturbation {j+1}")
        
        print(f"Generated {len(synthetic_data)} total synthetic QA pairs")
        return synthetic_data
    
    def prepare_bert_data(self, qa_pairs: List[Dict]) -> Dataset:
        """Prepare data for BERT training."""
        def tokenize_function(examples):
            questions = [q.strip() for q in examples["question"]]
            contexts = [c.strip() for c in examples["context"]]
            
            tokenized = self.bert_tokenizer(
                questions,
                contexts,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # Find answer positions
            start_positions = []
            end_positions = []
            
            for i, (context, answer) in enumerate(zip(contexts, examples["answer"])):
                answer_start = context.find(answer)
                if answer_start != -1:
                    # Convert character positions to token positions
                    start_positions.append(answer_start)
                    end_positions.append(answer_start + len(answer))
                else:
                    start_positions.append(0)
                    end_positions.append(0)
            
            tokenized["start_positions"] = start_positions
            tokenized["end_positions"] = end_positions
            
            return tokenized
        
        # Convert to dataset
        dataset = Dataset.from_list(qa_pairs)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def train_and_evaluate_bert(self, train_data: Dataset, eval_data: Dataset, 
                               model_name: str = "baseline") -> Dict[str, float]:
        """Train BERT model and evaluate performance."""
        print(f"Training {model_name} BERT model...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./bert_qa_{model_name}",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.bert_tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.bert_model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.bert_tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate()
        
        print(f"{model_name} Results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        return results
    
    def run_complete_pipeline(self, 
                            num_contexts: int = 10,
                            num_perturbations: int = 3,
                            num_questions_per_context: int = 2,
                            train_split_ratio: float = 0.8) -> Dict[str, Any]:
        """Run the complete synthetic data generation and evaluation pipeline."""
        
        print("=" * 60)
        print("SYNTHETIC DATA GENERATION PIPELINE")
        print("=" * 60)
        
        # Step 1: Select contexts from SQuAD
        selected_contexts = random.sample(
            [qa['context'] for qa in self.squad_data], 
            min(num_contexts, len(self.squad_data))
        )
        
        # Step 2: Generate synthetic data
        synthetic_qa_pairs = self.generate_synthetic_data(
            selected_contexts, 
            num_perturbations, 
            num_questions_per_context
        )
        
        # Step 3: Prepare datasets
        # Split original SQuAD data
        random.shuffle(self.squad_data)
        split_idx = int(len(self.squad_data) * train_split_ratio)
        train_squad = self.squad_data[:split_idx]
        eval_squad = self.squad_data[split_idx:]
        
        # Prepare datasets
        train_baseline = self.prepare_bert_data(train_squad)
        eval_dataset = self.prepare_bert_data(eval_squad)
        
        # Combine original + synthetic for enhanced training
        train_enhanced = train_squad + synthetic_qa_pairs
        train_enhanced_dataset = self.prepare_bert_data(train_enhanced)
        
        # Step 4: Train and evaluate baseline model
        print("\nTraining baseline model (SQuAD only)...")
        baseline_results = self.train_and_evaluate_bert(
            train_baseline, eval_dataset, "baseline"
        )
        
        # Step 5: Train and evaluate enhanced model
        print("\nTraining enhanced model (SQuAD + synthetic)...")
        enhanced_results = self.train_and_evaluate_bert(
            train_enhanced_dataset, eval_dataset, "enhanced"
        )
        
        # Step 6: Compare results
        comparison = {
            'baseline': baseline_results,
            'enhanced': enhanced_results,
            'synthetic_data_count': len(synthetic_qa_pairs),
            'original_data_count': len(train_squad),
            'improvement': {
                'eval_loss': baseline_results.get('eval_loss', 0) - enhanced_results.get('eval_loss', 0),
                'eval_accuracy': enhanced_results.get('eval_accuracy', 0) - baseline_results.get('eval_accuracy', 0)
            }
        }
        
        print("\n" + "=" * 60)
        print("FINAL COMPARISON")
        print("=" * 60)
        print(f"Original training data: {len(train_squad)} QA pairs")
        print(f"Synthetic data generated: {len(synthetic_qa_pairs)} QA pairs")
        print(f"Enhanced training data: {len(train_enhanced)} QA pairs")
        print(f"\nBaseline eval loss: {baseline_results.get('eval_loss', 0):.4f}")
        print(f"Enhanced eval loss: {enhanced_results.get('eval_loss', 0):.4f}")
        print(f"Improvement in loss: {comparison['improvement']['eval_loss']:.4f}")
        
        return comparison

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SyntheticDataPipeline(
        squad_file="squad_first_1000.jsonl",  # Adjust path as needed
        bert_model_name="distilbert-base-uncased",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        num_contexts=5,  # Start with small number for testing
        num_perturbations=2,
        num_questions_per_context=1
    )
    
    # Save results
    with open("pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to pipeline_results.json") 