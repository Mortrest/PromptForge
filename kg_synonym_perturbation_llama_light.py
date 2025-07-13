import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import re
import time

# Use a smaller model for better performance
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Smaller alternative
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # Original choice

class LightweightLlamaSynonymReplacer:
    def __init__(self, device=None, max_new_tokens=16):
        """
        Initialize the lightweight Llama-based synonym replacer.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            max_new_tokens: Maximum tokens to generate for synonym suggestions
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_new_tokens = max_new_tokens
        
        # Initialize model and tokenizer
        print(f"Loading Llama model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Use 8-bit quantization for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            load_in_8bit=True if self.device == 'cuda' else False,
            device_map="auto" if self.device == 'cuda' else None
        )
        
        if self.device == 'cpu':
            self.model.to(self.device)
        
        print("Model loaded successfully!")
        
        # Cache for storing synonym results to avoid repeated queries
        self.synonym_cache = {}
        
    def get_synonym_with_context(self, word, context=""):
        """
        Get a contextually appropriate synonym for a word using Llama.
        
        Args:
            word: The word to find synonyms for
            context: Optional context to help determine appropriate synonyms
            
        Returns:
            A synonym if found, otherwise the original word
        """
        # Check cache first
        cache_key = f"{word.lower()}_{context.lower()}"
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key]
        
        # Create optimized, shorter prompt
        if context:
            prompt = f"Context: {context}\nWord: {word}\nSynonym:"
        else:
            prompt = f"Word: {word}\nSynonym:"
        
        try:
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=0.1,
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (after the prompt)
            generated_text = response[len(prompt):].strip()
            
            # Clean up the response
            synonym = generated_text.split('\n')[0].strip()
            
            # Check if no suitable synonym was found or if it's the same word
            if not synonym or synonym == word or len(synonym) < 2:
                result = word
            else:
                # Clean up the synonym (remove quotes, extra punctuation)
                synonym = re.sub(r'^["\']|["\']$', '', synonym)
                synonym = synonym.strip()
                result = synonym if synonym and synonym != word else word
            
            # Cache the result
            self.synonym_cache[cache_key] = result
            
            # Minimal delay
            time.sleep(0.05)
            
            return result
            
        except Exception as e:
            print(f"Error getting synonym for '{word}': {e}")
            return word
    
    def get_synonym(self, word):
        """
        Get a synonym for a word without context.
        
        Args:
            word: The word to find synonyms for
            
        Returns:
            A synonym if found, otherwise the original word
        """
        return self.get_synonym_with_context(word)
    
    def clear_cache(self):
        """Clear the synonym cache."""
        self.synonym_cache.clear()

# Global instance for reuse
_synonym_replacer = None

def get_synonym_replacer():
    """Get or create the global synonym replacer instance."""
    global _synonym_replacer
    if _synonym_replacer is None:
        _synonym_replacer = LightweightLlamaSynonymReplacer()
    return _synonym_replacer

def get_synonym(word, context=""):
    """
    Return a contextually appropriate synonym for the word using Llama, 
    or the original word if no suitable synonym is found.
    
    Args:
        word: The word to find synonyms for
        context: Optional context to help determine appropriate synonyms
        
    Returns:
        A synonym if found, otherwise the original word
    """
    replacer = get_synonym_replacer()
    return replacer.get_synonym_with_context(word, context)

def synonym_perturb_kg(triples, percent=100, context_aware=True):
    """
    Given a list of (subject, relation, object) triples, replace nodes with synonyms in N% of the triples.
    
    Args:
        triples: List of (subject, relation, object) triples
        percent: Integer from 0 to 100, percent of triples to perturb
        context_aware: Whether to use context-aware synonym replacement
        
    Returns:
        List of perturbed triples
    """
    n = len(triples)
    k = int(n * percent / 100)
    indices = set(random.sample(range(n), k)) if k > 0 else set()
    perturbed = []
    
    for i, (s, r, o) in enumerate(triples):
        if i in indices:
            if context_aware:
                # Use relation as context for better synonym selection
                s_context = f"subject in relation '{r}'"
                o_context = f"object in relation '{r}'"
                s_syn = get_synonym(s, s_context)
                o_syn = get_synonym(o, o_context)
            else:
                s_syn = get_synonym(s)
                o_syn = get_synonym(o)
            
            perturbed.append((s_syn, r, o_syn))
        else:
            perturbed.append((s, r, o))
    
    return perturbed

# Example usage and testing
if __name__ == "__main__":
    # Test the synonym replacement
    test_triples = [
        ("cat", "eats", "fish"),
        ("person", "lives_in", "house"),
        ("car", "drives_on", "road")
    ]
    
    print("Original triples:")
    for triple in test_triples:
        print(triple)
    
    print("\nPerturbed triples (50%):")
    perturbed = synonym_perturb_kg(test_triples, percent=50, context_aware=True)
    for triple in perturbed:
        print(triple) 