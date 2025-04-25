import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import tqdm
import logging
import os
from typing import List, Tuple, Dict, Set, Optional


class Generator:
    """
    Generate diverse but relevant stories about a brave knight using 
    a combination of hard and soft prompts.
    """
    def __init__(
        self, 
        model_name: str = "gpt2-medium",
        soft_prompt_length: int = 10,
        learning_rate: float = 0.01,
        batch_size: int = 10,
        max_length: int = 200,
        diversity_weight: float = 0.5,
        num_reference_stories: int = 20,
        update_frequency: int = 5,
        recent_stories_window: int = 20,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the Knight Story Generator.
        
        Args:
            model_name: Name of the pretrained language model
            soft_prompt_length: Length of the soft prompt embedding
            learning_rate: Learning rate for soft prompt optimization
            batch_size: Number of candidate stories to generate per iteration
            max_length: Maximum length of generated stories
            diversity_weight: Weight (λ) for balancing relevance and diversity
            num_reference_stories: Number of reference stories to generate initially
            update_frequency: How often to update the soft prompt (in stories)
            recent_stories_window: Number of recent stories to consider for diversity
            device: Device to run computations on
        """
        self.model_name = model_name
        self.soft_prompt_length = soft_prompt_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.diversity_weight = diversity_weight
        self.num_reference_stories = num_reference_stories
        self.update_frequency = update_frequency
        self.recent_stories_window = recent_stories_window
        self.device = device
        self.hard_prompt = "Write a story about a brave knight."
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token for GPT models that don't have it
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.embedding_dim = self.model.config.hidden_size
        print(f"Model embedding dimension: {self.embedding_dim}")
        
        # Initialize storage for reference and selected stories
        self.reference_stories = []
        self.reference_embeddings = []
        self.selected_stories = []
        self.selected_embeddings = []
        
        # First initialize the soft prompt with placeholder values
        # This will be updated later with reference story data
        self.initialize_soft_prompt_placeholder()
        
        # Generate reference stories
        self.generate_reference_stories()
        
        # Now initialize the actual soft prompt using reference stories
        self.initialize_soft_prompt()
    
    def generate_reference_stories(self):
        """
        Generate initial reference stories using just the hard prompt.
        """
        print(f"Generating {self.num_reference_stories} reference stories...")
        
        # Tokenize the hard prompt
        inputs = self.tokenizer(self.hard_prompt, return_tensors="pt").to(self.device)
        
        # Generate reference stories
        for i in range(self.num_reference_stories):
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=self.max_length,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    top_k=50,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_return_sequences=1
                )
                
                # Decode the generated story
                story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Store the story
                self.reference_stories.append(story)
                
                # Compute and store the story embedding
                embedding = self.get_story_embedding(story)
                self.reference_embeddings.append(embedding)
                
                print(f"Reference story {i+1}: {story[:100]}...")
    
    def initialize_soft_prompt_placeholder(self):
        """
        Initialize the soft prompt with placeholder values before we have reference stories.
        This will be updated later with the actual values from reference stories.
        """
        print("Initializing soft prompt with placeholder values...")
        
        # Initialize with random values
        self.soft_prompt = torch.randn(self.soft_prompt_length, self.embedding_dim, device=self.device)
        self.soft_prompt.requires_grad = True
        
        # Store initial embeddings
        self.initial_embeddings = self.soft_prompt.clone().detach()
        
        # Setup optimizer
        self.optimizer = optim.Adam([self.soft_prompt], lr=self.learning_rate * 0.5)
        
        # Initialize anchor words (used for regularization)
        self.anchor_words = [
            "honor", "adventure", "courage", "quest", "legend", "triumph",
            "journey", "battle", "hero", "destiny"
        ]
        
        # Get embeddings for anchor words
        self.anchor_embeddings = []
        for word in self.anchor_words:
            token_id = self.tokenizer.encode(" " + word, add_special_tokens=False)[0]
            with torch.no_grad():
                anchor_embedding = self.model.transformer.wte.weight[token_id].clone()
            self.anchor_embeddings.append(anchor_embedding.to(self.device))
        
        print("Initialized placeholder soft prompt")
        
    def initialize_soft_prompt(self):
        """
        Initialize the soft prompt using the average of the first k tokens from reference stories,
        where k is the soft_prompt_length.
        """
        print(f"Initializing soft prompt using first {self.soft_prompt_length} tokens from reference stories...")
        
        # Check if we have reference stories
        if len(self.reference_stories) == 0:
            print("No reference stories available for soft prompt initialization, keeping placeholder")
            return
        
        # Create a tensor to hold all token embeddings for each position
        position_token_embeddings = [[] for _ in range(self.soft_prompt_length)]
        
        # Process each reference story
        for story in self.reference_stories:
            # Tokenize the whole story
            tokens = self.tokenizer.encode(story, add_special_tokens=False)
            
            # Take only the first k tokens or as many as available
            tokens = tokens[:min(len(tokens), self.soft_prompt_length)]
            
            # Get embeddings for these tokens
            with torch.no_grad():
                for i, token_id in enumerate(tokens):
                    if i >= self.soft_prompt_length:
                        break
                    # Get embedding from model
                    token_embedding = self.model.transformer.wte.weight[token_id].clone()
                    position_token_embeddings[i].append(token_embedding)
        
        # Calculate average embedding for each position
        soft_prompt_embeds = []
        for i, embeddings in enumerate(position_token_embeddings):
            if embeddings:  # Check if we have embeddings for this position
                position_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
                soft_prompt_embeds.append(position_embedding)
            else:
                # If we don't have enough tokens from stories, use a random embedding
                print(f"Warning: Not enough tokens for position {i}, using random embedding")
                random_embedding = torch.randn(self.embedding_dim, device=self.device)
                # Scale to match other embeddings' norm
                if soft_prompt_embeds:
                    norm_factor = torch.norm(soft_prompt_embeds[0]) / torch.norm(random_embedding)
                    random_embedding = random_embedding * norm_factor
                soft_prompt_embeds.append(random_embedding)
        
        # Stack embeddings to form the soft prompt
        self.soft_prompt = torch.stack(soft_prompt_embeds, dim=0).to(self.device)
        self.soft_prompt.requires_grad = True
        
        # Store these initial embeddings for regularization
        self.initial_embeddings = self.soft_prompt.clone().detach()
        
        # Recreate optimizer with the new parameters
        self.optimizer = optim.Adam([self.soft_prompt], lr=self.learning_rate * 0.5)
        
        print(f"Initialized soft prompt using reference story token embeddings")
        
        # Visualize what tokens the soft prompt is close to
        self.visualize_soft_prompt()
    
    def get_story_embedding(self, story: str) -> torch.Tensor:
        """
        Compute an embedding for a story by averaging token embeddings.
        
        Args:
            story: The text of the story
            
        Returns:
            Embedding tensor
        """
        # Tokenize the story
        tokens = self.tokenizer(story, return_tensors="pt").to(self.device)
        
        # Get embeddings for tokens
        with torch.no_grad():
            outputs = self.model(
                tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Use the hidden states from the last layer
            last_hidden_states = outputs.hidden_states[-1]
            
            # Average over tokens to get story embedding
            # Use attention mask to ignore padding tokens
            mask = tokens.attention_mask.unsqueeze(-1)
            embedding = (last_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            
            return embedding.squeeze(0)
    
    def generate_candidate_stories(self) -> List[str]:
        """
        Generate candidate stories using the soft prompt + hard prompt.
        
        Returns:
            List of candidate stories
        """
        # Tokenize hard prompt
        hard_prompt_ids = self.tokenizer.encode(self.hard_prompt, return_tensors="pt").to(self.device)
        
        # Get model embeddings for hard prompt
        with torch.no_grad():
            hard_prompt_embeds = self.model.transformer.wte(hard_prompt_ids)
            
        # Concat soft prompt and hard prompt embeddings
        input_embeds = torch.cat([self.soft_prompt.unsqueeze(0), hard_prompt_embeds], dim=1)
        
        # Create attention mask
        attention_mask = torch.ones(1, input_embeds.size(1), dtype=torch.long, device=self.device)
        
        # Generate candidate stories
        candidates = []
        
        for _ in range(self.batch_size):
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        max_length=input_embeds.size(1) + self.max_length,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        temperature=0.8,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        return_dict_in_generate=True
                    )
                    
                    # Extract the generated token IDs (skip the initial input tokens)
                    generated_ids = outputs.sequences[0][input_embeds.size(1):]
                    
                    # Decode to text
                    story = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    if story.strip():  # Only add non-empty texts
                        candidates.append(story)
                        
            except Exception as e:
                print(f"Error generating candidate: {e}")
                
        return candidates
    
    def compute_relevance_scores(self, candidates: List[str]) -> torch.Tensor:
        """
        Compute relevance scores for candidates based on log probability.
        
        Args:
            candidates: List of candidate stories
            
        Returns:
            Tensor of relevance scores
        """
        relevance_scores = []
        
        for story in candidates:
            try:
                # Tokenize the story
                tokens = self.tokenizer(story, return_tensors="pt").to(self.device)
                
                # Compute log probability
                with torch.no_grad():
                    outputs = self.model(
                        tokens.input_ids,
                        attention_mask=tokens.attention_mask,
                        labels=tokens.input_ids,
                        return_dict=True
                    )
                    
                    # Lower loss means higher probability
                    log_prob = -outputs.loss.item()
                    relevance_scores.append(log_prob)
                    
            except Exception as e:
                print(f"Error computing relevance: {e}")
                relevance_scores.append(-float('inf'))  # Assign worst score
                
        return torch.tensor(relevance_scores, device=self.device)
    
    def compute_diversity_scores(self, candidates: List[str]) -> torch.Tensor:
        """
        Compute diversity scores for candidates based on embeddings.
        
        Args:
            candidates: List of candidate stories
            
        Returns:
            Tensor of diversity scores
        """
        # If no stories selected yet, all candidates are maximally diverse
        if not self.selected_embeddings:
            return torch.ones(len(candidates), device=self.device)
        
        # Get recent story embeddings for diversity calculation
        recent_embeddings = self.selected_embeddings[-self.recent_stories_window:]
        
        diversity_scores = []
        
        for story in candidates:
            try:
                # Compute story embedding
                embedding = self.get_story_embedding(story)
                
                # Compute similarity to all recent selected stories
                similarities = torch.stack([
                    F.cosine_similarity(embedding, selected_emb, dim=0) 
                    for selected_emb in recent_embeddings
                ])
                
                # Diversity is negative of maximum similarity
                # (lower similarity to existing stories = higher diversity)
                max_similarity = torch.max(similarities)
                diversity = -max_similarity.item()
                
                diversity_scores.append(diversity)
                
            except Exception as e:
                print(f"Error computing diversity: {e}")
                diversity_scores.append(-float('inf'))  # Assign worst score
                
        return torch.tensor(diversity_scores, device=self.device)
    
    def update_soft_prompt(self):
        """
        Update the soft prompt using contrastive loss to balance
        relevance and diversity, with added regularization and anchoring.
        """
        print("Updating soft prompt...")
        
        # Skip if we don't have both reference and selected stories
        if not self.reference_stories or not self.selected_stories:
            print("Not enough stories to update soft prompt.")
            return
        
        # Use recent selected stories for efficiency
        recent_selected = self.selected_stories[-self.recent_stories_window:]
        
        # Hyperparameters for regularization and anchoring
        reg_weight = 0.1  # Weight for regularization term
        anchor_weight = 0.2  # Weight for anchor term
        
        # Contrastive loss optimization
        for _ in range(5):  # Update for 5 steps each time
            
            # Compute loss for reference stories (we want high probability for these)
            ref_loss = 0.0
            for story in self.reference_stories:
                # Get hard prompt embeddings
                hard_prompt_ids = self.tokenizer.encode(self.hard_prompt, return_tensors="pt").to(self.device)
                hard_prompt_embeds = self.model.transformer.wte(hard_prompt_ids)
                
                # Combine with soft prompt
                input_embeds = torch.cat([self.soft_prompt.unsqueeze(0), hard_prompt_embeds], dim=1)
                
                # Create attention mask
                attention_mask = torch.ones(1, input_embeds.size(1), dtype=torch.long, device=self.device)
                
                # Run the model to get the logits for the prompt
                with torch.no_grad():
                    outputs = self.model(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    prompt_logits = outputs.logits[:, -1, :]  # Take logits from last position
                
                # Get story tokens and compute their likelihood given the prompt
                story_tokens = self.tokenizer.encode(story, return_tensors="pt").to(self.device)
                
                # For first few tokens, compute logprob
                story_logprob = 0.0
                for i in range(min(5, story_tokens.size(1))):  # Look at first 5 tokens max
                    token_id = story_tokens[0, i].item()
                    token_logprob = F.log_softmax(prompt_logits, dim=-1)[0, token_id]
                    story_logprob += token_logprob
                
                # Add negative logprob to loss (we want to maximize probability)
                ref_loss -= story_logprob
            
            # Compute loss for selected stories (we want low probability for these)
            sel_loss = 0.0
            for story in recent_selected:
                # Get hard prompt embeddings
                hard_prompt_ids = self.tokenizer.encode(self.hard_prompt, return_tensors="pt").to(self.device)
                hard_prompt_embeds = self.model.transformer.wte(hard_prompt_ids)
                
                # Combine with soft prompt
                input_embeds = torch.cat([self.soft_prompt.unsqueeze(0), hard_prompt_embeds], dim=1)
                
                # Create attention mask
                attention_mask = torch.ones(1, input_embeds.size(1), dtype=torch.long, device=self.device)
                
                # Run the model to get the logits for the prompt
                with torch.no_grad():
                    outputs = self.model(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    prompt_logits = outputs.logits[:, -1, :]  # Take logits from last position
                
                # Get story tokens and compute their likelihood given the prompt
                story_tokens = self.tokenizer.encode(story, return_tensors="pt").to(self.device)
                
                # For first few tokens, compute logprob
                story_logprob = 0.0
                for i in range(min(5, story_tokens.size(1))):  # Look at first 5 tokens max
                    token_id = story_tokens[0, i].item()
                    token_logprob = F.log_softmax(prompt_logits, dim=-1)[0, token_id]
                    story_logprob += token_logprob
                
                # Add logprob to loss (we want to minimize probability)
                sel_loss += story_logprob
            
            # 2. Regularization: Add term to keep soft prompt close to real word embeddings
            # Compute distance to initial embeddings (which are real word embeddings)
            reg_loss = torch.norm(self.soft_prompt - self.initial_embeddings, p=2)
            
            # Total loss combines all objectives
            total_loss = ref_loss + sel_loss + reg_weight * reg_loss 
            
            # Update soft prompt
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 3. Careful updates: Apply gradient clipping to prevent large updates
            torch.nn.utils.clip_grad_norm_([self.soft_prompt], max_norm=1.0)
            
            # Perform update
            self.optimizer.step()
            
            print(f"Soft prompt update step, loss: {total_loss.item():.4f} (ref: {ref_loss.item():.4f}, "
                       f"sel: {sel_loss.item():.4f}, reg: {reg_loss.item():.4f})")
    
    def generate_stories(self, num_stories: int = 100, output_file: str = "knight_stories.txt"):
        """
        Generate a collection of diverse but relevant knight stories.
        
        Args:
            num_stories: Number of stories to generate
            output_file: File to save the stories
        """
        print(f"Generating {num_stories} knight stories...")
        
        # Create progress bar
        progress_bar = tqdm.tqdm(range(num_stories), desc="Generating stories")
        
        # Main generation loop
        for i in progress_bar:
            # Generate candidate stories
            candidates = self.generate_candidate_stories()
            
            if not candidates:
                print("No valid candidates generated, retrying...")
                continue
            
            # Score candidates for relevance and diversity
            relevance_scores = self.compute_relevance_scores(candidates)
            diversity_scores = self.compute_diversity_scores(candidates)
            
            # Combine scores
            combined_scores = relevance_scores + self.diversity_weight * diversity_scores
            
            # Find best candidate
            best_idx = torch.argmax(combined_scores).item()
            best_story = candidates[best_idx]
            
            # Add to selected stories
            self.selected_stories.append(best_story)
            
            # Compute and store embedding
            best_embedding = self.get_story_embedding(best_story)
            self.selected_embeddings.append(best_embedding)
            
            # Log progress
            if (i + 1) % 10 == 0:
                print(f"Generated story {i+1}: {best_story[:100]}...")
            
            # Update progress bar
            progress_bar.set_postfix({
                "Relevance": f"{relevance_scores[best_idx]:.4f}",
                "Diversity": f"{diversity_scores[best_idx]:.4f}"
            })
            
            # Update soft prompt periodically
            if (i + 1) % self.update_frequency == 0:
                self.update_soft_prompt()
        
        print(f"Saving {len(self.selected_stories)} stories to {output_file}...")
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, story in enumerate(self.selected_stories):
                    f.write(f"=== Story {i+1} ===\n")
                    f.write(story + '\n\n')
                    
            print(f"Successfully saved stories to {output_file}")
            
        except Exception as e:
            print(f"Error saving stories: {e}")
    
    def visualize_soft_prompt(self):
        """
        Visualize the soft prompt by finding the closest words in vocabulary.
        """
        print("\n=== Soft Prompt Visualization ===")
        
        for i, embedding in enumerate(self.soft_prompt):
            closest_words = self.find_closest_word_in_vocabulary(embedding, top_k=5)
            print(f"Position {i+1}: {', '.join(closest_words)}")
    
    def find_closest_word_in_vocabulary(self, embedding, top_k=5):
        """
        Find the closest word in vocabulary to the given embedding.
        
        Args:
            embedding: Embedding vector to match
            top_k: Number of closest matches to return
            
        Returns:
            List of closest word matches
        """
        with torch.no_grad():
            embedding_matrix = self.model.transformer.wte.weight
            
            embedding = embedding.to(embedding_matrix.device)
            cos_similarities = F.cosine_similarity(embedding.unsqueeze(0), embedding_matrix, dim=1)
            
            top_indices = torch.argsort(cos_similarities, descending=True)[:top_k].tolist()
            
            closest_words = []
            for idx in top_indices:
                token = self.tokenizer.decode([idx]).strip()
                closest_words.append(token)
            
            return closest_words


if __name__ == "__main__":
    try:
        generator = Generator(
            model_name="gpt2-medium",
            soft_prompt_length=5,  # Slightly shorter for more focused prompts
            batch_size=10,
            max_length=300,
            diversity_weight=0.7,
            num_reference_stories=20,
            update_frequency=5,
            learning_rate=0.005  # Reduced learning rate for careful updates
        )
        
        generator.generate_stories(
            num_stories=50,  # Reduced for testing
            output_file="outputs/knight_stories.txt"
        )
        
        print("\n=== Final Soft Prompt ===")
        generator.visualize_soft_prompt()
        
        print("\nSuccessfully generated knight stories!")
        
    except Exception as e:
        import traceback
        print(f"Error running Knight Story Generator: {e}")
        traceback.print_exc()