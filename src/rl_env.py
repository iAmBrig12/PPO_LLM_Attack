import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import random

class LLMAdversarialEnv(gym.Env):
    """
    Environment for training an RL agent to generate adversarial prompts against LLMs.
    """
    
    def __init__(self, 
                 target_model_name="distilbert-base-uncased-finetuned-sst-2-english",
                 model_type="classification",  # "classification" or "generation"
                 task="sentiment",
                 dataset_name="sst2",
                 max_tokens=50,
                 max_steps=5,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the adversarial environment.
        
        Args:
            target_model_name: The name/path of the target model
            model_type: Type of model (classification or generation)
            task: The task to perform (sentiment, paraphrase, classification)
            dataset_name: Name of the dataset to use
            max_tokens: Maximum number of tokens in the prompt
            max_steps: Maximum number of interaction steps
            device: Device to run the model on
        """
        super(LLMAdversarialEnv, self).__init__()
        
        self.target_model_name = target_model_name
        self.model_type = model_type
        self.task = task
        self.dataset_name = dataset_name
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.current_step = 0
        self.device = device
        
        # Set up tokenizer for the action space
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_size = self.tokenizer.vocab_size
        
        # Load the target model and tokenizer
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        
        if model_type == "classification":
            self.target_model = AutoModelForSequenceClassification.from_pretrained(target_model_name).to(device)
        elif model_type == "generation":
            self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Define action and observation spaces
        # Action space: modification operations on the prompt
        # 1. Replace token
        # 2. Insert token
        # 3. Delete token
        # 4. Swap adjacent tokens
        self.action_space = spaces.Dict({
            'operation': spaces.Discrete(4),  # 0:replace, 1:insert, 2:delete, 3:swap
            'position': spaces.Discrete(max_tokens),
            'token_id': spaces.Discrete(self.vocab_size)
        })
        
        # Observation space: tokenized prompt and model response
        self.observation_space = spaces.Dict({
            'prompt_tokens': spaces.Box(low=0, high=self.vocab_size, shape=(max_tokens,), dtype=np.int32),
            'response_embedding': spaces.Box(low=-10, high=10, shape=(768,), dtype=np.float32),
            'original_label': spaces.Discrete(2),  # Binary for sentiment/paraphrase
            'current_label': spaces.Discrete(2)
        })
        
        self.current_prompt = None
        self.original_label = None
        self.original_text = None
        
    def _get_model_response(self, prompt):
        """Get response from the target LLM."""
        inputs = self.target_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if self.model_type == "classification":
                outputs = self.target_model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
                return predicted_class
            else:  # generation
                generated_ids = self.target_model.generate(
                    inputs.input_ids, 
                    max_length=50, 
                    do_sample=True,
                    top_p=0.95
                )
                generated_text = self.target_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                return generated_text
    
    def _classify_response(self, response):
        """Classify the model's response based on the task."""
        if self.model_type == "classification":
            # For classification models, the response is already the class
            return response
        else:  # For generation models
            if self.task == "sentiment":
                # Simple heuristic for sentiment in generated text
                positive_words = ["good", "great", "excellent", "positive"]
                negative_words = ["bad", "terrible", "awful", "negative"]
                
                response_lower = response.lower()
                positive_count = sum(word in response_lower for word in positive_words)
                negative_count = sum(word in response_lower for word in negative_words)
                
                return 1 if positive_count > negative_count else 0
            
            elif self.task == "paraphrase":
                # For paraphrase detection with generation models
                return int("similar" in response.lower() or "same" in response.lower())
            
            else:
                # Default binary classification
                return 0
    
    def _calculate_reward(self, current_label):
        """Calculate reward based on whether the attack was successful."""
        # Reward is higher if we flip the label
        if current_label != self.original_label:
            return 10.0  # Successfully changed the classification
        
        # Small negative reward for unsuccessful attempts
        return -1.0
    
    def _modify_prompt(self, prompt, action):
        """Apply modification to the prompt based on the action."""
        tokens = self.tokenizer.tokenize(prompt)
        operation = action['operation']
        position = min(action['position'], len(tokens) - 1) if tokens else 0
        new_token_id = action['token_id']
        new_token = self.tokenizer.convert_ids_to_tokens([new_token_id])[0]
        
        if operation == 0:  # Replace
            if tokens:
                tokens[position] = new_token
        elif operation == 1:  # Insert
            tokens.insert(min(position, len(tokens)), new_token)
        elif operation == 2:  # Delete
            if tokens and len(tokens) > 1:  # Ensure we don't delete the only token
                tokens.pop(position)
        elif operation == 3:  # Swap
            if tokens and position < len(tokens) - 1:
                tokens[position], tokens[position + 1] = tokens[position + 1], tokens[position]
        
        modified_prompt = self.tokenizer.convert_tokens_to_string(tokens)
        return modified_prompt
    
    def _get_embedding(self, response):
        """Get embedding for the response."""
        if isinstance(response, int):
            # For classification outputs, create a one-hot embedding
            embedding = np.zeros(768)
            embedding[response] = 1.0
            return embedding
        else:
            # For text responses, get sentence embedding
            # Using mean pooling of token embeddings as a simple approach
            inputs = self.target_tokenizer(
                response, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=128, 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.target_model.base_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            # Normalize the embedding
            norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
            normalized_embedding = embeddings / norm
            
            return normalized_embedding.cpu().numpy()[0]
    
    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation."""
        super().reset(seed=seed)
        
        # Select a random example from a predefined set
        # In practice, you would load examples from the actual dataset
        sample_texts = [
            "This movie was fantastic and I enjoyed every minute of it!",
            "The product was terrible and broke within days of purchase.",
            "A beautiful day with clear skies and warm sunshine.",
            "The service at this restaurant was awful and the food was cold."
        ]
        
        self.original_text = random.choice(sample_texts)
        self.current_prompt = self.original_text
        
        # Get initial response and label
        initial_response = self._get_model_response(self.current_prompt)
        self.original_label = self._classify_response(initial_response)
        self.current_step = 0
        
        # Create observation
        tokens = self.tokenizer(self.current_prompt, padding="max_length", 
                              max_length=self.max_tokens, truncation=True)
        response_embedding = self._get_embedding(initial_response)
        
        observation = {
            'prompt_tokens': np.array(tokens['input_ids']),
            'response_embedding': response_embedding,
            'original_label': self.original_label, 
            'current_label': self.original_label
        }
        
        return observation, {}
    
    def step(self, action):
        """
        Take a step in the environment by modifying the prompt and getting the response.
        """
        self.current_step += 1
        
        # Modify the prompt based on the action
        self.current_prompt = self._modify_prompt(self.current_prompt, action)
        
        # Get response from the target model
        response = self._get_model_response(self.current_prompt)
        current_label = self._classify_response(response)
        
        # Calculate reward
        reward = self._calculate_reward(current_label)
        
        # Check if episode is done
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # If we successfully flipped the label, we can end early
        if current_label != self.original_label:
            terminated = True
        
        # Create observation
        tokens = self.tokenizer(self.current_prompt, padding="max_length", 
                              max_length=self.max_tokens, truncation=True)
        response_embedding = self._get_embedding(response)
        
        observation = {
            'prompt_tokens': np.array(tokens['input_ids']),
            'response_embedding': response_embedding,
            'original_label': self.original_label,
            'current_label': current_label
        }
        
        info = {
            'original_text': self.original_text,
            'current_prompt': self.current_prompt,
            'success': current_label != self.original_label
        }
        
        return observation, reward, terminated, truncated, info