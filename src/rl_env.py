import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import random
from datasets import load_dataset

class LLMAdversarialEnv(gym.Env):
    """
    Environment for training an RL agent to generate adversarial prompts against LLMs.
    """
    def __init__(self, 
                 target_model_name="distilbert-base-uncased-finetuned-sst-2-english",
                 model_type="classification",
                 task="sentiment",
                 dataset_name="sst2",
                 max_tokens=50,
                 max_steps=5,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        super(LLMAdversarialEnv, self).__init__()
        
        self.target_model_name = target_model_name
        self.model_type = model_type
        self.task = task
        self.dataset_name = dataset_name
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.current_step = 0
        self.device = device
        
        # Tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_size = self.tokenizer.vocab_size
        
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        if model_type == "classification":
            self.target_model = AutoModelForSequenceClassification.from_pretrained(target_model_name).to(device)
        elif model_type == "generation":
            self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Load dataset
        self.dataset = load_dataset(self.dataset_name, split="test")
        
        # Spaces
        self.action_space = spaces.MultiDiscrete([
            4,             # operation: 0-3
            self.max_tokens,  # position: 0-max_tokens-1
            self.vocab_size   # token id
        ])

        
        self.observation_space = spaces.Dict({
            'prompt_tokens': spaces.Box(low=0, high=self.vocab_size, shape=(max_tokens,), dtype=np.int32),
            'response_embedding': spaces.Box(low=-10, high=10, shape=(768,), dtype=np.float32),
            'original_label': spaces.Discrete(2),
            'current_label': spaces.Discrete(2)
        })
        
        self.current_prompt = None
        self.original_label = None
        self.original_text = None
        
    def _get_model_response(self, prompt):
        inputs = self.target_tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if self.model_type == "classification":
                outputs = self.target_model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()
                return predicted_class
            else:
                generated_ids = self.target_model.generate(
                    inputs.input_ids, max_length=50, do_sample=True, top_p=0.95
                )
                return self.target_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    def _classify_response(self, response):
        if self.model_type == "classification":
            return response
        else:
            if self.task == "sentiment":
                pos_words = ["good", "great", "excellent", "positive"]
                neg_words = ["bad", "terrible", "awful", "negative"]
                response_lower = response.lower()
                pos_count = sum(word in response_lower for word in pos_words)
                neg_count = sum(word in response_lower for word in neg_words)
                return 1 if pos_count > neg_count else 0
            else:
                return 0
    
    def _calculate_reward(self, current_label):
        return 10.0 if current_label != self.original_label else -1.0
    
    def _modify_prompt(self, prompt, action):
        tokens = self.tokenizer.tokenize(prompt)
        op = action['operation']
        pos = min(action['position'], len(tokens) - 1) if tokens else 0
        new_token = self.tokenizer.convert_ids_to_tokens([action['token_id']])[0]
        
        if op == 0 and tokens:
            tokens[pos] = new_token
        elif op == 1:
            tokens.insert(pos, new_token)
        elif op == 2 and tokens and len(tokens) > 1:
            tokens.pop(pos)
        elif op == 3 and tokens and pos < len(tokens) - 1:
            tokens[pos], tokens[pos + 1] = tokens[pos + 1], tokens[pos]
        
        return self.tokenizer.convert_tokens_to_string(tokens)
    
    def _get_embedding(self, response):
        if isinstance(response, int):
            embedding = np.zeros(768)
            embedding[response] = 1.0
            return embedding
        else:
            inputs = self.target_tokenizer(response, return_tensors="pt", padding="max_length", max_length=128, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.target_model.base_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
            return (embeddings / norm).cpu().numpy()[0]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = random.randint(0, len(self.dataset) - 1)
        
        if self.dataset_name == "sst2":
            self.original_text = self.dataset[idx]["sentence"]
        elif self.dataset_name == "imdb":
            self.original_text = self.dataset[idx]["text"]
        elif self.dataset_name == "mrpc":
            self.original_text = self.dataset[idx]["sentence1"] + " [SEP] " + self.dataset[idx]["sentence2"]
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        self.current_prompt = self.original_text
        initial_response = self._get_model_response(self.current_prompt)
        self.original_label = self._classify_response(initial_response)
        self.current_step = 0
        
        tokens = self.tokenizer(self.current_prompt, padding="max_length", max_length=self.max_tokens, truncation=True)
        response_embedding = self._get_embedding(initial_response)
        
        observation = {
            'prompt_tokens': np.array(tokens['input_ids']),
            'response_embedding': response_embedding,
            'original_label': self.original_label,
            'current_label': self.original_label
        }
        return observation, {}
    
    def step(self, action):
        self.current_step += 1
        
        parsed_action = {
            'operation': int(action[0]),
            'position': int(action[1]),
            'token_id': int(action[2])
        }
        self.current_prompt = self._modify_prompt(self.current_prompt, parsed_action)

        response = self._get_model_response(self.current_prompt)
        current_label = self._classify_response(response)
        reward = self._calculate_reward(current_label)
        terminated = current_label != self.original_label
        truncated = self.current_step >= self.max_steps
        
        tokens = self.tokenizer(self.current_prompt, padding="max_length", max_length=self.max_tokens, truncation=True)
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
