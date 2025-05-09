import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, Optional
import time 

class LLMInterface:
    def __init__(self, model_name: str, task: str = "classification", device: Optional[str] = None):
        print(f"LLMInterface: Initializing for model '{model_name}'...")
        self.model_name = model_name
        
        # Ensure task is classification for this focused interface
        if task != "classification":
            raise ValueError(f"This LLMInterface is configured for 'classification' task only, got '{task}'.")
        self.task = task

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LLMInterface: Using device: {self.device}")

        print(f"LLMInterface: Loading tokenizer for '{model_name}'...")
        tokenizer_load_start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"LLMInterface: Tokenizer loaded in {time.time() - tokenizer_load_start_time:.2f}s.")

        # Set Pad Token if Missing (Good general practice for classifiers if tokenizer doesn't have one)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                print(f"LLMInterface: Tokenizer for '{model_name}' is missing a pad token. Setting pad_token to eos_token: '{self.tokenizer.eos_token}'.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Fallback if no eos_token either, though most classification tokenizers will have pad_token
                new_pad_token = '[PAD]'
                print(f"LLMInterface: Tokenizer for '{model_name}' is missing both pad and eos tokens. Adding a new pad_token: '{new_pad_token}'.")
                self.tokenizer.add_special_tokens({'pad_token': new_pad_token})
                # If a new token is truly added to vocab, model embeddings might need resizing,
                # but for pre-finedtuned classifiers, this is mostly about tokenizer config.


        print(f"LLMInterface: Loading model '{model_name}' for task '{self.task}'...")
        model_load_start_time = time.time()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Ensure model's pad_token_id matches tokenizer's if set
        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode
        print(f"LLMInterface: Model loaded and moved to device in {time.time() - model_load_start_time:.2f}s.")

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        print(f"LLMInterface: Successfully loaded model '{self.model_name}'. Labels: {getattr(self, 'id2label', 'N/A')}")

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        
        predicted_label = self.id2label.get(predicted_id, f"UNKNOWN_ID_{predicted_id}")

        return {
            "logits": logits.squeeze().cpu(),
            "probabilities": probabilities.squeeze().cpu(),
            "predicted_label_id": predicted_id,
            "predicted_label": predicted_label
        }

    def get_label_id(self, label_name: str) -> Optional[int]:
        if hasattr(self, 'label2id'):
            return self.label2id.get(label_name)
        return None