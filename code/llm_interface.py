import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from typing import Tuple, Dict, Any, Optional

class LLMInterface:
    """Handles loading and querying Hugging Face models for specific tasks."""

    def __init__(self, model_name: str, task: str = "classification", device: Optional[str] = None):
        """
        Loads tokenizer and model.

        Args:
            model_name (str): Hugging Face model identifier (e.g., 'bert-base-uncased').
            task (str): The task type ('classification', 'generation', etc.). Affects model loading.
            device (Optional[str]): 'cuda', 'cpu', or None (auto-detect).
        """
        self.model_name = model_name
        self.task = task
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval() # Set to evaluation mode
            # Store label mapping if available
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
        elif self.task == "generation":
             # Basic loading, generation params might need tuning
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported task: {task}")

        print(f"Loaded model '{model_name}' for task '{task}'.")
        if hasattr(self, 'id2label'):
             print(f"Labels: {self.id2label}")


    @torch.no_grad() # Disable gradient calculations for inference
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Gets predictions from the loaded LLM for a given text input.

        Args:
            text (str): The input text prompt.

        Returns:
            Dict[str, Any]: A dictionary containing prediction results.
                            For classification: {'logits': tensor, 'probabilities': tensor, 'predicted_label_id': int, 'predicted_label': str}
                            For generation: {'generated_text': str} (basic example)
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        if self.task == "classification":
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_id = torch.argmax(probabilities, dim=-1).item()
            predicted_label = self.id2label.get(predicted_id, "Unknown") # Handle potential missing mapping

            return {
                "logits": logits.squeeze().cpu(),
                "probabilities": probabilities.squeeze().cpu(),
                "predicted_label_id": predicted_id,
                "predicted_label": predicted_label
            }
        elif self.task == "generation":
            # Basic generation example, params can be customized
            outputs = self.model.generate(**inputs, max_new_tokens=50)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"generated_text": generated_text}
        else:
            # Should not happen based on init check
            raise RuntimeError("Task type not supported during prediction.")

    def get_label_id(self, label_name: str) -> Optional[int]:
        """Returns the ID for a given label name for classification models."""
        if hasattr(self, 'label2id'):
            return self.label2id.get(label_name)
        return None

# Example Usage (test loading)
if __name__ == "__main__":
    # Test Classification
    try:
        print("\n--- Testing Classification ---")
        # Use a model fine-tuned for sentiment (e.g., on sst2)
        cls_interface = LLMInterface("distilbert-base-uncased-finetuned-sst-2-english", task="classification")
        positive_pred = cls_interface.predict("This movie is fantastic!")
        print(f"Input: 'This movie is fantastic!' -> Prediction: {positive_pred['predicted_label']} (ID: {positive_pred['predicted_label_id']})")
        negative_pred = cls_interface.predict("This movie is terrible.")
        print(f"Input: 'This movie is terrible.' -> Prediction: {negative_pred['predicted_label']} (ID: {negative_pred['predicted_label_id']})")
        print(f"Label 'POSITIVE' has ID: {cls_interface.get_label_id('POSITIVE')}")
        print(f"Label 'NEGATIVE' has ID: {cls_interface.get_label_id('NEGATIVE')}")

    except Exception as e:
        print(f"Could not run classification test: {e}")

    # Test Generation (example, might need larger model)
    try:
        print("\n--- Testing Generation ---")
        gen_interface = LLMInterface("gpt2", task="generation") # Use a small model like gpt2
        gen_output = gen_interface.predict("Once upon a time")
        print(f"Input: 'Once upon a time' -> Output: '{gen_output['generated_text'][:100]}...'") # Print start of output
    except Exception as e:
        print(f"Could not run generation test (maybe model too large or not found): {e}")