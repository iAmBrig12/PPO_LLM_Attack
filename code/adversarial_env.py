import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch # Import torch
from typing import Dict, Any, Tuple, Optional, List
import random

# Assuming llm_interface.py and text_actions.py are in the same directory or PYTHONPATH
from llm_interface import LLMInterface
from text_actions import TextActions

# --- Constants for Reward Shaping ---
# Adjust these weights as needed through experimentation
REWARD_WEIGHT_PROB_DECREASE_ORIGINAL = 0.5 # Weight for reward when original label prob decreases
REWARD_WEIGHT_PROB_INCREASE_TARGET = 0.5 # Weight for reward when target label prob increases
SUCCESS_REWARD = 10.0
TRUNCATION_PENALTY = -0.5 # Keep penalty for hitting max turns without success
# REMOVED: Small step penalty, replaced by shaping

class AdversarialEnv(gym.Env):
    """
    Gymnasium environment for training an RL agent to generate adversarial text prompts.
    Focuses on a classification task where the goal is to flip the original prediction.
    Includes reward shaping and probability-based observation space.
    """
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self,
                 llm_interface: LLMInterface,
                 dataset: List[Dict[str, Any]], # e.g., [{'text': '...', 'label': 'POSITIVE'}, ...]
                 max_turns: int = 10):
        """
        Args:
            llm_interface: An initialized LLMInterface object.
            dataset: A list of dictionaries, each containing 'text' and 'label' (original correct label NAME).
            max_turns: Maximum number of modifications allowed per episode.
        """
        super().__init__()

        if llm_interface.task != "classification":
            raise ValueError("This environment currently only supports classification tasks.")
        if llm_interface.model.config.num_labels != 2:
             print(f"Warning: Environment currently optimized for BINARY classification, but model has {llm_interface.model.config.num_labels} labels. Target logic might need adjustment.")
             # Add handling for multiclass target selection if needed later

        self.llm = llm_interface
        self.dataset = dataset
        self.max_turns = max_turns
        self.num_labels = self.llm.model.config.num_labels # Get number of labels

        self.text_modifier = TextActions()

        # Define action space: Discrete set of text modifications
        self.action_space = spaces.Discrete(self.text_modifier.num_actions)

        # --- Define NEW observation space ---
        # Includes:
        # 0: Current turn (normalized)
        # 1: Probability of the original correct class
        # 2: Probability of the target (opposite) class
        # More features could be added (e.g., text length, similarity to original)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        print(f"Initialized environment with observation space shape: {self.observation_space.shape}")

        # Internal state
        self.current_text: str = ""
        self.original_text: str = ""
        self.original_label_id: Optional[int] = None
        self.target_label_id: Optional[int] = None # The label ID we want the LLM to predict
        self.current_turn: int = 0
        self.last_probabilities: Optional[torch.Tensor] = None # Store probabilities from the previous step
        self._last_action_name: str = "None"


    def _get_obs(self) -> np.ndarray:
        """Generates the numerical observation from the current state including probabilities."""
        if self.last_probabilities is None:
             # Should only happen on the very first observation before first step
             # Initialize with uniform probabilities or zeros if needed
             # For simplicity, let's assume reset populates it correctly.
             # Fallback: return zeros or handle appropriately
             print("Warning: _get_obs called with self.last_probabilities as None. Returning zeros.")
             return np.zeros(self.observation_space.shape, dtype=np.float32)


        turn_normalized = self.current_turn / self.max_turns

        # Ensure probabilities are on CPU and detached before converting
        probs_cpu = self.last_probabilities.cpu().numpy()

        prob_original = probs_cpu[self.original_label_id] if self.original_label_id is not None else 0.0
        prob_target = probs_cpu[self.target_label_id] if self.target_label_id is not None else 0.0

        # Clip probabilities just in case (should be between 0 and 1)
        prob_original = np.clip(prob_original, 0.0, 1.0)
        prob_target = np.clip(prob_target, 0.0, 1.0)


        return np.array([turn_normalized, prob_original, prob_target], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Returns auxiliary information about the current step."""
        info = {
            "current_text": self.current_text,
            "original_text": self.original_text,
            "original_label_id": self.original_label_id,
            "target_label_id": self.target_label_id,
            "current_turn": self.current_turn,
            "action_applied_name": self._last_action_name,
            # Include probabilities for logging if desired
            "last_probabilities": self.last_probabilities.cpu().numpy().tolist() if self.last_probabilities is not None else None
        }
        # Add prediction info if available from step
        if hasattr(self, '_current_prediction'):
            info['predicted_label_id'] = self._current_prediction['predicted_label_id']
            info['predicted_label'] = self._current_prediction['predicted_label']
        return info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              initial_sample: Optional[Dict] = None, # <-- Add this parameter
              _retry_count=0) -> Tuple[np.ndarray | None, Dict[str, Any]]: # Return type might be None if reset fails
        """Starts a new episode, potentially using a specific initial sample."""
        super().reset(seed=seed)

        MAX_RESET_RETRIES = 5 # Reduce retries for evaluation? Or keep 20? Let's keep it low.

        if _retry_count >= MAX_RESET_RETRIES:
            # Indicate failure more clearly for evaluation
            print(f"  Reset failed after {MAX_RESET_RETRIES} attempts (sample {initial_sample.get('text', 'N/A')[:30]}...).")
            # Return None for observation to signal skip in evaluate.py
            return None, {"message": f"Failed to find correctly classified sample after {MAX_RESET_RETRIES} retries."}


        # --- START: Use initial_sample if provided ---
        if initial_sample:
            # Use the specific sample passed for evaluation
            sample = initial_sample
            # No need to increment retry count here, as we only try this specific sample once.
            is_eval_sample = True
        else:
            # Normal training: Sample randomly and handle retries
            sample = random.choice(self.dataset)
            is_eval_sample = False
        # --- END: Use initial_sample ---


        self.original_text = sample.get('text', '') # Use .get for safety
        original_label_name = sample.get('label', '') # Use .get for safety

        if not self.original_text or not original_label_name:
             print(f"  Skipping invalid sample (missing text or label): {sample}")
             # If evaluating, signal failure. If training, retry.
             if is_eval_sample: return None, {"message": "Invalid initial sample provided."}
             else: return self.reset(seed=seed, options=options, _retry_count=_retry_count + 1)


        self.original_label_id = self.llm.get_label_id(original_label_name)
        if self.original_label_id is None:
             print(f"  Warning: Label name '{original_label_name}' not found in model config ({self.llm.label2id}). Skipping/Resampling.")
             if is_eval_sample: return None, {"message": f"Label '{original_label_name}' not found."}
             else: return self.reset(seed=seed, options=options, _retry_count=_retry_count + 1)

        # Determine target label ID
        if self.num_labels == 2:
            self.target_label_id = 1 - self.original_label_id
        else:
            possible_targets = list(range(self.num_labels))
            possible_targets.remove(self.original_label_id)
            self.target_label_id = random.choice(possible_targets) if possible_targets else None


        # Ensure the original sample is classified correctly
        try:
            initial_pred = self.llm.predict(self.original_text)
            prediction_correct = initial_pred['predicted_label_id'] == self.original_label_id
            self.last_probabilities = initial_pred['probabilities']
        except Exception as e:
             print(f"  Warning: Error during initial prediction in reset for sample '{self.original_text[:30]}...': {e}. Skipping/Resampling.")
             prediction_correct = False
             self.last_probabilities = None

        # --- Handle Incorrect Initial Prediction ---
        if not prediction_correct or self.last_probabilities is None:
            if is_eval_sample:
                # During evaluation, skip this sample if LLM already gets it wrong
                print(f"  Skipping evaluation sample: LLM initial prediction was incorrect (Pred: {initial_pred.get('predicted_label_id', 'ERR')} vs Orig: {self.original_label_id}) or failed.")
                return None, {"message": "LLM misclassified the original sample initially."}
            else:
                # During training, retry with a different random sample
                # print(f"Warning: Original sample not classified correctly or prediction failed. Retrying reset (attempt {_retry_count+1}).")
                return self.reset(seed=seed, options=options, _retry_count=_retry_count + 1)
        # --- End Handle Incorrect Initial Prediction ---

        # --- If prediction was correct, proceed ---
        self.current_text = self.original_text
        self.current_turn = 0
        self._last_action_name = "None"

        observation = self._get_obs()
        info = self._get_info()
        info['message'] = "Episode reset" + (" (Evaluation Sample)" if is_eval_sample else "")

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies action, gets prediction, calculates shaped reward."""
        self._last_action_name = self.text_modifier.get_action_name(action)

        # Store probabilities from *before* the action
        previous_probabilities = self.last_probabilities.clone() if self.last_probabilities is not None else None

        # 1. Apply action
        previous_text = self.current_text
        self.current_text = self.text_modifier.apply_action(self.current_text, action)

        # Check if action was valid
        if not self.current_text or self.current_text == previous_text:
             # No change or empty text - Penalize slightly? Or just return 0 reward? Let's do 0 for now.
             reward = 0.0
             terminated = False
             truncated = self.current_turn >= self.max_turns -1
             # Keep last_probabilities unchanged
             observation = self._get_obs()
             info = self._get_info()
             info['message'] = f"Action '{self._last_action_name}' resulted in no change or empty text."
             self.current_turn += 1 # Consume turn
             return observation, reward, terminated, truncated, info


        # 2. Query LLM with the modified text
        try:
            prediction = self.llm.predict(self.current_text)
            self._current_prediction = prediction # Store for info
            current_probabilities = prediction['probabilities']
            predicted_label_id = prediction['predicted_label_id']
        except Exception as e:
            print(f"ERROR during LLM prediction in step: {e}")
            # How to handle? Penalize heavily? Skip step? Let's penalize and end turn.
            reward = -1.0 # Penalize prediction failure
            terminated = False
            truncated = self.current_turn >= self.max_turns -1
            # Can't update probabilities, observation might be stale
            observation = self._get_obs() # Use previous probabilities
            info = self._get_info()
            info['message'] = f"LLM Prediction failed after action '{self._last_action_name}'."
            self.current_turn += 1
            return observation, reward, terminated, truncated, info


        # 3. Calculate reward (Shaped + Success Bonus + Truncation Penalty)
        terminated = False
        reward = 0.0
        shaping_reward = 0.0
        info_message = "Step executed."

        # --- Reward Shaping ---
        if previous_probabilities is not None:
            # Change in probability of the original label
            prev_prob_original = previous_probabilities[self.original_label_id].item()
            curr_prob_original = current_probabilities[self.original_label_id].item()
            delta_original = prev_prob_original - curr_prob_original # Positive if prob decreased
            shaping_reward += REWARD_WEIGHT_PROB_DECREASE_ORIGINAL * delta_original

            # Change in probability of the target label
            prev_prob_target = previous_probabilities[self.target_label_id].item()
            curr_prob_target = current_probabilities[self.target_label_id].item()
            delta_target = curr_prob_target - prev_prob_target # Positive if prob increased
            shaping_reward += REWARD_WEIGHT_PROB_INCREASE_TARGET * delta_target

            # Add shaping reward to total step reward
            reward += shaping_reward
        # --- End Reward Shaping ---


        # --- Check for Success ---
        if predicted_label_id != self.original_label_id:
            # Success! The prediction is flipped.
            reward += SUCCESS_REWARD  # Add success bonus
            terminated = True # End the episode on success
            info_message = "Attack successful!"
        # --- End Check for Success ---

        # Update last probabilities for the next step
        self.last_probabilities = current_probabilities

        # 4. Check truncation (max turns reached)
        self.current_turn += 1
        truncated = self.current_turn >= self.max_turns
        if truncated and not terminated:
            # Apply truncation penalty only if max turns reached without success
            reward += TRUNCATION_PENALTY
            info_message = "Max turns reached without success."

        # 5. Get observation and info
        observation = self._get_obs() # Based on NEW (current) probabilities
        info = self._get_info()
        info['shaping_reward'] = shaping_reward # Add shaping component to info for debugging
        info['message'] = info_message

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        print("Closing AdversarialEnv.")


# --- Example Usage Block (No changes needed here) ---
if __name__ == "__main__":
    from datasets import load_dataset

    print("Testing Adversarial Environment...")
    # --- Configuration ---
    # Important: Use a model fine-tuned for the dataset's task!
    MODEL_NAME = "textattack/bert-base-uncased-imdb" # Fine-tuned for IMDB
    DATASET_NAME = "imdb"
    DATASET_SPLIT = "test[:100]" # Use test split for eval-like testing
    LABEL_COL = "label"
    TEXT_COL = "text"
    # Map dataset label IDs (0, 1) to human-readable names expected by the env
    # For IMDB: 0 -> negative, 1 -> positive
    DATASET_ID_TO_NAME = {0: "NEGATIVE", 1: "POSITIVE"} # Define the mapping for IMDB

    MAX_TURNS = 15
    # ---------------------

    try:
        # 1. Load LLM Interface - Ensure labels are loaded correctly (e.g., {0: 'NEGATIVE', 1: 'POSITIVE'})
        # For textattack/bert-base-uncased-imdb, the default config might be LABEL_0, LABEL_1.
        # We need to ensure our environment receives the correct names ('NEGATIVE', 'POSITIVE').
        # Option 1: Load model, manually set config labels if needed (demonstrated below)
        # Option 2: Choose a model that already has the desired label names in its config on the Hub.

        llm_interface = LLMInterface(MODEL_NAME, task="classification")
        # ---- IMPORTANT: Verify/Override Model Labels if Necessary ----
        # Check if the model's config matches our expected DATASET_ID_TO_NAME for the dataset
        # The textattack model likely uses LABEL_0, LABEL_1. Let's override for consistency with our dataset.
        if llm_interface.model.config.id2label != DATASET_ID_TO_NAME:
            print(f"Model config labels ({llm_interface.model.config.id2label}) don't match dataset mapping ({DATASET_ID_TO_NAME}). Overriding for environment.")
            # Create the inverse mapping as well
            label2id = {v: k for k, v in DATASET_ID_TO_NAME.items()}
            llm_interface.model.config.id2label = DATASET_ID_TO_NAME
            llm_interface.model.config.label2id = label2id
            # Update the interface attributes directly
            llm_interface.id2label = DATASET_ID_TO_NAME
            llm_interface.label2id = label2id
            print(f"Updated interface labels: {llm_interface.id2label}")
         # ------------------------------------------------------------

        # 2. Load Dataset
        raw_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

        # Convert dataset label IDs to names based on DATASET_ID_TO_NAME
        processed_dataset = [{"text": item[TEXT_COL], "label": DATASET_ID_TO_NAME[item[LABEL_COL]]} for item in raw_dataset]
        print(f"Example processed sample: {processed_dataset[0]}")


        # 3. Create Environment
        env = AdversarialEnv(llm_interface=llm_interface, dataset=processed_dataset, max_turns=MAX_TURNS)

        # 4. Basic Interaction Test
        obs, info = env.reset()
        print("\n--- Episode Start ---")
        print(f"Initial Observation (Turn, ProbOriginal, ProbTarget): {obs}")
        # print(f"Initial Info: {info}") # Info dict can be large
        print(f"Original Text: '{info['original_text'][:100]}...' (Orig Label ID: {info['original_label_id']})")
        print("---------------------\n")

        terminated = False
        truncated = False
        total_reward = 0.0
        step = 0

        while not terminated and not truncated:
            step += 1
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            print(f"--- Step {step} ---")
            print(f"Action Taken: {info.get('action_applied_name', 'N/A')} ({action})")
            print(f"Observation (Turn, ProbOriginal, ProbTarget): {obs}")
            print(f"Reward: {reward:.4f} (Shaping: {info.get('shaping_reward', 0):.4f})")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Current Text: '{info.get('current_text', '')[:100]}...'")
            print(f"LLM Prediction: {info.get('predicted_label', 'N/A')} (ID: {info.get('predicted_label_id', 'N/A')})")
            print(f"Info Msg: {info.get('message', '')}")
            print("--------------\n")


        print(f"--- Episode End ---")
        print(f"Total Reward: {total_reward:.4f}")
        print(f"Final Text: '{info.get('current_text', '')}'")
        print("-------------------\n")

        env.close()

    except Exception as e:
        print(f"\nAn error occurred during environment testing: {e}")
        import traceback
        traceback.print_exc()