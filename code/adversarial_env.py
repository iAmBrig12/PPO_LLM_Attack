import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import random

# Assuming llm_interface.py and text_actions.py are in the same directory or PYTHONPATH
from llm_interface import LLMInterface
from text_actions import TextActions

class AdversarialEnv(gym.Env):
    """
    Gymnasium environment for training an RL agent to generate adversarial text prompts.
    Focuses on a classification task where the goal is to flip the original prediction.
    """
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self,
                 llm_interface: LLMInterface,
                 dataset: List[Dict[str, Any]], # e.g., [{'text': '...', 'label': 'POSITIVE'}, ...]
                 max_turns: int = 10):
        """
        Args:
            llm_interface: An initialized LLMInterface object.
            dataset: A list of dictionaries, each containing 'text' and 'label' (original correct label).
            max_turns: Maximum number of modifications allowed per episode.
        """
        super().__init__()

        if llm_interface.task != "classification":
            raise ValueError("This environment currently only supports classification tasks.")

        self.llm = llm_interface
        self.dataset = dataset
        self.max_turns = max_turns

        self.text_modifier = TextActions()

        # Define action space: Discrete set of text modifications
        self.action_space = spaces.Discrete(self.text_modifier.num_actions)

        # Define observation space:
        # This is challenging for text. A simple approach might include:
        # - Current turn number
        # - Maybe length of the text?
        # - Similarity to original text (e.g., BLEU score - requires storing original)
        # A more complex approach would use text embeddings (e.g., from sentence-transformers).
        # For this skeleton, let's use a simple Dict space.
        # IMPORTANT: Standard SB3 policies (like MlpPolicy) expect flattened Box spaces.
        # You might need a custom policy network or use wrappers/feature extraction
        # (e.g., `FlattenObservation`) or switch to libraries supporting Dict spaces directly.
        # Or, represent state numerically (e.g., fixed-size embedding).
        # Let's use a simple numerical Box space placeholder for compatibility for now.
        # Proper implementation needs text embedding here.
        # Placeholder: [current_turn, text_length_normalized]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)


        # Internal state
        self.current_text: str = ""
        self.original_text: str = ""
        self.original_label_id: Optional[int] = None
        self.current_turn: int = 0

    def _get_obs(self) -> np.ndarray:
        """Generates the numerical observation from the current state."""
        # Placeholder implementation - replace with meaningful features/embeddings
        turn_normalized = self.current_turn / self.max_turns
        # Normalize length roughly (e.g., assuming max 500 chars)
        length_normalized = min(len(self.current_text) / 500.0, 1.0)
        return np.array([turn_normalized, length_normalized], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Returns auxiliary information about the current step."""
        return {
            "current_text": self.current_text,
            "original_text": self.original_text,
            "original_label_id": self.original_label_id,
            "current_turn": self.current_turn,
            "action_applied_name": self._last_action_name # Store last action for debugging
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Starts a new episode."""
        super().reset(seed=seed)

        # Sample a new starting point from the dataset
        sample = random.choice(self.dataset)
        self.original_text = sample['text']
        original_label_name = sample['label'] # Assumes dataset has label names matching model config
        self.original_label_id = self.llm.get_label_id(original_label_name)

        # Ensure the original sample is classified correctly by the LLM (or skip if not)
        # This prevents starting with an already "failed" sample
        initial_pred = self.llm.predict(self.original_text)
        if self.original_label_id is None or initial_pred['predicted_label_id'] != self.original_label_id:
             # print(f"Warning: Original sample '{self.original_text[:50]}...' not classified as '{original_label_name}' by the model (predicted {initial_pred['predicted_label']}). Skipping and resampling.")
             # In a real scenario, you might want better handling (e.g., filter dataset beforehand)
             # For now, just recurse to try again
             return self.reset(seed=seed, options=options)


        self.current_text = self.original_text
        self.current_turn = 0
        self._last_action_name = "None" # Track action for info

        observation = self._get_obs()
        info = self._get_info()
        info['message'] = "Episode reset" # Add a message for clarity

        # print(f"Resetting env. Original: '{self.original_text[:50]}...' (Label: {original_label_name}, ID: {self.original_label_id})")

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies an action, gets LLM prediction, calculates reward."""
        self._last_action_name = self.text_modifier.get_action_name(action) # Store for info

        # 1. Apply action to modify text
        previous_text = self.current_text
        self.current_text = self.text_modifier.apply_action(self.current_text, action)

        # Penalize if text becomes empty or action failed
        if not self.current_text or self.current_text == previous_text:
             penalty = -0.1 # Small penalty for invalid/no-op actions
             reward = penalty
             terminated = False # Not a success
             truncated = self.current_turn >= self.max_turns -1 # Check truncation early if action fails
             observation = self._get_obs()
             info = self._get_info()
             info['message'] = f"Action '{self._last_action_name}' resulted in no change or empty text."
             self.current_turn += 1 # Consume a turn
             return observation, reward, terminated, truncated, info


        # 2. Query the LLM with the modified text
        prediction = self.llm.predict(self.current_text)
        predicted_label_id = prediction['predicted_label_id']

        # 3. Calculate reward and termination conditions
        terminated = False
        reward = 0.0

        if predicted_label_id != self.original_label_id:
            # Success! The prediction is flipped.
            reward = 1.0  # Positive reward for successful attack
            terminated = True # End the episode on success
            info_message = "Attack successful!"
        else:
            # Failure - prediction remains the same.
            # Small negative reward to encourage change, could be 0 too.
            reward = -0.05
            info_message = "Attack failed, prediction unchanged."

        # 4. Check truncation (max turns reached)
        self.current_turn += 1
        truncated = self.current_turn >= self.max_turns
        if truncated and not terminated:
            # Penalize reaching max turns without success
            reward = -0.5
            info_message = "Max turns reached without success."


        # 5. Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        info['predicted_label_id'] = predicted_label_id
        info['predicted_label'] = prediction['predicted_label']
        info['message'] = info_message

        # print(f"Turn {self.current_turn}: Action='{self._last_action_name}', Pred={info['predicted_label']}, Reward={reward:.2f}, Term={terminated}, Trunc={truncated}")
        # print(f"   Text: '{self.current_text[:100]}...'")


        return observation, reward, terminated, truncated, info

    def render(self):
        # Rendering is not applicable for this text-based environment
        pass

    def close(self):
        # Clean up any resources if needed
        print("Closing AdversarialEnv.")


# Example Usage (test environment stepping)
if __name__ == "__main__":
    from datasets import load_dataset

    print("Testing Adversarial Environment...")

    # --- Configuration ---
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" # Make sure this model is available
    DATASET_NAME = "sst2" # Stanford Sentiment Treebank
    DATASET_SPLIT = "validation[:100]" # Use a small subset of validation for testing
    MAX_TURNS = 5
    # ---------------------

    try:
        # 1. Load LLM Interface
        llm_interface = LLMInterface(MODEL_NAME, task="classification")

        # 2. Load Dataset
        raw_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        # Convert label IDs (0, 1) to names ('negative', 'positive') expected by env init
        # This depends on the specific dataset's features
        id2label_func = lambda x: llm_interface.id2label.get(x, "Unknown")
        processed_dataset = [{"text": item["sentence"], "label": id2label_func(item["label"])} for item in raw_dataset]


        # 3. Create Environment
        env = AdversarialEnv(llm_interface=llm_interface, dataset=processed_dataset, max_turns=MAX_TURNS)

        # 4. Basic Interaction Test
        obs, info = env.reset()
        print("\n--- Episode Start ---")
        print(f"Initial Observation: {obs}")
        print(f"Initial Info: {info}")
        print("---------------------\n")

        terminated = False
        truncated = False
        total_reward = 0
        step = 0

        while not terminated and not truncated:
            step += 1
            # Take a random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"--- Step {step} ---")
            print(f"Action Taken: {info.get('action_applied_name', 'N/A')} ({action})")
            print(f"Observation: {obs}")
            print(f"Reward: {reward:.2f}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            print(f"Current Text: '{info.get('current_text', '')[:100]}...'")
            print(f"LLM Prediction: {info.get('predicted_label', 'N/A')}")
            print(f"Info Msg: {info.get('message', '')}")
            print("--------------\n")

            total_reward += reward

        print(f"--- Episode End ---")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Text: '{info.get('current_text', '')}'")
        print("-------------------\n")

        env.close()

    except Exception as e:
        print(f"\nAn error occurred during environment testing: {e}")
        import traceback
        traceback.print_exc()