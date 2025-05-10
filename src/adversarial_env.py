import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch 
from typing import Dict, Any, Tuple, Optional, List
import random

# Assuming llm_interface.py and text_actions.py are in the same directory or PYTHONPATH
from llm_interface import LLMInterface
from text_actions import TextActions

# --- Constants for Reward Shaping ---
REWARD_WEIGHT_PROB_DECREASE_ORIGINAL = 0.5  # Weight for reward when original label prob decreases
REWARD_WEIGHT_PROB_INCREASE_TARGET = 0.5    # Weight for reward when target label prob increases
SUCCESS_REWARD = 5.0                        # Reward for successful attack (prediction flip) 
TRUNCATION_PENALTY = -0.5                   # Keep penalty for hitting max turns without success

class AdversarialEnv(gym.Env):
    def __init__(self,
                 llm_interface: LLMInterface,
                 dataset: List[Dict[str, Any]],
                 max_turns: int = 10):
        super().__init__()

        if llm_interface.task != "classification":
            raise ValueError("This environment currently only supports classification tasks.")

        self.llm = llm_interface
        self.dataset = dataset
        self.max_turns = max_turns

        self.num_labels = self.llm.model.config.num_labels
        if self.num_labels < 2:
            raise ValueError("Classification task must have at least 2 labels.")

        self.text_modifier = TextActions()
        self.action_space = spaces.Discrete(self.text_modifier.num_actions)

        # --- Define Observation Space for Multi-Class ---
        # Shape: [turn_normalized, P(class0), P(class1), ..., P(classN-1)]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1 + self.num_labels,), dtype=np.float32   # Notice the comma: (shape_value,)
        )
        print(f"Initialized environment with observation space shape: {self.observation_space.shape} (Num Labels: {self.num_labels})")

        # Internal state
        self.current_text: str = ""
        self.original_text: str = ""
        self.original_label_id: Optional[int] = None
        self.target_label_id: Optional[int] = None              # The label ID we want the LLM to predict
        self.current_turn: int = 0
        self.last_probabilities: Optional[torch.Tensor] = None  # Store probabilities from the previous step
        self._last_action_name: str = "None"


    def _get_obs(self) -> np.ndarray:
        if self.last_probabilities is None:
            print("Warning: _get_obs called with self.last_probabilities as None. Returning zeros.")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        turn_normalized = self.current_turn / self.max_turns
        probs_cpu = self.last_probabilities.cpu().numpy() # Should have shape (num_labels,)

        # Ensure probs_cpu is correctly shaped if it's a scalar for some reason
        if probs_cpu.ndim == 0: # If it's a scalar tensor that became a 0-d array
            print(f"Warning: probs_cpu is a scalar in _get_obs: {probs_cpu}. Expanding to shape ({self.num_labels},).")
            temp_probs = np.zeros(self.num_labels, dtype=np.float32)
            if self.original_label_id is not None and 0 <= self.original_label_id < self.num_labels:
                temp_probs[self.original_label_id] = probs_cpu
            probs_cpu = temp_probs


        # Concatenate turn_normalized with all class probabilities
        observation = np.concatenate(([turn_normalized], probs_cpu)).astype(np.float32)

        # Ensure observation matches the defined space shape
        if observation.shape != self.observation_space.shape:
            print(f"ERROR: Observation shape mismatch! Expected {self.observation_space.shape}, got {observation.shape}")
            correct_shape_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            common_len = min(len(observation), len(correct_shape_obs))
            correct_shape_obs[:common_len] = observation[:common_len]
            return correct_shape_obs

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """Returns auxiliary information about the current step."""
        info = {
            "current_text": self.current_text,
            "original_text": self.original_text,
            "original_label_id": self.original_label_id,
            "target_label_id": self.target_label_id,
            "current_turn": self.current_turn,
            "action_applied_name": self._last_action_name,
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
              initial_sample: Optional[Dict] = None,
              _retry_count=0) -> Tuple[np.ndarray | None, Dict[str, Any]]:
        super().reset(seed=seed)

        MAX_RESET_RETRIES = 50

        if _retry_count >= MAX_RESET_RETRIES:
            sample_text_snippet = "N/A (random sampling exhausted retries)"     # Default message for training
            if initial_sample is not None:                                      # Only try .get if initial_sample exists
                sample_text_snippet = initial_sample.get('text', 'N/A (initial_sample provided but no text key)')[:30]
            print(f"  Reset failed after {MAX_RESET_RETRIES} attempts (sample context: {sample_text_snippet}).")
            raise RuntimeError(f"Failed to find a correctly classified sample after {MAX_RESET_RETRIES} attempts in env.reset() during training.")


        if initial_sample:
            sample = initial_sample
            is_eval_sample = True   # We are in evaluation mode if initial_sample is provided
        else:
            # Normal training: Sample randomly
            sample = random.choice(self.dataset)
            is_eval_sample = False

        self.original_text = sample.get('text', '') 
        original_label_name = sample.get('label', '')

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
        else: # For multi-class
            possible_targets = list(range(self.num_labels))
            if self.original_label_id is not None and self.original_label_id in possible_targets:
                possible_targets.remove(self.original_label_id)
            self.target_label_id = random.choice(possible_targets) if possible_targets else None
            if self.target_label_id is None and self.num_labels > 1:
                print(f"Warning: Could not select a distinct target_label_id for original_label_id {self.original_label_id}")

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
                return self.reset(seed=seed, options=options, _retry_count=_retry_count + 1)

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
            self._current_prediction = prediction   # Store for info
            current_probabilities = prediction['probabilities']
            predicted_label_id = prediction['predicted_label_id']
        except Exception as e:
            print(f"ERROR during LLM prediction in step: {e}")
            reward = -1.0   # Penalize prediction failure
            terminated = False
            truncated = self.current_turn >= self.max_turns -1
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
            delta_original = prev_prob_original - curr_prob_original    # Positive if prob decreased
            shaping_reward += REWARD_WEIGHT_PROB_DECREASE_ORIGINAL * delta_original

            # Change in probability of the target label
            prev_prob_target = previous_probabilities[self.target_label_id].item()
            curr_prob_target = current_probabilities[self.target_label_id].item()
            delta_target = curr_prob_target - prev_prob_target          # Positive if prob increased
            shaping_reward += REWARD_WEIGHT_PROB_INCREASE_TARGET * delta_target

            # Add shaping reward to total step reward
            reward += shaping_reward


        # --- Check for Success ---
        if predicted_label_id != self.original_label_id:
            reward += SUCCESS_REWARD  # Add success bonus
            terminated = True # End the episode on success
            info_message = "Attack successful!"

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
