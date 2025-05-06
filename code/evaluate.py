# evaluate.py

import argparse
import os
import json
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from datasets import load_dataset

# Assuming llm_interface.py and adversarial_env.py are in the same directory or accessible
# Also assumes adversarial_env.py has the modified reset method accepting `initial_sample`
from llm_interface import LLMInterface
from adversarial_env import AdversarialEnv

# --- Define Dataset Label Mappings (Should match train.py) ---
DATASET_LABEL_MAPS = {
    "imdb": {0: "NEGATIVE", 1: "POSITIVE"},
    "sst2": {0: "NEGATIVE", 1: "POSITIVE"},
    # Add other datasets used during training here
}

def get_dataset_mapping(dataset_name):
    name_lower = dataset_name.lower()
    if name_lower in DATASET_LABEL_MAPS:
        return DATASET_LABEL_MAPS[name_lower]
    else:
        print(f"Warning: No predefined label mapping found for dataset '{dataset_name}'. Attempting to use model's default or numerical labels.")
        return None

def evaluate_agent(args):
    """Loads components and evaluates the trained PPO agent."""

    print("--- Starting Evaluation ---")
    print(f"Arguments: {args}")

    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        return

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {args.device}")

    # 1. Load LLM Interface
    print(f"Loading LLM: {args.llm_model_name}")
    try:
        llm_interface = LLMInterface(model_name=args.llm_model_name, task="classification", device=args.device)
    except Exception as e:
        print(f"Error loading LLM {args.llm_model_name}: {e}")
        return

    # --- Get Dataset Label Mapping ---
    dataset_label_mapping = get_dataset_mapping(args.dataset_name)
    if not dataset_label_mapping:
         # Fallback to model's default if no mapping defined
         print(f"Warning: Using LLM's default label mapping: {llm_interface.id2label}")
         dataset_label_mapping = llm_interface.id2label

    # --- Ensure LLM Interface uses the correct mapping for consistency ---
    # (This ensures the environment gets the right ID when it looks up the label name)
    if llm_interface.id2label != dataset_label_mapping:
         print(f"Overriding LLM config labels ({llm_interface.id2label}) to match dataset mapping ({dataset_label_mapping}) for environment consistency.")
         label2id = {v: k for k, v in dataset_label_mapping.items()}
         llm_interface.model.config.id2label = dataset_label_mapping
         llm_interface.model.config.label2id = label2id
         llm_interface.id2label = dataset_label_mapping
         llm_interface.label2id = label2id

    # 2. Load Dataset
    print(f"Loading Dataset: {args.dataset_name}, Split: {args.dataset_split}")
    try:
        # Load the full split first, then select the number of samples
        full_raw_dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        if args.num_samples > 0 and args.num_samples < len(full_raw_dataset):
             raw_dataset = full_raw_dataset.select(range(args.num_samples))
             print(f"Selected first {args.num_samples} samples for evaluation.")
        else:
             raw_dataset = full_raw_dataset
             print(f"Using all {len(raw_dataset)} samples from the split.")
    except Exception as e:
        print(f"Error loading dataset '{args.dataset_name}' split '{args.dataset_split}': {e}")
        return

    # Preprocess dataset
    text_col = "text" if "text" in raw_dataset.column_names else "sentence"
    label_col = "label"
    if text_col not in raw_dataset.column_names or label_col not in raw_dataset.column_names:
         print(f"Error: Dataset must contain '{text_col}' and '{label_col}' columns. Found: {raw_dataset.column_names}")
         return

    id2name_func = lambda x: dataset_label_mapping.get(x, f"UNKNOWN_LABEL_{x}")

    try:
        processed_test_dataset = [{"text": item[text_col], "label": id2name_func(item[label_col])} for item in raw_dataset]
        print(f"Processed {len(processed_test_dataset)} test samples.")
        if not processed_test_dataset:
             print("Error: No samples left after processing.")
             return
    except KeyError as e:
         print(f"Error processing dataset: Label ID {e} not found in mapping {dataset_label_mapping}. Check dataset and mapping.")
         return
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return

    # 3. Load Trained Agent
    print(f"Loading trained agent from: {args.model_path}")
    try:
        # Note: The environment is not needed for loading, but we pass device
        agent = PPO.load(args.model_path, device=args.device)
        print("Agent loaded successfully.")
    except Exception as e:
        print(f"Error loading agent: {e}")
        return

    # 4. Instantiate Environment (using the full test dataset - reset will handle specifics)
    # Note: The 'dataset' param here is primarily used by env.reset() for *random* sampling,
    # which we will override in the loop using the initial_sample argument.
    # However, it's useful to pass it so the env is aware of the potential samples.
    print("Initializing evaluation environment...")
    try:
        # Important: Use the same max_turns the agent was trained with for fair evaluation
        env = AdversarialEnv(llm_interface=llm_interface, dataset=processed_test_dataset, max_turns=args.max_turns)
    except Exception as e:
        print(f"Error initializing AdversarialEnv: {e}")
        return

    successful_attacks = 0
    total_queries = 0
    successful_attack_turns = []
    skipped_samples = 0
    results_data = []

    print(f"\n--- Starting Evaluation Loop ({len(processed_test_dataset)} samples) ---")
    start_time = time.time()

    for i, sample in enumerate(processed_test_dataset):
        print(f"Evaluating sample {i+1}/{len(processed_test_dataset)}...")
        current_sample_queries = 0
        obs, info = None, None # Initialize obs and info

        try:
            # Reset the environment with the specific sample
            obs, info = env.reset(initial_sample=sample)

            # --- Check if reset failed BEFORE accessing info details ---
            if obs is None:
                 print(f"  Skipping sample: Reset failed. Reason: {info.get('message', 'Unknown error during reset')}")
                 skipped_samples += 1
                 continue # Skip to the next sample

            # --- Reset succeeded, now safely access info ---
            original_text = info.get('original_text', 'ERROR: Original text not found in info') # Use .get for safety
            original_label_id = info.get('original_label_id', 'ERROR: Original label ID not found in info')

        except Exception as e: # Catch any other unexpected errors during reset
             print(f"  Skipping sample {i+1}: Unexpected error during reset processing - {e}")
             import traceback
             traceback.print_exc()
             skipped_samples += 1
             continue

        # --- Proceed with the evaluation loop only if reset was successful ---
        terminated = False
        truncated = False
        attack_successful_this_sample = False
        num_turns_this_sample = 0

        # Add initial predict call query count (done during reset)
        current_sample_queries += 1

        while not terminated and not truncated:
            num_turns_this_sample += 1
            action, _ = agent.predict(obs, deterministic=True)
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                current_sample_queries += 1
            except Exception as e:
                 print(f"  ERROR during env.step for sample {i+1}: {e}. Ending episode for this sample.")
                 break

            if terminated:
                successful_attacks += 1
                successful_attack_turns.append(num_turns_this_sample)
                attack_successful_this_sample = True
                print(f"  Success! Attack succeeded in {num_turns_this_sample} turns.")
                results_data.append({
                    "index": i,
                    "status": "success",
                    "original_text": original_text, # Safe to use now
                    "adversarial_text": info.get('current_text', 'ERROR'), # Use .get
                    "original_label_id": original_label_id, # Safe to use now
                    "final_label_id": info.get('predicted_label_id', 'N/A'),
                    "turns": num_turns_this_sample,
                    "queries": current_sample_queries
                })
                break

        total_queries += current_sample_queries

        if not attack_successful_this_sample:
            print(f"  Failed. Attack did not succeed within {args.max_turns} turns.")
            results_data.append({
                "index": i,
                "status": "failure",
                "original_text": original_text, # Safe to use now
                "final_text": info.get('current_text', 'ERROR') if 'info' in locals() else 'N/A', # Get last text if available
                "original_label_id": original_label_id, # Safe to use now
                "final_label_id": info.get('predicted_label_id', 'N/A') if 'info' in locals() else 'N/A',
                "turns": num_turns_this_sample,
                "queries": current_sample_queries
            })

    env.close() # Close the single environment instance
    end_time = time.time()
    print("--- Evaluation Loop Finished ---")

    # 6. Calculate and Print Metrics
    num_evaluated = len(processed_test_dataset) - skipped_samples
    print("\n--- Evaluation Results ---")
    print(f"Total samples in split: {len(processed_test_dataset)}")
    print(f"Samples skipped (initial misclassification/error): {skipped_samples}")
    print(f"Samples evaluated: {num_evaluated}")

    if num_evaluated > 0:
        asr = (successful_attacks / num_evaluated) * 100 if num_evaluated > 0 else 0
        aqs = np.mean(successful_attack_turns) if successful_attack_turns else 0
        avg_queries_total = total_queries / num_evaluated if num_evaluated > 0 else 0
        avg_queries_success = np.mean([item['queries'] for item in results_data if item['status']=='success']) if successful_attacks > 0 else 0


        print(f"Attack Success Rate (ASR): {asr:.2f}% ({successful_attacks}/{num_evaluated})")
        print(f"Average Queries per Successful Attack (AQS): {aqs:.2f}" if successful_attack_turns else "N/A (No successful attacks)")
        print(f"Average Queries per Evaluated Sample (includes failures): {avg_queries_total:.2f}")
        print(f"Average Queries per Successful Sample (end-to-end): {avg_queries_success:.2f}" if successful_attacks > 0 else "N/A (No successful attacks)")

    else:
        print("No samples were successfully evaluated.")

    print(f"Total evaluation time: {end_time - start_time:.2f} seconds")

    # 7. Save Detailed Results (Optional)
    if args.output_file:
        print(f"Saving detailed results to: {args.output_file}")
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
            with open(args.output_file, 'w') as f:
                json.dump(results_data, f, indent=4)
            print("Results saved.")
        except Exception as e:
            print(f"Error saving results to {args.output_file}: {e}")

    print("--- Evaluation Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent for adversarial attacks.")

    # Required Args
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved PPO agent .zip file.")
    parser.add_argument("--llm_model_name", type=str, required=True, help="Hugging Face model identifier for the target LLM (must match training).")
    parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name (e.g., 'imdb').")

    # Optional Args
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use for evaluation (e.g., 'test', 'validation[:10%]').")
    parser.add_argument("--max_turns", type=int, default=25, help="Maximum attack turns allowed per sample (should match agent's training).")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples from the dataset split to evaluate (0 for all).")
    parser.add_argument("--output_file", type=str, default=None, help="Optional path to save detailed evaluation results (JSON format).")
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('cuda', 'cpu', 'auto').")

    args = parser.parse_args()
    evaluate_agent(args)