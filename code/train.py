import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from datasets import load_dataset
import torch 
from llm_interface import LLMInterface
from adversarial_env import AdversarialEnv

# --- Define Dataset Label Mappings ---
DATASET_LABEL_MAPS = {
    "sst2": {0: "NEGATIVE", 1: "POSITIVE"},
    "mrpc": {0: "not_equivalent", 1: "equivalent"},
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"}
}

def get_dataset_mapping(dataset_name):
    name_lower = dataset_name.lower()
    if name_lower in DATASET_LABEL_MAPS:
        return DATASET_LABEL_MAPS[name_lower]
    else:
        print(f"Warning: No predefined label mapping found for dataset '{dataset_name}'. Attempting to use model's default or numerical labels.")
        return None 


def train_agent(args):
    """Loads components and trains the PPO agent."""
    print("--- Starting Training ---")
    print(f"Arguments: {args}")

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {args.device}")

    # 1. Load LLM Interface
    print(f"Loading LLM: {args.model_name}")
    llm_interface = LLMInterface(model_name=args.model_name, task="classification", device=args.device)

    # --- Get Dataset Label Mapping ---
    dataset_label_mapping = get_dataset_mapping(args.dataset_name)

    # --- Optional: Override LLM config labels if needed ---
    # If a specific dataset mapping exists, ensure the LLM interface uses these labels
    # This ensures consistency when the environment asks for label IDs by name.
    if dataset_label_mapping:
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
        # --- START: Handle GLUE datasets ---
        glue_datasets = ["mrpc", "sst2", "mnli", "qqp", "qnli", "rte", "wnli", "cola"] # List of GLUE tasks
        if args.dataset_name.lower() in glue_datasets:
            print(f"Recognized GLUE task: {args.dataset_name}. Loading from 'glue' collection.")
            # Load using the GLUE collection name and the specific task name
            raw_dataset = load_dataset("glue", args.dataset_name.lower(), split=args.dataset_split)
            # --- Identify text/label columns for GLUE/MRPC ---
            if args.dataset_name.lower() == "mrpc":
                text_col = ('sentence1', 'sentence2') 
            elif args.dataset_name.lower() == "mnli":
                text_col = ('premise', 'hypothesis')
            elif args.dataset_name.lower() == "sst2":
                text_col = 'sentence'
            else:
                # Add logic for other GLUE tasks if needed, or raise error
                 raise ValueError(f"Text column identification not implemented for GLUE task: {args.dataset_name}")
            label_col = "label"
        else:
            # Load non-GLUE dataset directly
            print(f"Loading dataset '{args.dataset_name}' directly.")
            raw_dataset = load_dataset(args.dataset_name, split=args.dataset_split)
            # Standard text/label column guessing for non-GLUE
            text_col = "text" if "text" in raw_dataset.column_names else "sentence"
            label_col = "label"
        # --- END: Handle GLUE datasets ---

    except Exception as e:
        print(f"Error loading dataset '{args.dataset_name}'. Make sure it exists and you have access.")
        print(f"If it's a GLUE task (like mrpc, sst2), ensure it's listed in glue_datasets.")
        raise e

    # Preprocess dataset
    # --- START: Handle single text vs. text pair ---
    if isinstance(text_col, tuple): # Handle pairs 
         if len(text_col) != 2 or any(col not in raw_dataset.column_names for col in text_col):
             raise ValueError(f"Expected text columns {text_col} not found in dataset {args.dataset_name}. Found: {raw_dataset.column_names}")
         print(f"Using text columns: {text_col}")
         # Combine the pair for the environment (or adapt env to handle pairs)
         # Simple combination using SEP token might work for BERT-like models
         sep_token = llm_interface.tokenizer.sep_token if llm_interface.tokenizer.sep_token else "[SEP]"
         get_text_func = lambda item: f"{item[text_col[0]]} {sep_token} {item[text_col[1]]}"
    else: # Handle single text column
         if text_col not in raw_dataset.column_names:
             raise ValueError(f"Expected text column '{text_col}' not found in dataset {args.dataset_name}. Found: {raw_dataset.column_names}")
         print(f"Using text column: '{text_col}'")
         get_text_func = lambda item: item[text_col]
    # --- END: Handle single text vs. text pair ---

    if label_col not in raw_dataset.column_names:
        raise ValueError(f"Expected label column '{label_col}' not found in dataset {args.dataset_name}. Found: {raw_dataset.column_names}")

    # Get label mapping (should already be defined in DATASET_LABEL_MAPS)
    dataset_label_mapping = get_dataset_mapping(args.dataset_name)
    if not dataset_label_mapping:
         # Fallback needed if map isn't predefined (though it should be for GLUE tasks)
         print(f"Warning: Using fallback label name conversion for {args.dataset_name}.")
         id2name_func = lambda x: llm_interface.id2label.get(x, str(x))
    else:
        print(f"Using label mapping: {dataset_label_mapping}")
        id2name_func = lambda x: dataset_label_mapping.get(x, f"UNKNOWN_LABEL_{x}")

    try:
        # Apply text extraction and label mapping
        processed_dataset = [{"text": get_text_func(item), "label": id2name_func(item[label_col])} for item in raw_dataset]
        print(f"Processed {len(processed_dataset)} samples. Example: {processed_dataset[0] if processed_dataset else 'N/A'}")
        if not processed_dataset:
             raise ValueError("Dataset processing resulted in zero samples.")
    except Exception as e:
        print(f"Error processing dataset. Check labels and mapping.")
        print(f"Original error: {e}")
        raise e

    # --- Update DATASET_LABEL_MAPS if needed ---
    # Make sure MRPC mapping exists
    if "mrpc" not in DATASET_LABEL_MAPS:
         DATASET_LABEL_MAPS["mrpc"] = {0: "not_equivalent", 1: "equivalent"} # Standard MRPC labels


    # 3. Create Vectorized Environment
    VecEnvClass = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv

    print(f"Creating {args.num_envs} vectorized environments...")
    env_kwargs = dict(llm_interface=llm_interface, dataset=processed_dataset, max_turns=args.max_turns)
    env = make_vec_env(AdversarialEnv,
                       n_envs=args.num_envs,
                       seed=args.seed,
                       vec_env_cls=VecEnvClass,
                       env_kwargs=env_kwargs)
    print("Environments created.")

    # 4. Define and Train the Agent
    # MlpPolicy might struggle with the new observation space if probabilities fluctuate wildly.
    # Consider monitoring performance closely or exploring alternative policies if needed.
    policy_type = 'MlpPolicy'
    print(f"Initializing PPO agent with policy: {policy_type}")
    agent = PPO(policy_type,
                env,
                verbose=1,
                tensorboard_log=args.log_dir,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                seed=args.seed,
                device=args.device
               )

    print(f"Starting training for {args.total_timesteps} timesteps...")
    log_name = f"ppo_shaped_{args.model_name.split('/')[-1]}_{args.dataset_name}"
    agent.learn(total_timesteps=args.total_timesteps,
                tb_log_name=log_name) # Log name reflects shaped reward

    # 5. Save the Trained Agent
    model_dir = os.path.join(args.save_dir, log_name) # Save in a subdirectory named after the log
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "final_model")
    print(f"Training complete. Saving agent to {save_path}.zip")
    agent.save(save_path)

    # Clean up environments
    env.close()
    print("--- Training Finished ---")


if __name__ == "__main__":
    import os 
    parser = argparse.ArgumentParser(description="Train an RL agent for adversarial attacks on LLMs.")

    # Environment Args
    parser.add_argument("--model_name", type=str, default="textattack/bert-base-uncased-imdb", help="Hugging Face model identifier for the target LLM.")
    parser.add_argument("--dataset_name", type=str, default="imdb", help="Hugging Face dataset name (e.g., 'sst2', 'imdb').")
    parser.add_argument("--dataset_split", type=str, default="train[:5%]", help="Dataset split to use (e.g., 'train', 'validation', 'train[:1000]', 'train[:5%]'). Use small subsets for faster testing.")
    parser.add_argument("--max_turns", type=int, default=15, help="Maximum modifications per episode.") 

    # Training Args
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="Total training timesteps.") 
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for PPO.")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps to run for each environment per update.")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size for PPO.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of optimization epochs per update.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Factor for trade-off Generalized Advantage Estimation.")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping parameter for PPO.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cuda', 'cpu', 'auto').")

    # Logging/Saving Args
    parser.add_argument("--log_dir", type=str, default="./tensorboard_logs/", help="Directory for Tensorboard logs.")
    parser.add_argument("--save_dir", type=str, default="./trained_models/", help="Directory to save trained models.")

    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    train_agent(args)