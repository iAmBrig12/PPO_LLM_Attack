import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv # Or DummyVecEnv for simpler debugging
from datasets import load_dataset
import torch

# Assuming llm_interface.py and adversarial_env.py are accessible
from llm_interface import LLMInterface
from adversarial_env import AdversarialEnv

def train_agent(args):
    """Loads components and trains the PPO agent."""

    print("--- Starting Training ---")
    print(f"Arguments: {args}")

    # 1. Load LLM Interface
    print(f"Loading LLM: {args.model_name}")
    llm_interface = LLMInterface(model_name=args.model_name, task="classification", device=args.device)

    # 2. Load Dataset
    print(f"Loading Dataset: {args.dataset_name}, Split: {args.dataset_split}")
    try:
        raw_dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    except Exception as e:
        print(f"Error loading dataset '{args.dataset_name}'. Make sure it exists and you have access.")
        print(f"Check Hugging Face datasets documentation. Example: 'sst2', 'imdb'")
        raise e

    # Preprocess dataset to match environment expectations ({'text': ..., 'label': 'LABEL_NAME'})
    # This needs adjustment based on the specific dataset's structure (column names)
    text_col = "text" if "text" in raw_dataset.column_names else "sentence" # Common variations
    label_col = "label"
    if text_col not in raw_dataset.column_names or label_col not in raw_dataset.column_names:
         raise ValueError(f"Dataset must contain '{text_col}' and '{label_col}' columns. Found: {raw_dataset.column_names}")

    # Map label IDs to names using the model's config
    id2label_func = lambda x: llm_interface.id2label.get(x, "Unknown")
    try:
        processed_dataset = [{"text": item[text_col], "label": id2label_func(item[label_col])} for item in raw_dataset]
        print(f"Processed {len(processed_dataset)} samples from the dataset.")
        if not processed_dataset:
             raise ValueError("Dataset processing resulted in zero samples.")
    except Exception as e:
        print(f"Error processing dataset. Ensure label IDs in '{label_col}' match the model config: {llm_interface.id2label}")
        print(f"Original error: {e}")
        raise e


    # 3. Create Vectorized Environment
    # Use SubprocVecEnv for parallel environments (faster training)
    # Use DummyVecEnv for easier debugging (runs sequentially)
    VecEnvClass = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv_SB3 # Use SB3 DummyVecEnv

    print(f"Creating {args.num_envs} vectorized environments...")
    # We need to pass arguments to the env constructor within make_vec_env
    env_kwargs = dict(llm_interface=llm_interface, dataset=processed_dataset, max_turns=args.max_turns)
    env = make_vec_env(AdversarialEnv,
                       n_envs=args.num_envs,
                       seed=args.seed,
                       vec_env_cls=VecEnvClass,
                       env_kwargs=env_kwargs)

    print("Environments created.")

    # 4. Define and Train the Agent
    # IMPORTANT: The default 'MlpPolicy' assumes a flattened numerical observation space.
    # Our current simple Box(2,) observation space works, but a real implementation
    # should use embeddings and likely a custom policy network (e.g., using RNNs or Transformers).
    policy_type = 'MlpPolicy'
    print(f"Initializing PPO agent with policy: {policy_type}")
    agent = PPO(policy_type,
                env,
                verbose=1, # Print training progress
                tensorboard_log=args.log_dir,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                seed=args.seed,
                device=args.device # Pass device to agent too
               )

    print(f"Starting training for {args.total_timesteps} timesteps...")
    # Add callbacks if needed (e.g., EvalCallback for periodic evaluation)
    agent.learn(total_timesteps=args.total_timesteps,
                tb_log_name=f"ppo_{args.model_name.split('/')[-1]}_{args.dataset_name}") # Log name

    # 5. Save the Trained Agent
    save_path = f"{args.save_dir}/adversarial_ppo_{args.model_name.split('/')[-1]}_{args.dataset_name}"
    print(f"Training complete. Saving agent to {save_path}.zip")
    agent.save(save_path)

    # Clean up environments
    env.close()
    print("--- Training Finished ---")


# Wrapper needed if using DummyVecEnv because make_vec_env might expect it from SB3 directly
try:
    from stable_baselines3.common.vec_env import DummyVecEnv as DummyVecEnv_SB3
except ImportError:
    print("Warning: Could not import DummyVecEnv from SB3 directly. Ensure stable-baselines3 is installed.")
    # Fallback or define a simple wrapper if needed, though SubprocVecEnv is preferred for performance.
    class DummyVecEnv_SB3: # Minimal placeholder if import fails
         def __init__(self, env_fns):
            self.envs = [fn() for fn in env_fns]
            # ... implement necessary DummyVecEnv methods ...
            raise NotImplementedError("Minimal DummyVecEnv placeholder used.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent for adversarial attacks on LLMs.")

    # Environment Args
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased-finetuned-sst-2-english", help="Hugging Face model identifier for the target LLM.")
    parser.add_argument("--dataset_name", type=str, default="sst2", help="Hugging Face dataset name (e.g., 'sst2', 'imdb').")
    parser.add_argument("--dataset_split", type=str, default="train[:5%]", help="Dataset split to use (e.g., 'train', 'validation', 'train[:1000]', 'train[:5%%]'). Use small subsets for faster testing.")
    parser.add_argument("--max_turns", type=int, default=10, help="Maximum modifications per episode.")

    # Training Args
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--total_timesteps", type=int, default=100_000, help="Total training timesteps.")
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
    import os
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train_agent(args)