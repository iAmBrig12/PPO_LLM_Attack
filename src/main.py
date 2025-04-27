import os
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Import our custom environment
from llm_adversarial_env import LLMAdversarialEnv

def load_real_examples(dataset_name, task, split="test", num_examples=100):
    """Load real examples from datasets."""
    if dataset_name == "sst2":
        dataset = load_dataset("glue", "sst2", split=split)
        # Select num_examples random samples
        indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
        examples = []
        for i in indices:
            examples.append({
                "text": dataset[i]["sentence"],
                "label": dataset[i]["label"]
            })
    elif dataset_name == "imdb":
        dataset = load_dataset("imdb", split=split)
        indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
        examples = []
        for i in indices:
            examples.append({
                "text": dataset[i]["text"],
                "label": dataset[i]["label"]
            })
    elif dataset_name == "mrpc":
        dataset = load_dataset("glue", "mrpc", split=split)
        indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
        examples = []
        for i in indices:
            examples.append({
                "text": dataset[i]["sentence1"] + " [SEP] " + dataset[i]["sentence2"],
                "label": dataset[i]["label"]
            })
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return examples

def evaluate_attack(env, agent=None, num_examples=50):
    """Evaluate attack effectiveness."""
    success_count = 0
    steps_to_success = []
    examples = []
    
    for _ in tqdm(range(num_examples), desc="Evaluating attacks"):
        obs, _ = env.reset()
        original_text = env.original_text
        original_label = env.original_label
        success = False
        
        for step in range(env.max_steps):
            if agent:
                # Use the trained agent
                action, _ = agent.predict(obs, deterministic=True)
            else:
                # Use random actions as baseline
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info.get('success', False):
                success = True
                steps_to_success.append(step + 1)
                examples.append({
                    "original_text": original_text,
                    "adversarial_text": info['current_prompt'],
                    "original_label": original_label,
                    "new_label": obs['current_label']
                })
                break
            
            if terminated or truncated:
                break
        
        if success:
            success_count += 1
    
    success_rate = success_count / num_examples
    avg_steps = np.mean(steps_to_success) if steps_to_success else float('inf')
    
    return {
        "success_rate": success_rate,
        "avg_steps_to_success": avg_steps,
        "examples": examples
    }

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and evaluate RL-based adversarial attacks on LLMs")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased-finetuned-sst-2-english",
                        help="Target model name (Hugging Face model ID)")
    parser.add_argument("--task", type=str, default="sentiment", 
                        choices=["sentiment", "paraphrase", "classification"],
                        help="Task type")
    parser.add_argument("--dataset", type=str, default="sst2",
                        choices=["sst2", "imdb", "mrpc"],
                        help="Dataset name")
    parser.add_argument("--model_type", type=str, default="classification",
                        choices=["classification", "generation"],
                        help="Model type (classification or generation)")
    parser.add_argument("--train", action="store_true", help="Train the RL agent")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate attack effectiveness")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps")
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "run.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Create environment
    logger.info(f"Creating environment with model {args.model}")
    env = LLMAdversarialEnv(
        target_model_name=args.model,
        model_type=args.model_type,
        task=args.task,
        dataset_name=args.dataset,
        max_tokens=50,
        max_steps=5
    )
    
    # Train the agent if requested
    agent = None
    if args.train:
        logger.info("Training adversarial agent...")
        
        # Set up checkpointing
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=os.path.join(args.output_dir, "checkpoints"),
            name_prefix="adversarial_agent"
        )
        
        # Create and train the agent
        agent = PPO(
            "MultiInputPolicy", 
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99
        )
        
        agent.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback
        )
        
        # Save the final model
        agent_path = os.path.join(args.output_dir, "final_model")
        agent.save(agent_path)
        logger.info(f"Agent saved to {agent_path}")
    
    # Evaluate if requested
    if args.evaluate:
        logger.info("Evaluating attack effectiveness...")
        
        # Evaluate random baseline
        logger.info("Evaluating random baseline...")
        random_results = evaluate_attack(env, agent=None, num_examples=50)
        logger.info(f"Random baseline success rate: {random_results['success_rate']:.2f}")
        
        # Evaluate trained agent if available
        if agent is not None:
            logger.info("Evaluating trained agent...")
            agent_results = evaluate_attack(env, agent=agent, num_examples=50)
            logger.info(f"Agent success rate: {agent_results['success_rate']:.2f}")
            
            # Save examples of successful attacks
            import json
            with open(os.path.join(args.output_dir, "successful_attacks.json"), "w") as f:
                json.dump(agent_results["examples"], f, indent=2)

if __name__ == "__main__":
    main()