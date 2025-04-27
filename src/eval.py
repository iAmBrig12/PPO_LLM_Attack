import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import time
import datasets

class AdversarialAttackEvaluator:
    """
    Framework for evaluating the effectiveness of adversarial attacks against LLMs.
    """
    
    def __init__(self, 
                 target_models,
                 datasets_config,
                 agent=None,
                 random_baseline=True):
        """
        Initialize the evaluator.
        
        Args:
            target_models: List of target LLM models to evaluate against
            datasets_config: Dict mapping task names to dataset configurations
            agent: Trained RL agent (optional)
            random_baseline: Whether to evaluate a random baseline
        """
        self.target_models = target_models
        self.datasets_config = datasets_config
        self.agent = agent
        self.random_baseline = random_baseline
        self.results = defaultdict(dict)
        
    def load_dataset(self, task, dataset_name, split="test"):
        """
        Load a dataset for evaluation.
        """
        # Use the Hugging Face datasets library
        if dataset_name == "imdb":
            dataset = datasets.load_dataset("imdb", split=split)
        elif dataset_name == "sst2":
            dataset = datasets.load_dataset("glue", "sst2", split=split)
        elif dataset_name == "mrpc":
            dataset = datasets.load_dataset("glue", "mrpc", split=split)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        return dataset
    
    def evaluate_random_baseline(self, env, num_examples=100, max_steps=5):
        """
        Evaluate a random action baseline.
        """
        success_count = 0
        steps_to_success = []
        
        for _ in tqdm(range(num_examples), desc="Evaluating random baseline"):
            obs, _ = env.reset()
            success = False
            
            for step in range(max_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if info.get('success', False):
                    success = True
                    steps_to_success.append(step + 1)
                    break
                
                if terminated or truncated:
                    break
            
            if success:
                success_count += 1
        
        success_rate = success_count / num_examples
        avg_steps = np.mean(steps_to_success) if steps_to_success else float('inf')
        
        return {
            "success_rate": success_rate,
            "avg_steps_to_success": avg_steps
        }
    
    def evaluate_agent(self, env, agent, num_examples=100):
        """
        Evaluate the RL agent.
        """
        success_count = 0
        steps_to_success = []
        modified_prompts = []
        
        for _ in tqdm(range(num_examples), desc="Evaluating RL agent"):
            obs, _ = env.reset()
            success = False
            
            done = False
            step = 0
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if info.get('success', False):
                    success = True
                    steps_to_success.append(step + 1)
                    modified_prompts.append({
                        'original': info['original_text'],
                        'adversarial': info['current_prompt']
                    })
                    break
                
                step += 1
                done = terminated or truncated
            
            if success:
                success_count += 1
        
        success_rate = success_count / num_examples
        avg_steps = np.mean(steps_to_success) if steps_to_success else float('inf')
        
        return {
            "success_rate": success_rate,
            "avg_steps_to_success": avg_steps,
            "example_prompts": modified_prompts[:5]  # Save a few examples
        }
    
    def run_evaluation(self, num_examples=100):
        """
        Run a comprehensive evaluation across all models and datasets.
        """
        for model_name in self.target_models:
            print(f"Evaluating model: {model_name}")
            
            for task, dataset_config in self.datasets_config.items():
                dataset_name = dataset_config['name']
                print(f"  Task: {task}, Dataset: {dataset_name}")
                
                # Create environment for this model and dataset
                env = LLMAdversarialEnv(
                    target_model_name=model_name,
                    task=task,
                    dataset_name=dataset_name,
                    max_steps=5
                )
                
                # Evaluate random baseline if requested
                if self.random_baseline:
                    baseline_results = self.evaluate_random_baseline(
                        env, num_examples=num_examples
                    )
                    self.results[model_name][f"{task}_{dataset_name}_random"] = baseline_results
                
                # Evaluate agent if provided
                if self.agent:
                    agent_results = self.evaluate_agent(
                        env, self.agent, num_examples=num_examples
                    )
                    self.results[model_name][f"{task}_{dataset_name}_agent"] = agent_results
        
        return self.results
    
    def generate_report(self, output_file=None):
        """
        Generate a comprehensive report of the evaluation results.
        """
        report = {
            "summary": {},
            "details": self.results
        }
        
        # Compute summary statistics
        for model_name in self.target_models:
            model_results = self.results[model_name]
            
            # Compute average success rates
            random_success_rates = [
                result["success_rate"] for key, result in model_results.items()
                if key.endswith("_random")
            ]
            
            agent_success_rates = [
                result["success_rate"] for key, result in model_results.items()
                if key.endswith("_agent")
            ]
            
            report["summary"][model_name] = {
                "avg_random_success_rate": np.mean(random_success_rates) if random_success_rates else None,
                "avg_agent_success_rate": np.mean(agent_success_rates) if agent_success_rates else None,
                "improvement": np.mean(agent_success_rates) - np.mean(random_success_rates)
                if agent_success_rates and random_success_rates else None
            }
        
        # Save to file if requested
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def plot_results(self, output_file=None):
        """
        Plot the evaluation results.
        """
        models = self.target_models
        tasks = list(self.datasets_config.keys())
        
        # Prepare data for plotting
        random_data = []
        agent_data = []
        
        for model in models:
            model_results = self.results[model]
            
            for task in tasks:
                dataset_name = self.datasets_config[task]['name']
                
                # Get random baseline results
                random_key = f"{task}_{dataset_name}_random"
                if random_key in model_results:
                    random_data.append({
                        'model': model,
                        'task': task,
                        'success_rate': model_results[random_key]['success_rate']
                    })
                
                # Get agent results
                agent_key = f"{task}_{dataset_name}_agent"
                if agent_key in model_results:
                    agent_data.append({
                        'model': model,
                        'task': task,
                        'success_rate': model_results[agent_key]['success_rate']
                    })
        
        # Convert to DataFrame for easier plotting
        random_df = pd.DataFrame(random_data)
        agent_df = pd.DataFrame(agent_data)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        width = 0.35
        x = np.arange(len(tasks))
        
        for i, model in enumerate(models):
            random_rates = random_df[random_df['model'] == model]['success_rate'].values
            agent_rates = agent_df[agent_df['model'] == model]['success_rate'].values
            
            plt.bar(x - width/2 + i*width/len(models), random_rates, 
                    width/len(models), label=f'{model} (Random)')
            plt.bar(x + width/2 + i*width/len(models), agent_rates, 
                    width/len(models), label=f'{model} (Agent)')
        
        plt.xlabel('Tasks')
        plt.ylabel('Success Rate')
        plt.title('Adversarial Attack Success Rate by Model and Task')
        plt.xticks(x, tasks)
        plt.legend()
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()