import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

class DummyExtractor(torch.nn.Module):
    """
    Dummy MLP extractor for PPO that simply passes features along
    but defines latent_dim_pi and latent_dim_vf attributes.
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        self.latent_dim_pi = feature_dim
        self.latent_dim_vf = feature_dim

    def forward(self, features):
        return features, features


class CustomNetwork(torch.nn.Module):
    """
    Custom network for processing the LLM adversarial environment observations.
    """
    def __init__(self, 
                 feature_dim: int,
                 embedding_dim: int = 768):
        super(CustomNetwork, self).__init__()
        
        # Process prompt tokens
        self.token_embedding = torch.nn.Embedding(30000, 64)  # Assuming vocab size up to 30000
        self.token_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(128, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        
        # Process response embedding
        self.response_encoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU()
        )
        
        # Process label information
        self.label_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 16),  # 2 inputs: original_label and current_label
            torch.nn.ReLU()
        )
        
        # Combine all features
        combined_dim = 128 + 128 + 16  # token_features + response_features + label_features
        self.combined_encoder = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, feature_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, observations):
        # Process prompt tokens
        prompt_tokens = observations['prompt_tokens'].long()
        token_emb = self.token_embedding(prompt_tokens)
        token_emb = token_emb.permute(0, 2, 1)  # [batch, embed_dim, seq_len]
        token_features = self.token_encoder(token_emb).squeeze(-1)  # [batch, 128]
        
        # Process response embedding
        response_emb = observations['response_embedding']
        response_features = self.response_encoder(response_emb)  # [batch, 128]
        
        # Process labels
        labels = torch.stack([
            observations['original_label'].float(),
            observations['current_label'].float()
        ], dim=1)  # [batch, 2]
        label_features = self.label_encoder(labels)  # [batch, 16]
        
        # Combine all features
        combined = torch.cat([token_features, response_features, label_features], dim=1)
        features = self.combined_encoder(combined)
        
        return features

class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom policy for the LLM adversarial task.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        feature_dim = 256

        self.mlp_extractor = DummyExtractor(feature_dim=feature_dim)

        # Policy network
        policy_net_layers = []
        current_dim = feature_dim

        for layer_dim in self.net_arch["pi"]:
            policy_net_layers.append(torch.nn.Linear(current_dim, layer_dim))
            policy_net_layers.append(torch.nn.Tanh())
            current_dim = layer_dim

        self.policy_net = torch.nn.Sequential(*policy_net_layers)

        # Value network
        value_net_layers = []
        current_dim = feature_dim

        for layer_dim in self.net_arch["vf"]:
            value_net_layers.append(torch.nn.Linear(current_dim, layer_dim))
            value_net_layers.append(torch.nn.Tanh())
            current_dim = layer_dim

        self.value_net = torch.nn.Sequential(*value_net_layers)

        # Custom multi-head action outputs
        self.custom_action_net = torch.nn.ModuleDict({
            "operation": torch.nn.Linear(current_dim, 4),
            "position": torch.nn.Linear(current_dim, 50),
            "token_id": torch.nn.Linear(current_dim, 30522)
        })



    def forward(self, obs, deterministic=False):
        features = self.features_extractor(obs)

        # Actor head
        pi_latent = self.policy_net(features)

        # Compute logits separately
        operation_logits = self.custom_action_net["operation"](pi_latent)
        position_logits = self.custom_action_net["position"](pi_latent)
        token_id_logits = self.custom_action_net["token_id"](pi_latent)

        # 🔥 Concatenate all logits together along dim=1
        all_logits = torch.cat([operation_logits, position_logits, token_id_logits], dim=1)

        # Stable-Baselines3 expects a single big tensor
        distribution = self.action_dist.proba_distribution(action_logits=all_logits)

        # Sample action
        actions = distribution.get_actions(deterministic=deterministic)

        # Value head
        values = self.value_net(features)

        # Now compute log_probs for sampled actions
        log_probs = distribution.log_prob(actions)

        return actions, values, log_probs


def train_adversarial_agent(env, total_timesteps=10000, learning_rate=0.0003, n_steps=2048):
    """
    Train the adversarial RL agent.
    
    Args:
        env: The LLM adversarial environment
        total_timesteps: Total number of timesteps to train
        learning_rate: Learning rate for the optimizer
        n_steps: Number of steps to run for each environment per update
        
    Returns:
        The trained PPO agent
    """
    policy_kwargs = {
        "net_arch": {
            "pi": [256, 256],
            "vf": [256, 256]
        }
    }
    
    agent = PPO(
        CustomActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    agent.learn(total_timesteps=total_timesteps)
    return agent