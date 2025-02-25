import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os
import optparse
import pickle
from enum import Enum
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.distributions as distributions


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


##############################################
# Memory Class
##############################################

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


##############################################
# ActorCritic Class
##############################################

# The base of this implementation of the ActorCritic class was taken from the solution to exercise 08_Gym-PPO-solution/PPO.py
# of the Reinforcement Learning course WiSe 24/25 by Prof. Martius:
# Adjustments have been made
# - implementing multiple layers of NN
# - handling both continous and discrete actions
# - adding condition for evaluating the model with memory == None

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        # We can have continous or discrete action spaces:
        if has_continuous_action_space:
            self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.LayerNorm(n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.LayerNorm(n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Tanh()  # Use Tanh to bound actions to [-1, 1]
            )
            # Learnable log standard deviation parameter
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        else:
            self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.LayerNorm(n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.LayerNorm(n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.LayerNorm(n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.LayerNorm(n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):

        if self.has_continuous_action_space:
            state = torch.from_numpy(state).float().to(device)
            action_mean = self.action_layer(state)
            action_std = self.log_std.exp().expand_as(action_mean)  # Expand log_std to match action_mean
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()

            if memory is not None:
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(dist.log_prob(action).sum(-1))
            else:
                torch.no_grad()

            return action.detach().cpu().numpy()

        else:
            state = torch.from_numpy(state).float().to(device)
            action_probs = self.action_layer(state)
            dist = Categorical(action_probs)
            action = dist.sample()

            if memory is not None:
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(dist.log_prob(action).sum(-1))
            else:
                torch.no_grad()

            return action.item()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.action_layer(state)
            action_std = self.log_std.exp().expand_as(action_mean)  # Expand log_std to match action_mean
            dist = torch.distributions.Normal(action_mean, action_std)

            action_logprobs = dist.log_prob(action).sum(-1)  # Sum over action dimensions
            dist_entropy = dist.entropy().sum(-1)  # Sum over action dimensions

            state_value = self.value_layer(state)

            return action_logprobs, torch.squeeze(state_value), dist_entropy

        else:
            action_probs = self.action_layer(state)
            dist = Categorical(action_probs)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

            state_value = self.value_layer(state)

            return action_logprobs, torch.squeeze(state_value), dist_entropy




##############################################
# PPO Class
##############################################

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

    # Monte Carlo estimate
    def mc_rewards(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype = torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        return rewards

    def update(self, memory):
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        rewards = self.mc_rewards(memory)

        # optimize policy for K epochs:
        for _ in range(self.K_epochs):

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # ratio of probability between the old and the new policies
            # = importance weights (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # advantages:
            advantages = rewards - state_values.detach()

            # clipped loss:
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_clipped = -torch.min(surr1, surr2)

            # value function loss:
            loss_vf = self.MseLoss(state_values, rewards)

            # total PPO Loss:
            loss = loss_clipped + 0.5 * loss_vf - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_checkpoint(self, checkpoint_dir, episode):
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'action_std': self.action_std if self.has_continuous_action_space else None
        }, f"{checkpoint_dir}/checkpoint_{episode}.pth")
        print(f"Checkpoint saved at {checkpoint_dir}/checkpoint_{episode}.pth")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        if self.has_continuous_action_space and 'action_std' in checkpoint:
            self.set_action_std(checkpoint['action_std'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    def save_final_model(self, model_path):
        """Save only the trained model's weights at the specified path."""
        torch.save(self.policy.state_dict(), model_path)
        print(f"Final trained model saved as {model_path}")
