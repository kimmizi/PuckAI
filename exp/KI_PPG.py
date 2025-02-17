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
# policy (actor) & value function (critic)
##############################################

# The base of this implementation of the ActorCritic class was taken from the solution to exercise 08_Gym-PPO-solution/PPO.py
# of the Reinforcement Learning course WiSe 24/25 by Prof. Martius:

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

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

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action).sum(-1))  # Sum over action dimensions

            return action.detach().cpu().numpy()

        else:
            state = torch.from_numpy(state).float().to(device)
            action_probs = self.action_layer(state)
            dist = Categorical(action_probs)
            action = dist.sample()

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

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
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, has_continuous_action_space = True, action_std_init = 0.6):
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

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
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



class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # Policy network
        self.policy_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.LayerNorm(n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.LayerNorm(n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Tanh() if has_continuous_action_space else nn.Softmax(dim=-1)
        )

        # Value network
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.LayerNorm(n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.LayerNorm(n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

        # Auxiliary value head (for PPG)
        self.aux_value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.LayerNorm(n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.LayerNorm(n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self, state):
        action_probs = self.policy_layer(state)
        state_value = self.value_layer(state)
        aux_state_value = self.aux_value_layer(state)
        return action_probs, state_value, aux_state_value

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs, state_value, _ = self.forward(state)

        if self.has_continuous_action_space:
            dist = torch.distributions.Normal(action_probs, self.action_var)
        else:
            dist = Categorical(action_probs)

        action = dist.sample()
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.detach().cpu().numpy()

    def evaluate(self, state, action):
        action_probs, state_value, aux_state_value = self.forward(state)

        if self.has_continuous_action_space:
            dist = torch.distributions.Normal(action_probs, self.action_var)
        else:
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy, torch.squeeze(aux_state_value)


class PPG(PPO):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip,
                 has_continuous_action_space=True, action_std_init=0.6):
        super(PPG, self).__init__(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip,
                                  has_continuous_action_space, action_std_init)

        # Additional parameters for PPG
        self.aux_epochs = 5  # Number of epochs for auxiliary phase
        self.clone_coef = 0.1  # Coefficient for behavioral cloning loss

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Policy Phase (PPO)
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy, _ = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Auxiliary Phase
        for _ in range(self.aux_epochs):
            _, state_values, _, aux_state_values = self.policy.evaluate(old_states, old_actions)

            # Auxiliary value loss
            aux_value_loss = 0.5 * self.MseLoss(aux_state_values, rewards)

            # Behavioral cloning loss
            _, old_state_values, _, _ = self.policy_old.evaluate(old_states, old_actions)
            kl_div = F.kl_div(F.log_softmax(self.policy.policy_layer(old_states), F.softmax(self.policy_old.policy_layer(old_states)), reduction='batchmean'))

            # Joint loss
            joint_loss = aux_value_loss + self.clone_coef * kl_div

            self.optimizer.zero_grad()
            joint_loss.mean().backward()
            self.optimizer.step()

            # Copy new weights into old policy:
            self.policy_old.load_state_dict(self.policy.state_dict())