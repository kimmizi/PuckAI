import torch
import torch.nn as nn
# import torch.optim as optim
# import numpy as np
import os
# from torch.distributions import MultivariateNormal
from torch.distributions import Categorical, Beta
# import torch.nn.functional as F
# import torch.distributions as distributions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, state_dim, action_dim, n_latent_var_actor, n_latent_var_critic, has_continuous_action_space, action_std_init, network_depth_actor, network_depth_critic):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            self.log_std = nn.Parameter(torch.zeros(action_dim))


        # Activation function
        activation_function = nn.Tanh  # Default, can be modified

        # Actor Network Initialization
        self.action_layer = []

        self.action_layer.append(nn.Linear(state_dim, n_latent_var_actor))
        self.action_layer.append(activation_function())

        # Actor Network Initialization
        for i in range(network_depth_actor):
            self.action_layer.append(nn.Linear(n_latent_var_actor, n_latent_var_actor))
            self.action_layer.append(activation_function())

        self.action_layer.append(nn.Linear(n_latent_var_actor, action_dim))
        if has_continuous_action_space:
            self.action_layer.append(nn.Tanh())


        # self.action_layer = nn.Sequential(
        #     nn.Linear(state_dim, n_latent_var),
        #     nn.LayerNorm(n_latent_var),
        #     activation_function(),
        #     nn.Linear(n_latent_var, n_latent_var),
        #     nn.LayerNorm(n_latent_var),
        #     activation_function(),
        #     nn.Linear(n_latent_var, action_dim),
        #     nn.Tanh() if has_continuous_action_space else nn.Softmax(dim=-1)
        # )

        # Critic Network Initialization
        self.value_layer = []

        self.value_layer.append(nn.Linear(state_dim, n_latent_var_critic))
        self.value_layer.append(activation_function())

        for i in range(network_depth_critic):
            self.value_layer.append(nn.Linear(n_latent_var_critic, n_latent_var_critic))
            self.value_layer.append(activation_function())

        self.value_layer.append(nn.Linear(n_latent_var_critic, 1))


        # self.value_layer = nn.Sequential(
        #     nn.Linear(state_dim, n_latent_var),
        #     nn.LayerNorm(n_latent_var),
        #     activation_function(),
        #     nn.Linear(n_latent_var, n_latent_var),
        #     nn.LayerNorm(n_latent_var),
        #     activation_function(),
        #     nn.Linear(n_latent_var, 1)
        # )

        self.critic = nn.Sequential(*self.value_layer)

        # Initialization
        for layer in self.action_layer[:-2]:
            if isinstance(layer, nn.Linear):
                # torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
                torch.nn.init.orthogonal_(layer.weight, gain=0.01)

        for layer in self.value_layer[:-1]:
            if isinstance(layer, nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=0.01)

        self.action_layer = nn.Sequential(*self.action_layer)

        self.value_layer = nn.Sequential(*self.value_layer)

        self.to(device)

    def set_action_std(self, new_action_std):
        """Update action standard deviation"""
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
            self.log_std.data.fill_(torch.log(torch.tensor(new_action_std)))  # Update log_std

    def act(self, state, memory):
        if self.has_continuous_action_space:
            state = torch.from_numpy(state).float().to(device)
            action_mean = self.action_layer(state)
            action_std = self.log_std.exp().expand_as(action_mean)
            # dist = torch.distributions.Normal(action_mean, action_std)
            # action = dist.sample()

            # alpha, beta > 1: log(1 + exp(x)) + 1
            alpha = torch.log(1 + torch.exp(action_mean)) + 1
            beta = torch.log(1 + torch.exp(action_std)) + 1

            dist = Beta(alpha, beta)
            action = dist.sample()

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action).sum(-1))

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
            action_std = self.log_std.exp().expand_as(action_mean)
            dist = torch.distributions.Normal(action_mean, action_std)

            action_logprobs = dist.log_prob(action).sum(-1)
            dist_entropy = dist.entropy().sum(-1)
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

class PPO_init:
    def __init__(self, state_dim, action_dim, n_latent_var_actor, n_latent_var_critic, lr, betas, gamma, K_epochs, eps_clip, has_continuous_action_space,
                 action_std_init, c1, c2, beta_clone, network_depth_actor, network_depth_critic):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var_actor, n_latent_var_critic, has_continuous_action_space, action_std_init, network_depth_actor, network_depth_critic).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)  # Reduce LR every 10 epochs
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var_actor, n_latent_var_critic, has_continuous_action_space, action_std_init, network_depth_actor, network_depth_critic).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.has_continuous_action_space = has_continuous_action_space

        self.c1 = c1
        self.c2 = c2

        self.beta_clone = beta_clone

        self.aux_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)  # Separate optimizer for auxiliary phase

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
            # self.action_std = self.action_std - action_std_decay_rate
            # self.action_std = round(self.action_std, 4)
            # if (self.action_std <= min_action_std):
            #     self.action_std = min_action_std
            #     print("setting actor output action_std to min_action_std : ", self.action_std)
            # else:
            #     print("setting actor output action_std to : ", self.action_std)
            self.action_std = max(self.action_std * action_std_decay_rate, min_action_std)
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
        rewards = torch.clamp(rewards, min=-10, max=10)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        # old_states = torch.stack(memory.states).to(device).detach()
        # old_actions = torch.stack(memory.actions).to(device).detach()
        # old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        old_states = torch.squeeze(torch.stack([memory.states[i] for i in range(len(memory.states))])).to(device).detach()
        old_actions = torch.squeeze(torch.stack([memory.actions[i] for i in range(len(memory.actions))])).to(device).detach()
        old_logprobs = torch.squeeze(torch.stack([memory.logprobs[i] for i in range(len(memory.logprobs))])).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + self.c1 * self.MseLoss(state_values, rewards) - self.c2 * dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm = 0.2)  # Gradient clipping
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def auxiliary_phase(self, memory):
        """
        Train the value function with distillation loss.
        """
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Monte Carlo estimate of state rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = torch.clamp(rewards, min=-10, max=10)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Train the value function
        for _ in range(self.K_epochs):

            # Auxiliary loss: Laux = 0.5 * ^E[(V(s) - V_targ)^2
            _, state_values, _ = self.policy.evaluate(old_states, old_actions)
            aux_loss = 0.5 * self.MseLoss(state_values, rewards)

            # Behavioral cloning loss: Lbc = 0.5 * ^E[(V(s) - V_old(s))^2
            with torch.no_grad():
                _, old_state_values, _ = self.policy_old.evaluate(old_states, old_actions)
            bc_loss = self.MseLoss(state_values, old_state_values)

            # Joint loss
            total_loss = aux_loss + self.beta_clone * bc_loss

            # Take gradient step
            self.aux_optimizer.zero_grad()
            total_loss.mean().backward()
            self.aux_optimizer.step()

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
