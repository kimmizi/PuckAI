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
from torch.distributions import Categorical, Beta
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
# - adding Beta distribution for parameterizing policy action space instead of Normal distribution

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var_actor, n_latent_var_critic, network_depth_actor, network_depth_critic, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Activation function
        activation_function = nn.Tanh

        # Actor Network Initialization
        self.action_layer = []
        self.action_layer.append(nn.Linear(state_dim, n_latent_var_actor))
        self.action_layer.append(nn.LayerNorm(n_latent_var_actor)),
        self.action_layer.append(activation_function())

        for i in range(network_depth_actor):
            self.action_layer.append(nn.Linear(n_latent_var_actor, n_latent_var_actor))
            self.action_layer.append(nn.LayerNorm(n_latent_var_actor)),
            self.action_layer.append(activation_function())

        self.action_layer.append(nn.Linear(n_latent_var_actor, action_dim))
        if has_continuous_action_space:
            self.action_layer.append(nn.Tanh())


        # Critic Network Initialization
        self.value_layer = []
        self.value_layer.append(nn.Linear(state_dim, n_latent_var_critic))
        self.value_layer.append(nn.LayerNorm(n_latent_var_critic)),
        self.value_layer.append(activation_function())

        for i in range(network_depth_critic):
            self.value_layer.append(nn.Linear(n_latent_var_critic, n_latent_var_critic))
            self.value_layer.append(nn.LayerNorm(n_latent_var_critic)),
            self.value_layer.append(activation_function())

        self.value_layer.append(nn.Linear(n_latent_var_critic, 1))

        self.actor = nn.Sequential(*self.action_layer)
        self.critic = nn.Sequential(*self.value_layer)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):

        if self.has_continuous_action_space:
            state = torch.from_numpy(state).float().to(device)
            action_mean = self.actor(state)
            action_std = self.log_std.exp().expand_as(action_mean)  # Expand log_std to match action_mean
            # dist = torch.distributions.Normal(action_mean, action_std)
            # action = dist.sample()

            #######################
            ### ADJUSTMENT TO VANILLA PPO:
            ### **SOURCE:** Hsu et al. (2020)
            ### https://arxiv.org/abs/2009.10897
            # Using Beta distribution for parameterizing policy action space instead of Normal distribution
            # alpha, beta > 1: log(1 + exp(x)) + 1
            alpha = torch.log(1 + torch.exp(action_mean)) + 1
            beta = torch.log(1 + torch.exp(action_std)) + 1
            dist = Beta(alpha, beta)
            action = dist.sample()
            #######################

            if memory is not None:
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(dist.log_prob(action).sum(-1))
            else:
                torch.no_grad()

            return action.detach().cpu().numpy()

        else:
            state = torch.from_numpy(state).float().to(device)
            action_probs = self.actor(state)
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
            action_mean = self.actor(state)
            action_std = self.log_std.exp().expand_as(action_mean)  # Expand log_std to match action_mean
            dist = torch.distributions.Normal(action_mean, action_std)

            action_logprobs = dist.log_prob(action).sum(-1)  # Sum over action dimensions
            dist_entropy = dist.entropy().sum(-1)  # Sum over action dimensions

            state_value = self.critic(state)

            return action_logprobs, torch.squeeze(state_value), dist_entropy

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

            state_value = self.critic(state)

            return action_logprobs, torch.squeeze(state_value), dist_entropy




##############################################
# PPO Class
##############################################

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var_actor, n_latent_var_critic, network_depth_actor, network_depth_critic, has_continuous_action_space, action_std_init,
                 lr, betas, gamma, K_epochs, eps_clip, c1, c2, beta_clone):

        self.has_continuous_action_space = has_continuous_action_space

        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.c1 = c1
        self.c2 = c2
        self.beta_clone = beta_clone


        self.policy = ActorCritic(state_dim, action_dim, n_latent_var_actor, n_latent_var_critic, network_depth_actor, network_depth_critic, has_continuous_action_space, action_std_init).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var_actor, n_latent_var_critic, network_depth_actor, network_depth_critic, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = lr, betas = betas)
        self.aux_optimizer = torch.optim.Adam(self.policy.parameters(), lr = lr, betas = betas)
        self.MseLoss = nn.MSELoss()

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

    #######################
    ### ADJUSTMENT TO VANILLA PPO:
    ### **SOURCE:** Schulman et al. (2018)
    ### https://arxiv.org/abs/1506.02438
    # Using Generalized Advantage Estimation (GAE) for advantage calculation
    def gae(self, memory):
        state_values = self.policy.critic(torch.stack(memory.states).to(device)).detach()
        next_state_values = self.policy.critic(torch.stack(memory.states + [memory.states[-1]]).to(device)).detach()
        deltas = [r + self.gamma * next_v - v for r, next_v, v in zip(memory.rewards, next_state_values, state_values)]

        advantages = []
        advantage = 0

        for delta in deltas[::-1]:
            advantage = delta + self.gamma * advantage
            advantages.insert(0, advantage)

        advantages = torch.tensor(advantages, dtype = torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        return advantages
    #######################

    # Monte Carlo estimate
    def mc_rewards(self, memory):
        old_states = torch.stack(memory.states).to(device).detach()

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

        # GAE for advantages and Monte Carlo for rewards
        advantages = self.gae(memory)
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
            loss = loss_clipped + self.c1 * loss_vf - self.c2 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    #######################
    #### ADJUSTMENT TO VANILLA PPO:
    ### **SOURCE:** Cobbe et al. (2020)
    ### https://arxiv.org/abs/2009.04416
    # Implementing the auxiliary phase, where we train the value function
    def auxiliary_phase(self, memory):

        if not memory.states:
            print("Warning: No states available, skipping auxiliary phase.")
            return

        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # compute state values from old policy
        with torch.no_grad():
            old_values = self.policy_old.evaluate(old_states, old_actions)[1].squeeze()

        # Monte Carlo estimate of state rewards
        rewards = self.mc_rewards(memory)

        # train the value function
        for _ in range(self.K_epochs):

            # Auxiliary loss: L_aux = 0.5 * ^E[(V(s) - V_targ)^2
            _, state_values, _ = self.policy.evaluate(old_states, old_actions)
            aux_loss = 0.5 * self.MseLoss(state_values, rewards)

            # Behavioral cloning loss: Lbc = 0.5 * ^E[(V(s) - V_old(s))^2
            with torch.no_grad():
                _, old_state_values, _ = self.policy_old.evaluate(old_states, old_actions)
            bc_loss = self.MseLoss(state_values, old_state_values)

            # Joint loss
            total_loss = aux_loss + self.beta_clone * bc_loss

            # take gradient step
            self.aux_optimizer.zero_grad()
            total_loss.mean().backward()
            self.aux_optimizer.step()
       #######################

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
