import hockey.hockey_env as h_env
from hockey.hockey_env import Mode
import gymnasium as gym
import numpy as np
import random
import torch
import matplotlib.pyplot as plt


env_hockey = h_env.HockeyEnv_BasicOpponent(mode = Mode.NORMAL, weak_opponent = False)
state_dim = env_hockey.observation_space.shape[0]
action_dim = env_hockey.action_space.shape[0]


from PPG_GAE_KL import PPO, Memory

ppg_KL = PPO(
    state_dim,
    action_dim,
    n_latent_var_actor = 128,
    n_latent_var_critic = 128,
    network_depth_actor = 2,
    network_depth_critic = 2,
    has_continuous_action_space = True,
    action_std_init = 0.5,
    lr = 0.0003,
    betas = (0.9, 0.999),
    gamma = 0.99,
    K_epochs = 10,
    eps_clip = 0.25,
    c1 = 0.5,
    c2 = 0.01,
    beta_clone = 0.95
)

episodes = 5000
ppg_KL.load_checkpoint("checkpoints_ppg_kl/checkpoint_final.pth")
print("Model loaded")

checkpoint_freq = 1000
checkpoint_dir = "./checkpoints_ppg_kl"

update_timestep = 500


def train_hockey_ppg(env, max_episodes, max_timesteps, update_timestep, aux_phase_freq, agent):
    memory = Memory()
    episode_rewards = []
    info_list = []

    for episode in range(episodes, max_episodes):
        # state, _ = env.reset()
        # state = state.flatten()
        done = False
        episode_reward = 0

        # Randomly choose weak or strong opponent
        env = h_env.HockeyEnv(mode = Mode.NORMAL)
        state, _ = env.reset()
        state = state.flatten()
        mode_random = random.choice([True, False])
        player2 = h_env.BasicOpponent(weak = mode_random)
        obs_agent2 = env.obs_agent_two()

        while not done:
            action = agent.policy_old.act(state, memory)

            action_opp = player2.act(obs_agent2)

            next_state, reward, done, _, info = env.step(np.hstack([action, action_opp]))
            next_state = next_state.flatten()

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward
            state = next_state

            obs_agent2 = env.obs_agent_two()

            if done:
                break

        # Auxiliary phase: Train value function
        if episode % aux_phase_freq == 0:
            agent.auxiliary_phase(memory)
            print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if episode % update_timestep == 0:
            agent.update(memory)
            memory.clear_memory()

        # Log episode results
        episode_rewards.append(episode_reward)
        info_list.append(info.get("winner", None))

        if episode % checkpoint_freq == 0:
            agent.save_checkpoint(checkpoint_dir, episode)

    return episode_rewards, info_list, agent


# Train model for 1000 episodes
max_eps = 200001
max_tsteps = 500
upd_tsteps = 500
aux_phase = 50

rewards_ppg_kl, info_ppg_kl, ppg_KL = train_hockey_ppg(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppg_KL)


# Save final model
ppg_KL.save_checkpoint(checkpoint_dir, "final")

np.save( "rewards_ppg_kl.npy", rewards_ppg_kl)
np.stack("info_ppg_kl", info_ppg_kl)
