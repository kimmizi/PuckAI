import hockey.hockey_env as h_env
from hockey.hockey_env import Mode
import gymnasium as gym
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PPG_improved import PPO_init, Memory


env = h_env.HockeyEnv_BasicOpponent(mode = Mode.NORMAL, weak_opponent = False)

# env = h_env.HockeyEnv(mode = Mode.NORMAL)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


ppo_2 = PPO_init(
    state_dim,
    action_dim,
    n_latent_var_actor = 128,
    n_latent_var_critic = 256,
    lr = 0.00003,
    betas = (0.9, 0.999),
    gamma = 0.95,
    K_epochs = 10,
    eps_clip = 0.25,
    has_continuous_action_space = True,
    action_std_init = 0.5,
    c1 = 0.5,
    c2 = 0.01,
    beta_clone = 0.95,
    network_depth_actor = 2,
    network_depth_critic = 4
)


checkpoint_freq = 10000
checkpoint_dir = "./checkpoints_ppg_init/"

update_timestep = 500

episodes = 100001
checkp = "checkpoint_final.pth"
ppo_2.load_checkpoint(checkpoint_dir + checkp)

checkpoint_dir = "./checkpoints_ppg_init"

def train_ppg(env, ppo_agent, num_episodes, aux_phase_freq):
    """
    Training loop with PPG.
    """
    memory = Memory()
    episode_rewards = []  # Track rewards for each episode
    info_list = []  # Track additional info (e.g., winner)

    for episode in range(episodes, num_episodes):
        state, _ = env.reset()
        state = state.flatten()
        done = False
        episode_reward = 0

        env = h_env.HockeyEnv(mode = Mode.NORMAL)
        o, info = env.reset()

        mode_random = random.choice([True, False])

        player2 = h_env.BasicOpponent(weak = mode_random)
        obs_agent2 = env.obs_agent_two()

        # Policy phase: Collect trajectories
        while not done:
            action = ppo_agent.policy_old.act(state, memory)

            action_opp = player2.act(obs_agent2)

            next_state, reward, done, _, info = env.step(np.hstack([action, action_opp]))
            next_state = next_state.flatten()

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward  # Accumulate reward
            state = next_state

            obs_agent2 = env.obs_agent_two()


        # Auxiliary phase: Train value function
        if episode % aux_phase_freq == 0:
            ppo_agent.auxiliary_phase(memory)

        if episode % update_timestep == 0:
            ppo_agent.update(memory)
            memory.clear_memory()

        # ppo_agent.update(memory)
        # memory.clear_memory()

        # Log episode results
        episode_rewards.append(episode_reward)
        info_list.append(info.get("winner", None))
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if episode % checkpoint_freq == 0:
            ppo_agent.save_checkpoint(checkpoint_dir, episode)

    return episode_rewards, info_list, ppo_agent

episode_rewards_ppg2, info_list_ppg2, ppo_2 = train_ppg(env, ppo_2, num_episodes = 200000, aux_phase_freq = 50)


# Save final model
ppo_2.save_checkpoint(checkpoint_dir, "final")
