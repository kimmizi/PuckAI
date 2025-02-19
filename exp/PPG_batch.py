import numpy as np
import torch
import gymnasium as gym
import hockey.hockey_env as h_env
from hockey.hockey_env import Mode
from KI_PPO_ppg2 import PPO_2, Memory


env = h_env.HockeyEnv_BasicOpponent(mode = Mode.TRAIN_SHOOTING, weak_opponent = False)
# env = h_env.HockeyEnv(mode = Mode.NORMAL)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

checkpoint_freq = 500
checkpoint_dir = "./checkpoints_ppg_shoot"

update_timestep = 149

ppo_2 = PPO_2(
    state_dim,
    action_dim,
    n_latent_var = 43,
    lr = 8.988645063280326e-05,
    betas = (0.9, 0.999),
    gamma = 0.9967844742150109,
    K_epochs = 24,
    eps_clip = 0.2788302938997803,
    has_continuous_action_space = True,
    action_std_init = 0.3462835582997581
)

def train_ppg(env, ppo_agent, num_episodes, aux_phase_freq):
    """
    Training loop with PPG.
    """
    memory = Memory()
    episode_rewards = []  # Track rewards for each episode
    info_list = []  # Track additional info (e.g., winner)

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state.flatten()  # Ensure the state is 1D
        done = False
        episode_reward = 0  # Initialize episode reward

        # Policy phase: Collect trajectories
        while not done:
            action = ppo_agent.policy_old.act(state, memory)

            next_state, reward, done, _, info = env.step(action)
            next_state = next_state.flatten()

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward  # Accumulate reward
            state = next_state

        # while not done:
        #     action = ppo_agent.policy_old.act(state, memory)
        #
        #     opponent_action = np.zeros(action_dim)
        #     final_action = np.concatenate([action, opponent_action])
        #
        #     next_state, reward, done, t, info = env.step(final_action)
        #     next_state = state.flatten()
        #
        #     memory.rewards.append(reward)
        #     memory.is_terminals.append(done)
        #     episode_reward += reward
        #     state = next_state

        # Auxiliary phase: Train value function
        if episode % aux_phase_freq == 0:
            ppo_agent.auxiliary_phase(memory)

        if episode % update_timestep == 0:
            ppo_agent.update(memory)
            memory.clear_memory()
            timestep = 0

        # Log episode results
        episode_rewards.append(episode_reward)
        info_list.append(info.get("winner", None))
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if episode % checkpoint_freq == 0:
            ppo_agent.save_checkpoint(checkpoint_dir, episode)
            # Save results
            np.save("episode_rewards_ppg2.npy", episode_rewards)
            np.save("info_list_ppg2.npy", info_list)

    return episode_rewards, info_list, ppo_agent

episode_rewards_ppg2, info_list_ppg2, ppo_2 = train_ppg(env, ppo_2, num_episodes = 100001, aux_phase_freq = 72)

# Save final model
ppo_2.save_checkpoint(checkpoint_dir, "final")
