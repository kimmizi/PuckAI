import hockey.hockey_env as h_env
from hockey.hockey_env import Mode
import numpy as np
import random

env_hockey = h_env.HockeyEnv_BasicOpponent(mode = Mode.NORMAL, weak_opponent = False)
state_dim = env_hockey.observation_space.shape[0]
action_dim = env_hockey.action_space.shape[0]


from PPO import PPO, Memory

ppo = PPO(
    state_dim,
    action_dim,
    n_latent_var = 128,
    has_continuous_action_space = True,
    action_std_init = 0.5,
    lr = 0.0003,
    betas = (0.9, 0.999),
    gamma = 0.99,
    K_epochs = 10,
    eps_clip = 0.25,
)

# Load model
# episodes = 5000
# ppo.load_checkpoint("checkpoints_ppg_kl/checkpoint_final_2.pth")
# print("Model loaded")


checkpoint_freq = 1000
checkpoint_dir = "./checkpoints_ppo"
update_timestep = 500


def train_hockey_ppg(env, max_episodes, max_timesteps, update_timestep, aux_phase_freq, agent):
    memory = Memory()
    episode_rewards = []
    info_list = []

    for episode in range(max_episodes):
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
            # agent.auxiliary_phase(memory)
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


# Train model
max_eps = 200001
max_tsteps = 500
upd_tsteps = 500
aux_phase = 50

rewards_ppo, info_ppo, ppo = train_hockey_ppg(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppo)

# Save final model, rewards and info
ppo.save_checkpoint(checkpoint_dir, "final")

np.save( "rewards_ppo.npy", rewards_ppo)
np.stack("info_ppo", info_ppo)
