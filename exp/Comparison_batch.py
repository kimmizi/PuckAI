import hockey.hockey_env as h_env
from hockey.hockey_env import Mode
import numpy as np
import random


env_hockey = h_env.HockeyEnv_BasicOpponent(mode = Mode.NORMAL, weak_opponent = False)
state_dim = env_hockey.observation_space.shape[0]
action_dim = env_hockey.action_space.shape[0]



from PPO import PPO, Memory

ppo_vanilla = PPO(
    state_dim,
    action_dim,
    n_latent_var = 128,
    lr = 0.0003,
    betas = (0.9, 0.999),
    gamma = 0.99,
    K_epochs = 10,
    eps_clip = 0.25,
    has_continuous_action_space = True,
    action_std_init = 0.5
)


from PPG import PPO
ppg = PPO(
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


from PPG_GAE import PPO, Memory

ppg_GAE = PPO(
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

from PPG_GAE_KL import PPO

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

from PPG_GAE_Beta import PPO

ppg_GAE_beta = PPO(
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

from PPG_GAE_KL_Beta import PPO

ppg_KL_beta = PPO(
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


def train_hockey_ppo(env, max_episodes, max_timesteps, update_timestep, aux_phase_freq, agent):
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

        # if episode % checkpoint_freq == 0:
        #     ppo_agent.save_checkpoint(checkpoint_dir, episode)

    return episode_rewards, info_list, agent



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
            agent.auxiliary_phase(memory)
            print(f"Episode {episode + 1}, Reward: {episode_reward}")

        if episode % update_timestep == 0:
            agent.update(memory)
            memory.clear_memory()

        # Log episode results
        episode_rewards.append(episode_reward)
        info_list.append(info.get("winner", None))

        # if episode % checkpoint_freq == 0:
        #     ppo_agent.save_checkpoint(checkpoint_dir, episode)

    return episode_rewards, info_list, agent


# Train model for 1000 episodes
max_eps = 5000
max_tsteps = 500
upd_tsteps = 500
aux_phase = 50

rewards_ppo, info_ppo, ppo_vanilla = train_hockey_ppo(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppo_vanilla)
rewards_ppg, info_ppg, ppg = train_hockey_ppg(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppg)
rewards_ppg_gae, info_ppg_gae, ppg_GAE = train_hockey_ppg(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppg_GAE)
rewards_ppg_kl, info_ppg_kl, ppg_KL = train_hockey_ppg(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppg_KL)
rewards_ppg_beta, info_ppg_beta, ppg_beta = train_hockey_ppg(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppg_GAE_beta)
rewards_ppg_kl_beta, info_ppg_kl_beta, ppg_kl_beta = train_hockey_ppg(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppg_KL_beta)

np.save( "rewards_ppo.npy", rewards_ppo)
np.save( "rewards_ppg.npy", rewards_ppg)
np.save( "rewards_ppg_gae.npy", rewards_ppg_gae)
np.save( "rewards_ppg_kl.npy", rewards_ppg_kl)
np.save( "rewards_ppg_beta.npy", rewards_ppg_beta)
np.save( "rewards_ppg_kl_beta.npy", rewards_ppg_kl_beta)

np.save( "info_ppo.npy", info_ppo)
np.save( "info_ppg.npy", info_ppg)
np.save( "info_ppg_gae.npy", info_ppg_gae)
np.save( "info_ppg_kl.npy", info_ppg_kl)
np.save( "info_ppg_beta.npy", info_ppg_beta)
np.save( "info_ppg_kl_beta.npy", info_ppg_kl_beta)
