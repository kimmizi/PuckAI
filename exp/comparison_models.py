import numpy as np
# import random
import gymnasium as gym
import hockey.hockey_env as h_env
from hockey.hockey_env import Mode


env = "Hockey"

if env == "Hockey":
    # Hockey Env
    env_hockey = h_env.HockeyEnv_BasicOpponent(mode = Mode.NORMAL, weak_opponent = True)
    state_dim = env_hockey.observation_space.shape[0]
    action_dim = 4

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


    from PPG import PPO, Memory
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

    from PPG_KL import PPO
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

    from PPG_Beta import PPO
    ppg_beta = PPO(
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

    from PPG_KL_Beta import PPO
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

elif env == "Pendulum":
    # Gymnasium: Pendulum
    env_pendulum = gym.make("Pendulum-v1")
    state_dim = env_pendulum.observation_space.shape[0]
    action_dim = env_pendulum.action_space.shape[0]
elif env == "HalfCheetah":
    # Gymnasium: HalfCheetah
    env_halfcheetah = gym.make("HalfCheetah-v5")
    state_dim = env_halfcheetah.observation_space.shape[0]
    action_dim = env_halfcheetah.action_space.shape[0]

if env == "Pendulum" or env == "HalfCheetah":
    from PPO import PPO, Memory
    ppo_vanilla = PPO(
        state_dim,
        action_dim,
        n_latent_var = 128,
        lr = 0.003,
        betas = (0.9, 0.999),
        gamma = 0.99,
        K_epochs = 10,
        eps_clip = 0.25,
        has_continuous_action_space = True,
        action_std_init = 0.5
    )

    from PPG import PPO, Memory
    ppg = PPO(
        state_dim,
        action_dim,
        n_latent_var_actor = 128,
        n_latent_var_critic = 128,
        network_depth_actor = 1,
        network_depth_critic = 1,
        has_continuous_action_space = True,
        action_std_init = 0.5,
        lr = 0.003,
        betas = (0.9, 0.999),
        gamma = 0.99,
        K_epochs = 10,
        eps_clip = 0.25,
        c1 = 0.5,
        c2 = 0.01,
        beta_clone = 0.95
    )

    from PPG_KL import PPO
    ppg_KL = PPO(
        state_dim,
        action_dim,
        n_latent_var_actor = 128,
        n_latent_var_critic = 128,
        network_depth_actor = 1,
        network_depth_critic = 1,
        has_continuous_action_space = True,
        action_std_init = 0.5,
        lr = 0.003,
        betas = (0.9, 0.999),
        gamma = 0.99,
        K_epochs = 10,
        eps_clip = 0.25,
        c1 = 0.5,
        c2 = 0.01,
        beta_clone = 0.95
    )

    from PPG_Beta import PPO
    ppg_beta = PPO(
        state_dim,
        action_dim,
        n_latent_var_actor = 128,
        n_latent_var_critic = 128,
        network_depth_actor = 1,
        network_depth_critic = 1,
        has_continuous_action_space = True,
        action_std_init = 0.5,
        lr = 0.003,
        betas = (0.9, 0.999),
        gamma = 0.99,
        K_epochs = 10,
        eps_clip = 0.25,
        c1 = 0.5,
        c2 = 0.01,
        beta_clone = 0.95
    )

    from PPG_KL_Beta import PPO
    ppg_KL_beta = PPO(
        state_dim,
        action_dim,
        n_latent_var_actor = 128,
        n_latent_var_critic = 128,
        network_depth_actor = 1,
        network_depth_critic = 1,
        has_continuous_action_space = True,
        action_std_init = 0.5,
        lr = 0.003,
        betas = (0.9, 0.999),
        gamma = 0.99,
        K_epochs = 10,
        eps_clip = 0.25,
        c1 = 0.5,
        c2 = 0.01,
        beta_clone = 0.95
    )

def train_hockey_ppo_weak(env, max_episodes, max_timesteps, update_timestep, aux_phase_freq, agent):
    memory = Memory()
    episode_rewards = []
    info_list = []

    for episode in range(max_episodes):
        # state, _ = env.reset()
        # state = state.flatten()
        done = False
        episode_reward = 0

        # # Randomly choose weak or strong opponent
        # env = h_env.HockeyEnv(mode = Mode.NORMAL)
        # state, _ = env.reset()
        # state = state.flatten()
        # mode_random = random.choice([True, False])
        # player2 = h_env.BasicOpponent(weak = mode_random)
        # obs_agent2 = env.obs_agent_two()

        env = h_env.HockeyEnv(mode = Mode.NORMAL)
        state, _ = env.reset()
        state = state.flatten()
        player2 = h_env.BasicOpponent(weak = True)
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



def train_hockey_ppg_weak(env, max_episodes, max_timesteps, update_timestep, aux_phase_freq, agent):
    memory = Memory()
    episode_rewards = []
    info_list = []

    for episode in range(max_episodes):
        # state, _ = env.reset()
        # state = state.flatten()
        done = False
        episode_reward = 0

        # # Randomly choose weak or strong opponent
        # env = h_env.HockeyEnv(mode = Mode.NORMAL)
        # state, _ = env.reset()
        # state = state.flatten()
        # mode_random = random.choice([True, False])
        # player2 = h_env.BasicOpponent(weak = mode_random)
        # obs_agent2 = env.obs_agent_two()

        env = h_env.HockeyEnv(mode = Mode.NORMAL)
        state, _ = env.reset()
        state = state.flatten()
        player2 = h_env.BasicOpponent(weak = True)
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

        # Update PPO
        if episode % update_timestep == 0:
            agent.update(memory)
            memory.clear_memory()

        # Log episode results
        episode_rewards.append(episode_reward)
        info_list.append(info.get("winner", None))

        # if episode % checkpoint_freq == 0:
        #     ppo_agent.save_checkpoint(checkpoint_dir, episode)

    return episode_rewards, info_list, agent

# Training loop
def train_pendulum_ppo(env, max_episodes, max_timesteps, update_timestep, agent):
    memory = Memory()
    timestep = 0
    episode_rewards = []

    print("Agent: ", agent)

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = state.flatten()
        episode_reward = 0

        for t in range(max_timesteps):
            timestep += 1

            # Sample action
            action = agent.policy_old.act(state, memory)
            state, reward, done, _, _ = env.step(action)
            state = state.flatten()

            # Save reward and terminal state
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward

            if done:
                break

        # Update PPO if it's time
        if timestep % update_timestep == 0:
            agent.update(memory)
            memory.clear_memory()
            timestep = 0

        if episode % 50 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward}")

        episode_rewards.append(episode_reward)

    return episode_rewards

# Training loop
def train_pendulum_ppg(env, max_episodes, max_timesteps, update_timestep, agent):
    memory = Memory()
    episode_rewards = []
    timestep = 0

    print("Agent: ", agent)

    for episode in range(max_episodes):
        state, _ = env.reset()
        state = state.flatten()
        episode_reward = 0
        done = False

        for t in range(max_timesteps):
            timestep += 1
            # Sample action
            action = agent.policy_old.act(state, memory)
            state, reward, done, _, _ = env.step(action)

            # Save reward and terminal state
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward

            if done:
                break

        # Auxiliary phase: Train value function
        if episode % 50 == 0:
            agent.auxiliary_phase(memory)
            print(f"Episode {episode + 1}, Reward: {episode_reward}")

        # Update PPO if it's time
        if episode % update_timestep == 0:
            agent.update(memory)
            memory.clear_memory()

        episode_rewards.append(episode_reward)

    return episode_rewards


# Train models
max_eps = 5000
max_tsteps = 500
upd_tsteps = 250
aux_phase = 50

if env == "Hockey":
    rewards_ppo, info_ppo, ppo_vanilla = train_hockey_ppo_weak(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppo_vanilla)
    np.save( "rewards_ppo.npy", rewards_ppo)
    np.save( "info_ppo.npy", info_ppo)

    rewards_ppg, info_ppg, ppg = train_hockey_ppg_weak(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppg)
    np.save( "rewards_ppg.npy", rewards_ppg)
    np.save( "info_ppg.npy", info_ppg)

    rewards_ppg_kl, info_ppg_kl, ppg_KL = train_hockey_ppg_weak(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppg_KL)
    np.save( "rewards_ppg_kl.npy", rewards_ppg_kl)
    np.save( "info_ppg_kl.npy", info_ppg_kl)

    rewards_ppg_beta, info_ppg_beta, ppg_beta = train_hockey_ppg_weak(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppg_beta)
    np.save( "rewards_ppg_beta.npy", rewards_ppg_beta)
    np.save( "info_ppg_beta.npy", info_ppg_beta)

    rewards_ppg_kl_beta, info_ppg_kl_beta, ppg_kl_beta = train_hockey_ppg_weak(env_hockey, max_eps, max_tsteps, upd_tsteps, aux_phase, ppg_KL_beta)
    np.save( "rewards_ppg_kl_beta.npy", rewards_ppg_kl_beta)
    np.save( "info_ppg_kl_beta.npy", info_ppg_kl_beta)

elif env == "Pendulum":
    print("Agent: ", ppg)
    rewards_ppg = train_pendulum_ppg(env_pendulum, max_eps, max_tsteps, upd_tsteps, ppg)
    np.save( "../dat/pendulum/rewards_ppg_pendulum.npy", rewards_ppg)
    print("rewards_ppg_pendulum.npy saved")

    print("Agent: ", ppg_KL)
    rewards_ppg_kl = train_pendulum_ppg(env_pendulum, max_eps, max_tsteps, upd_tsteps, ppg_KL)
    np.save( "../dat/pendulum/rewards_ppg_kl_pendulum.npy", rewards_ppg_kl)
    print("rewards_ppg_kl_pendulum.npy saved")

    print("Agent: ", ppg_beta)
    rewards_ppg_beta = train_pendulum_ppg(env_pendulum, max_eps, max_tsteps, upd_tsteps, ppg_beta)
    np.save( "../dat/pendulum/rewards_ppg_beta_pendulum.npy", rewards_ppg_beta)
    print("rewards_ppg_beta_pendulum.npy saved")

    print("Agent: ", ppg_KL_beta)
    rewards_ppg_kl_beta = train_pendulum_ppg(env_pendulum, max_eps, max_tsteps, upd_tsteps, ppg_KL_beta)
    np.save( "../dat/pendulum/rewards_ppg_kl_beta_pendulum.npy", rewards_ppg_kl_beta)
    print("rewards_ppg_kl_beta_pendulum.npy saved")

    print("Agent: ", ppo_vanilla)
    rewards_ppo = train_pendulum_ppo(env_pendulum, max_eps, max_tsteps, upd_tsteps, ppo_vanilla)
    np.save( "../dat/pendulum/rewards_ppo_pendulum.npy", rewards_ppo)

elif env == "HalfCheetah":
    rewards_ppg = train_pendulum_ppg(env_halfcheetah, max_eps, max_tsteps, upd_tsteps, ppg)
    np.save("../dat/cheetah/rewards_ppg_halfcheetah.npy", rewards_ppg)

    rewards_ppg_kl = train_pendulum_ppg(env_halfcheetah, max_eps, max_tsteps, upd_tsteps, ppg_KL)
    np.save("../dat/cheetah/rewards_ppg_kl_halfcheetah.npy", rewards_ppg_kl)

    rewards_ppg_beta = train_pendulum_ppg(env_halfcheetah, max_eps, max_tsteps, upd_tsteps, ppg_beta)
    np.save("../dat/cheetah/rewards_ppg_beta_halfcheetah.npy", rewards_ppg_beta)

    rewards_ppg_kl_beta = train_pendulum_ppg(env_halfcheetah, max_eps, max_tsteps, upd_tsteps, ppg_KL_beta)
    np.save("../dat/cheetah/rewards_ppg_kl_beta_halfcheetah.npy", rewards_ppg_kl_beta)

    rewards_ppo = train_pendulum_ppo(env_halfcheetah, max_eps, max_tsteps, upd_tsteps, ppo_vanilla)
    np.save( "../dat/rewards_ppo_halfcheetah.npy", rewards_ppo)
