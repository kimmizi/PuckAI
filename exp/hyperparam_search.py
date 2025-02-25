import optuna
import numpy as np
import torch
import random
import hockey.hockey_env as h_env
from hockey.hockey_env import Mode
from PPG_KL import PPO, Memory

env = h_env.HockeyEnv_BasicOpponent(mode = Mode.NORMAL, weak_opponent = False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


def train_ppo(lr, gamma, eps_clip, K_epochs, n_latent_var_actor, n_latent_var_critic, action_std, aux_phase, update_timestep, betas, c1, c2, beta_clone, network_depth_actor, network_depth_critic):

    # Initialize PPO agent with given hyperparameters
    ppg_agent = PPO(
        state_dim,
        action_dim,
        n_latent_var_actor = n_latent_var_actor,
        n_latent_var_critic = n_latent_var_critic,
        network_depth_actor = network_depth_actor,
        network_depth_critic = network_depth_critic,
        has_continuous_action_space = True,
        action_std_init = action_std,
        lr = lr,
        betas = betas,
        gamma = gamma,
        K_epochs = K_epochs,
        eps_clip = eps_clip,
        c1 = c1,
        c2 = c2,
        beta_clone = beta_clone
    )

    memory = Memory()

    # Train the agent
    episode_rewards = []
    for episode in range(500):
        print("Episode: ", episode)

        env = h_env.HockeyEnv(mode = Mode.NORMAL)
        o, info = env.reset()

        state, _ = env.reset()
        state = state.flatten()
        done = False
        episode_reward = 0


        mode_random = random.choice([True, False])

        player2 = h_env.BasicOpponent(weak = mode_random)
        obs_agent2 = env.obs_agent_two()


        while not done:
            action = ppg_agent.policy_old.act(state, memory)

            action_opp = player2.act(obs_agent2)

            next_state, reward, done, _, info = env.step(np.hstack([action, action_opp]))
            next_state = next_state.flatten()

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward  # Accumulate reward
            state = next_state

        # Auxiliary phase: Train value function
        if episode % aux_phase == 0:
            ppg_agent.auxiliary_phase(memory)

        if episode % update_timestep == 0:
            ppg_agent.update(memory)
            memory.clear_memory()

        # ppg_agent.update(memory)
        episode_rewards.append(episode_reward)

    # Return the average reward
    return torch.mean(torch.tensor(episode_rewards)).item()

def objective(trial):
    lr = trial.suggest_float("lr", 1e-8, 1e-2, log = True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    eps_clip = trial.suggest_float("eps_clip", 0.1, 0.3)
    K_epochs = trial.suggest_int("K_epochs", 4, 50)
    n_latent_var_actor = trial.suggest_int("n_latent_var", 32, 512, log = True)
    n_latent_var_critic = trial.suggest_int("n_latent_var", 32, 512, log = True)
    action_std = trial.suggest_float("action_std", 0.1, 1.0)
    aux_phase = trial.suggest_int("aux_phase", 20, 100)
    update_timestep = trial.suggest_int("update_timestep", 100, 1000)
    beta1 = trial.suggest_float("beta1", 0.9, 0.999)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    betas = (beta1, beta2)
    c1 = trial.suggest_float("c1", 0.1, 1.0)
    c2 = trial.suggest_float("c2", 0.1, 1.0)
    beta_clone = trial.suggest_float("beta_clone", 0.1, 1.0)
    network_depth_actor = trial.suggest_int("network_depth_actor", 1, 5)
    network_depth_critic = trial.suggest_int("network_depth_critic", 1, 5)
    return train_ppo(lr = lr, gamma = gamma, eps_clip = eps_clip, K_epochs = K_epochs,
                     n_latent_var_actor = n_latent_var_actor, n_latent_var_critic = n_latent_var_critic,
                     action_std = action_std, aux_phase = aux_phase, update_timestep = update_timestep,
                     betas = betas, c1 = c1, c2 = c2, beta_clone = beta_clone,
                     network_depth_actor = network_depth_actor, network_depth_critic = network_depth_critic)

study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials = 25)

print("Best hyperparameters:", study.best_params)
print("Best value:", study.best_value)

# Save the best hyperparameters
best_params = study.best_params
best_value = study.best_value
np.save("best_params.npy", best_params)
np.save("best_value.npy", best_value)
