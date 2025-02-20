import optuna
import numpy as np
import torch
import hockey.hockey_env as h_env
from hockey.hockey_env import Mode
# from KI_PPO_ppg2 import PPO_2, Memory
# from KI_PPO_PPG_3fails import PPO_optim, Memory
from KI_PPO_ppg2_initialized import PPO_init, Memory

memory = Memory()
timestep = 0

env = h_env.HockeyEnv_BasicOpponent(mode = Mode.NORMAL, weak_opponent = False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


def train_ppo(lr, gamma, eps_clip, K_epochs, n_latent_var, action_std, aux_phase, update_timestep, betas, c1, c2, beta_clone):

    # Initialize PPO agent with given hyperparameters
    ppo_agent = PPO_init(state_dim, action_dim, n_latent_var = n_latent_var, lr = lr, gamma = gamma,
                      eps_clip = eps_clip, K_epochs = K_epochs, has_continuous_action_space = True,
                      action_std_init = action_std, betas = betas, c1 = c1, c2 = c2, beta_clone = beta_clone)

    memory = Memory()

    # Train the agent
    episode_rewards = []
    for episode in range(500):
        print("Episode: ", episode)

        state, _ = env.reset()
        state = state.flatten()
        done = False
        episode_reward = 0

        while not done:
            action = ppo_agent.policy_old.act(state, memory)

            next_state, reward, done, _, info = env.step(action)
            next_state = next_state.flatten()

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            episode_reward += reward  # Accumulate reward
            state = next_state

        # Auxiliary phase: Train value function
        if episode % aux_phase == 0:
            ppo_agent.auxiliary_phase(memory)

        if episode % update_timestep == 0:
            ppo_agent.update(memory)
            memory.clear_memory()
            timestep = 0

        # ppo_agent.update(memory)
        episode_rewards.append(episode_reward)

    # Return the average reward
    return torch.mean(torch.tensor(episode_rewards)).item()

def objective(trial):
    lr = trial.suggest_float("lr", 1e-8, 1e-2, log = True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    eps_clip = trial.suggest_float("eps_clip", 0.1, 0.3)
    K_epochs = trial.suggest_int("K_epochs", 4, 50)
    n_latent_var = trial.suggest_int("n_latent_var", 32, 512, log = True)
    action_std = trial.suggest_float("action_std", 0.1, 1.0)
    aux_phase = trial.suggest_int("aux_phase", 20, 100)
    update_timestep = trial.suggest_int("update_timestep", 100, 1000)
    beta1 = trial.suggest_float("beta1", 0.9, 0.999)
    beta2 = trial.suggest_float("beta2", 0.9, 0.999)
    betas = (beta1, beta2)
    c1 = trial.suggest_float("c1", 0.1, 1.0)
    c2 = trial.suggest_float("c2", 0.1, 1.0)
    beta_clone = trial.suggest_float("beta_clone", 0.1, 1.0)
    return train_ppo(lr = lr, gamma = gamma, eps_clip = eps_clip, K_epochs = K_epochs,
                     n_latent_var = n_latent_var, action_std = action_std,
                     aux_phase = aux_phase, update_timestep = update_timestep, betas = betas,
                     c1 = c1, c2 = c2, beta_clone = beta_clone)

study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials = 50)

print("Best hyperparameters:", study.best_params)
print("Best value:", study.best_value)

# Save the best hyperparameters
best_params = study.best_params
best_value = study.best_value
np.save("best_params.npy", best_params)
np.save("best_value.npy", best_value)
