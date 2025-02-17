import optuna
import numpy as np
import torch
import hockey.hockey_env as h_env
from hockey.hockey_env import Mode
from KI_PPO_ppg2 import PPO_2, Memory

checkpoint_freq = 500
checkpoint_dir = "./checkpoints_ppg"

max_episodes = 2001
max_timesteps = 500
update_timestep = 5001
checkpoint_freq = 500

memory = Memory()
timestep = 0

info_list_ppg = []
episode_rewards_ppg = []

env = h_env.HockeyEnv_BasicOpponent(mode = Mode.NORMAL, weak_opponent = False)
# env = h_env.HockeyEnv(mode = Mode.NORMAL)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


def train_ppo(lr, gamma, eps_clip, K_epochs, n_latent_var, action_std, aux_phase ):
    # Initialize PPO agent with given hyperparameters
    ppo_agent = PPO_2(state_dim, action_dim, n_latent_var=n_latent_var, lr=lr, gamma=gamma, eps_clip=eps_clip, K_epochs=K_epochs, has_continuous_action_space=True, action_std_init=action_std, betas=(0.9, 0.999))

    memory = Memory()

    # Train the agent
    episode_rewards = []
    for episode in range(100):
        print("Episode: ", episode)

        state, _ = env.reset()
        state = state.flatten()
        done = False
        episode_reward = 0

        while not done:
            action = ppo_agent.policy_old.act(state, memory)

            opponent_action = np.zeros(action_dim)
            final_action = np.concatenate([action, opponent_action])

            next_state, reward, done, t, info = env.step(final_action)
            next_state = state.flatten()

            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            episode_reward += reward
            state = next_state

        ppo_agent.update(memory)
        episode_rewards.append(episode_reward)

    # Return the average reward
    return torch.mean(torch.tensor(episode_rewards)).item()

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    eps_clip = trial.suggest_float("eps_clip", 0.1, 0.3)
    K_epochs = trial.suggest_int("K_epochs", 4, 20)
    n_latent_var = trial.suggest_int("n_latent_var", 32, 256, log=True)
    action_std = trial.suggest_float("action_std", 0.1, 1.0)
    aux_phase = trial.suggest_int("aux_phase", 5, 100)
    return train_ppo(lr = lr, gamma = gamma, eps_clip = eps_clip, K_epochs = K_epochs, n_latent_var = n_latent_var, action_std = action_std, aux_phase = aux_phase)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)

# Save the best hyperparameters
best_params = study.best_params
np.save("best_params.npy", best_params)
