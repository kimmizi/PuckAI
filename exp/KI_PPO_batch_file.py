import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import hockey.hockey_env as h_env
from enum import Enum
from hockey.hockey_env import Mode
from KI_PPO import PPO, Memory
import os

### Define the function to parse the arguments
def parse_args():
    parser = argparse.ArgumentParser(description = "PPO Reinforcement Learning")

    parser.add_argument("--env", default = "Hockey-v0", help = "Gym environment name")
    parser.add_argument("--mode", choices = ["shooting", "defense", "normal"], default = "normal", help = "Training mode")
    parser.add_argument("--seed", type = int, default = 0, help = "Random seed")
    parser.add_argument("--max_episodes", type = int, default = 10, help = "Max training episodes (for testing)")
    parser.add_argument("--batch_size", type = int, default = 256, help = "Batch size")
    parser.add_argument("--n_latent_var", type = int, default = 64, help = "Number of hidden units in the neural network")
    parser.add_argument("--lr", type = float, default = 0.002, help = "Learning rate")
    parser.add_argument("--betas", type = tuple, default = (0.9, 0.999), help = "Adam optimizer betas")
    parser.add_argument("--gamma", type = float, default = 0.99, help = "Discount factor")
    parser.add_argument("--K_epochs", type = int, default = 10, help = "Number of policy update epochs per iteration")
    parser.add_argument("--eps_clip", type = float, default = 0.2, help = "Clipping parameter for PPO")
    parser.add_argument("--has_continuous_action_space", type = bool, default = True, help = "Whether action space is continuous")
    parser.add_argument("--action_std_init", type = float, default = 0.6, help = "Initial standard deviation for action distribution")
    parser.add_argument("--checkpoint_dir", default = "./checkpoints", help = "Checkpoint directory")
    parser.add_argument("--checkpoint_freq", type = int, default = 100000, help = "Checkpoint frequency")
    parser.add_argument("--resume", help = "Path to checkpoint to resume training")
    parser.add_argument("--evaluate", action = "store_true", help = "Evaluate the trained agent")

    return parser.parse_args()


### Define the function to train the agent
def train(args):
    if args.mode == "normal":
        mode = Mode.NORMAL
    elif args.mode == "shooting":
        mode = Mode.TRAIN_SHOOTING
    elif args.mode == "defense":
        mode = Mode.TRAIN_DEFENSE
    else:
        raise ValueError("Invalid mode! Choose from: shooting, defense, normal")

    # Create the Hockey environment
    env = h_env.HockeyEnv_BasicOpponent(mode = mode, weak_opponent = True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo = PPO(state_dim, action_dim, args)

    if args.resume:
        ppo.load_checkpoint(args.resume)

    memory = Memory()
    timestep = 0

    info_list = []

    episode_num = 0
    rewards = []

    while episode_num < args.max_episodes:

        state_info = env.reset(one_starting = True)
        if isinstance(state_info, tuple):
            state, info = state_info
        else:
            state = state_info
            info = {}

        episode_reward = 0
        done = False

        while not done:
            # Select action and pad with opponent action (zero vector for now)
            action = ppo.policy_old.act(state, memory)
            opponent_action = np.zeros(action_dim)  # Placeholder for weak opponent
            final_action = np.concatenate([action, opponent_action])

            next_state, reward, terminated, truncated, info = env.step(final_action)

            done = terminated or truncated

            # Store transition
            ppo.replay_buffer.add(state, action, reward, next_state, done)

            # Train agent
            if ppo.replay_buffer.size > args.batch_size:
                ppo.train()

            state = next_state
            episode_reward += reward

            # Save checkpoint periodically
            if episode_num % args.checkpoint_freq == 0:
                ppo.save_checkpoint(args.checkpoint_dir, episode_num)

        rewards.append(episode_reward)
        info_list.append(info["winner"])

        episode_num += 1
        if episode_num % 100 == 0:
            print(f"Episode {episode_num} | Reward: {episode_reward:.2f}")

    env.close()

    # **Ensure `rewards/` directory exists**
    rewards_dir = "rewards"
    os.makedirs(rewards_dir, exist_ok=True)

    # **Save rewards inside `rewards/`**
    rewards_file = f"{rewards_dir}/rewards_{args.mode}_{args.max_episodes}.npy"
    np.save(rewards_file, np.array(rewards))
    print(f"Rewards saved to {rewards_file}")

    # Save final trained model with mode-specific filename
    final_model_dir = os.path.join(args.checkpoint_dir, "final_models")
    os.makedirs(final_model_dir, exist_ok=True)
    model_save_path = os.path.join(final_model_dir, f"td3_model_{args.mode}.pth")

    ppo.save_final_model(model_save_path)
    print(f"Trained model saved as {model_save_path}")

    return rewards

if __name__ == "__main__":
    args = parse_args()
    rewards = train(args)

    if args.evaluate:
        print("Evaluating agent...")
        env = h_env.HockeyEnv_BasicOpponent(mode=h_env.HockeyEnv_BasicOpponent.NORMAL, weak_opponent=False)
        ppo.evaluate(env, num_episodes=10)




