from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np


from comprl.client import Agent, launch_client

from PPG_KL_Beta import PPO, Memory


class PPO_kimi(Agent):
    def __init__(self, load_model = None):
        super().__init__()

        self.env = h_env.HockeyEnv(mode = h_env.Mode.NORMAL)
        state_dim = self.env.observation_space.shape[0]
        # action_dim = self.env.action_space.shape[0]

        self.ppo = PPO(
            state_dim,
            4,
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

        if load_model:
            self.ppo.load_checkpoint(load_model)
            print(f"Model was loaded: {load_model}")

    def get_step(self, obv: list[float]) -> list[float]:

        memory = None

        state, _ = self.env.reset()
        state = state.flatten()

        action = self.ppo.policy_old.act(state, memory)
        action = action.tolist()

        return action

    def on_start_game(self, game_id: int) -> None:
        game_id = uuid.UUID(int = int.from_bytes(game_id, byteorder = "big"))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "my_agent"],
        default="weak",
        help="Which agent to use.",
    )
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "my_agent":
        agent = PPO_kimi(load_model = args.model)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
