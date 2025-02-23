from __future__ import annotations
import argparse
import uuid

import torch
import numpy as np
import hockey.hockey_env as h_env
from gymnasium import spaces
from comprl.client import Agent, launch_client

from agent import DuelingAgent
from config_test import config as config_test


class MyAgent(Agent):
    def __init__(self, path_weights=None):
        super().__init__()
        
        self.env_stub = h_env.HockeyEnv()
        discrete_actions = 8
        ac_map = {
            tuple(self.env_stub.discrete_to_continous_action(i)): i 
            for i in range(discrete_actions)
        }
        ac_space = spaces.Discrete(len(ac_map))

        self.dueling_agent = DuelingAgent(
            observation_space=self.env_stub.observation_space,
            action_space=ac_space,
            config=config_test
        )

        if path_weights:
            loaded_sd = torch.load(path_weights, map_location="cpu")
            self.dueling_agent.Q.load_state_dict(loaded_sd)
            print(f"Loaded model weights from {path_weights}")

        self.env_for_conversion = h_env.HockeyEnv()

    def get_step(self, observation: list[float]) -> list[float]:
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        discrete_a = self.dueling_agent.act(obs_tensor, eps=0.0)
        continuous_a = self.env_for_conversion.discrete_to_continous_action(discrete_a)
        return continuous_a

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder="big"))
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


def initialize_agent(agent_args: list[str]) -> Agent:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "my_agent"],
        default="weak",
        help="Which agent to use.",
    )
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args(agent_args)
    
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "my_agent":
        agent = MyAgent(path_weights=args.weights)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")
    
    return agent

def main() -> None:
    launch_client(initialize_agent)

if __name__ == "__main__":
    main()
