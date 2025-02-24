# PuckAI

This repo contains the code used for the final project report of **Reinforcement Learning lecture 2024/25** by Prof. Martius at University of Tübingen.

The goal is to develop a reinforcement learning (RL) agent capable of solving simple control tasks and competing in a simulated hockey game.

The core model is based on Proximal Policy Optimization (PPO) and extended with techniques such as Phasic Policy Gradient (PPG), KL divergence regularization, and Beta policy parameterization to address common PPO failure modes.

---

## Structure

You will find the models I build in the directory `.\src`, the experiments we conducted in `.\exp`, the data (like rewards, information, checkpoints run at cluster) in `.\dat`, and the report and figures in `.\doc`.

---

## Models

I compared five PPO models that get increasingly optimized:
