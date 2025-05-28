# train.py

import gym
import torch
import yaml
import numpy as np
from agents.dqn_agent import DQNAgent
from utils.plot import plot_rewards
import os

# Load config
with open("config/dqn_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Set environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Set seed for reproducibility
seed = config.get("seed", 42)  # Default seed
np.random.seed(seed)
torch.manual_seed(seed)
state, _ = env.reset(seed=seed)

# Agent
agent = DQNAgent(state_dim, action_dim, config)

# Training loop
num_episodes = config["num_episodes"]
target_reward = config.get("target_reward", 475) # If we reach to traget_reward before num_episodes, we wil save the model.
save_path = "results/saved_model.pth"

reward_history = []

for episode in range(1, num_episodes + 1):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action) # Take action in the environment
        agent.store_transition(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        episode_reward += reward

    reward_history.append(episode_reward)

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")

    # Save model if solved
    if np.mean(reward_history[-100:]) >= target_reward:
        print(f"Solved in episode {episode}, saving model...")
        agent.save(save_path)
        break

# Final save and plot
agent.save(save_path)
plot_rewards(reward_history, path="results/rewards_plot.png")
