# test.py

import gym
import torch
import yaml
import numpy as np
from agents.dqn_agent import DQNAgent


def main():
    # Load config
    with open("config/dqn_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Set environment (no render)
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Agent
    agent = DQNAgent(state_dim, action_dim, config)
    agent.load("results/saved_model.pth")

    # Evaluation
    num_test_episodes = 100
    rewards = []

    for episode in range(1, num_test_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = agent.select_action(state, epsilon=0.0)  # greedy policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            step += 1
        print(f"Episode {episode}, Reward: {total_reward}, Steps: {step}")
        rewards.append(total_reward)


    env.close()
    print(f"\nAverage Reward over {num_test_episodes} episodes: {np.mean(rewards):.2f}")
    
    

if __name__ == "__main__":
    main()