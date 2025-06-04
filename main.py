import argparse
from train import main as train_main
from test import main as test_main
import gym
import torch
import yaml
import os
import time
from agents.dqn_agent import DQNAgent
from gym.wrappers import RecordVideo

def render_agent():
    # Load config
    with open("config/dqn_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Create folder to save videos
    os.makedirs("videos/", exist_ok=True)

    # Set environment 
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # Video recording
    env = RecordVideo(env, video_folder="videos/", episode_trigger=lambda x: True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load trained agent
    agent = DQNAgent(state_dim, action_dim, config)
    agent.load("results/saved_model.pth")

    # Run one episode
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action = agent.select_action(state, epsilon=0.0)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        step += 1

    print(f"Video saved: reward = {total_reward}, steps = {step}")
    time.sleep(1)  # Let video recorder finalize writing
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'render'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train_main()
    elif args.mode == 'test':
        test_main()
    elif args.mode == 'render':
        render_agent()
