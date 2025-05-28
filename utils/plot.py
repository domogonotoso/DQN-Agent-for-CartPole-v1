# utils/plot.py

import matplotlib.pyplot as plt

def plot_rewards(rewards, path="results/rewards_plot.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
