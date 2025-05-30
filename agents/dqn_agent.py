import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque # Replay buffer to store transitions
from models.q_network import QNetwork  


class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Dictionary with hyperparameters
        self.gamma = config["gamma"]                     # Discout factor at bellman equation
        self.epsilon = config["epsilon_start"]           # Initial exploaration rate
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]     # Well trained model doesn't need to explore
        self.batch_size = config["batch_size"]           # For use replay buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-network
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config["lr"]) 

        # Replay Buffer
        self.replay_buffer = deque(maxlen=config["replay_buffer_size"]) #Save 10000 succesive transitions

        # Update timing
        self.update_freq = config["update_freq"]
        self.learn_step = 0

    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randrange(self.action_dim)  # Explore, return random action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, state_dim)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()  # Exploit, return action with max Q-value

    def store_transition(self, state, action, reward, next_state, done):
        
        self.replay_buffer.append((state, action, reward, next_state, done)) # Store at replay buffer

    def update(self):
        
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        # Sample a batch of transitions
        batch = random.sample(self.replay_buffer, self.batch_size) # Pick random batch size transitions from replay buffer
        states, actions, rewards, next_states, dones = zip(*batch) # Unzip the batch

        states = torch.FloatTensor(states).to(self.device)                # (batch_size, state_dim)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # (batch_size, 1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q(s, a)
        q_values = self.q_network(states).gather(1, actions)

        # Compute max Q(s', a') from the target network
        with torch.no_grad(): 
            # From bellman equation, we can get this formula
            # Q(s, a) = r + gamma * max_a' Q(s', a')
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1) 
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones) # We can see discounted cumulative reward 
        # Compute loss
        criterion = nn.MSELoss() # Mean Squared Error Loss
        loss = criterion(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Soft update target network for stability
        self.learn_step += 1 
        if self.learn_step % self.update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.q_network.eval()
