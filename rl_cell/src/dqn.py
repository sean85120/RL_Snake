import os
import random
from collections import deque
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gdsfactory.component import Component

from rl_cell.src.layer_mapping import LayerMapping
from rl_cell.src.standard_cell_game import Action, StandardCellGame


# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            target = reward
            if not done:
                target = (
                    reward
                    + self.gamma * torch.max(self.target_model(next_state)).item()
                )
            target_f = self.model(state)
            target_f[0][action] = target
            loss = nn.MSELoss()(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# TODO: implement the structure of the input state
def preprocess_state(component: Component) -> np.ndarray:
    """
    Convert the GDS component state into a 1D numpy array.

    Args:
        component (Component): The current layout component from the game.

    Returns:
        np.ndarray: A flattened array representing the state.
    """
    # Example extraction of layer data; customize as needed for your component details
    layer_data = []
    layer_color_mapping = {
        LayerMapping.nwell: 1,
        LayerMapping.diff: 2,
        LayerMapping.poly: 3,
        LayerMapping.li_ct: 4,
        LayerMapping.li: 5,
        LayerMapping.metal_ct: 6,
        LayerMapping.metal1: 7,
        LayerMapping.via1: 8,
    }

    for layer, polygons in component.get_polygons(by_spec=True).items():
        if layer in [mapping.value for mapping in LayerMapping]:
            # Use an integer to represent the layer type
            layer_value = layer_color_mapping[LayerMapping(layer)]
            for polygon in polygons:
                # Use the polygon data or other features as needed
                # Here we use the number of points in the polygon and their average position
                num_points = len(polygon)
                avg_x = sum(point[0] for point in polygon) / num_points
                avg_y = sum(point[1] for point in polygon) / num_points
                # Add this feature representation to layer_data
                layer_data.extend([layer_value, num_points, avg_x, avg_y])

    # Convert list to numpy array
    state_array = np.array(layer_data)

    # Flatten the array to ensure it's a 1D input for the DQN model
    flattened_state = state_array.flatten()

    # Optionally normalize the data
    # normalized_state = flattened_state / np.max(flattened_state)

    return flattened_state  # or normalized_state if normalization is applied


# Training loop
def train_dqn():
    gds_filepath = "./generated_gds/test_data/li/li_0b760201-1951-4bea-ac0d-e48e973a979d/li_0b760201-1951-4bea-ac0d-e48e973a979d.gds"
    drc_rule_file = "./drc_sky130.lydrc"
    env = StandardCellGame(gds_filepath, drc_rule_file)

    state_size = len(
        preprocess_state(env.get_state())
    )  # You'll need to define this based on your state representation
    action_size = len(Action)  # Number of possible actions
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        state = preprocess_state(state)
        for time in range(500):  # 500 steps per episode
            action = agent.act(state)
            next_state, reward, done = env.step(
                Action(action + 1)
            )  # Assuming Action enum starts at 1
            next_state = preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(
                    f"episode: {e}/{episodes}, score: {env.score}, e: {agent.epsilon:.2}"
                )
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.update_target_model()

    # save the model checkpoint
    output_checkpoint_dir = "model_checkpoints"
    os.makedirs(output_checkpoint_dir, exist_ok=True)
    checkpoint_fileapth = os.path.join(
        output_checkpoint_dir, f"dqn_model_{datetime.now()}.pth"
    )
    agent.save(checkpoint_fileapth)


def play_game_with_dqn_agent(
    agent: DQNAgent, env: StandardCellGame, num_episodes: int = 10
) -> List[Tuple[int, float, int]]:
    episode_results: List[Tuple[int, float, int]] = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward: float = 0.0
        done: bool = False
        step: int = 0

        while not done:
            action: Action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            step += 1

            # Optional: Visualize the game state
            env.draw(env.WINDOW)  # Assuming your game has a draw method

        episode_results.append((episode + 1, total_reward, step))
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Steps: {step}")

    print("Game play finished.")
    return episode_results


if __name__ == "__main__":
    train_dqn()
