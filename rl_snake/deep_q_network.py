import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

from rl_snake.snake_game import WINDOW_HEIGHT, WINDOW_WIDTH, Color, SnakeGame


class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.memory: Deque[
            Tuple[Tuple[bool, ...], int, float, Tuple[bool, ...], bool]
        ] = deque(maxlen=10000)
        self.gamma: float = 0.95  # discount rate
        self.epsilon: float = 1.0  # exploration rate
        self.epsilon_min: float = 0.01
        self.epsilon_decay: float = 0.995
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_model = DQN(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(
        self,
        state: Tuple[bool, ...],
        action: int,
        reward: float,
        next_state: Tuple[bool, ...],
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: Tuple[bool, ...]) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()

    def replay(self, batch_size: int) -> None:
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())


def train_dqn_agent(
    episodes: int = 10000, max_steps: int = 1000, batch_size: int = 32
) -> DQNAgent:
    game = SnakeGame()
    state_size = 12  # Adjust based on your state representation
    action_size = 4  # UP, DOWN, LEFT, RIGHT
    agent = DQNAgent(state_size, action_size)

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                break

        if episode % 10 == 0:
            agent.update_target_model()

        if episode % 100 == 0:
            print(
                f"Episode {episode}, Score: {game.score}, Total Reward: {total_reward}"
            )

    return agent


def play_game_with_dqn_agent(agent: DQNAgent, fps: int = 10) -> None:
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake Game - DQN Agent")
    clock = pygame.time.Clock()

    game = SnakeGame()
    state = game.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.act(state)
        state, _, done = game.step(action)
        game.draw(window)

        if done:
            font = pygame.font.Font(None, 48)
            game_over_text = font.render(
                f"Game Over! Score: {game.score}", True, Color.WHITE.value
            )
            window.blit(
                game_over_text, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2 - 24)
            )
            pygame.display.update()
            pygame.time.wait(2000)
            state = game.reset()

        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    trained_agent = train_dqn_agent(episodes=10000)
    play_game_with_dqn_agent(trained_agent)
