import random
from enum import Enum, auto
from typing import List, Tuple

import pygame

# Initialize Pygame
pygame.init()


class Color(Enum):
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLACK = (0, 0, 0)


GRID_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 15
CELL_SIZE = 30

WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Snake Game - Continuous Movement")


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


class SnakeGame:
    def __init__(self) -> None:
        self.initial_speed = 5  # Initial speed in cells per second
        self.speed_increase = 0.5  # Speed increase per score point
        self.reset()

    def reset(self) -> Tuple[bool, ...]:
        self.snake: List[Tuple[int, int]] = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction: Direction = random.choice(list(Direction))
        self.food: Tuple[int, int] = self.spawn_food()
        self.score: int = 0
        self.game_over: bool = False
        self.steps: int = 0
        self.speed = self.initial_speed
        return self.get_state()

    def spawn_food(self) -> Tuple[int, int]:
        while True:
            food = (
                random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1),
            )
            if food not in self.snake:
                return food

    def get_state(self) -> Tuple[bool, ...]:
        head = self.snake[0]
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        return (
            (point_l[0] < 0 or point_l in self.snake),
            (point_r[0] >= GRID_WIDTH or point_r in self.snake),
            (point_u[1] < 0 or point_u in self.snake),
            (point_d[1] >= GRID_HEIGHT or point_d in self.snake),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1],  # food down
        )

    def step(self, action: int) -> Tuple[Tuple[bool, ...], float, bool]:
        # Convert action index to direction
        self.direction = list(Direction)[action]

        return self.move_snake()

    def move_snake(self) -> Tuple[Tuple[bool, ...], float, bool]:
        head = self.snake[0]
        if self.direction == Direction.UP:
            new_head = (head[0], head[1] - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head[0], head[1] + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head[0] - 1, head[1])
        elif self.direction == Direction.RIGHT:
            new_head = (head[0] + 1, head[1])

        self.snake.insert(0, new_head)
        self.steps += 1

        reward = 0.0
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            reward = 10.0
            self.speed += self.speed_increase  # Increase speed when food is eaten
        else:
            self.snake.pop()

        if (
            new_head[0] < 0
            or new_head[0] >= GRID_WIDTH
            or new_head[1] < 0
            or new_head[1] >= GRID_HEIGHT
            or new_head in self.snake[1:]
        ):
            self.game_over = True
            reward = -10.0

        return self.get_state(), reward, self.game_over

    def draw(self, window: pygame.Surface) -> None:
        window.fill(Color.BLACK.value)

        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(
                window,
                Color.GREEN.value,
                (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
            )

        # Draw food
        pygame.draw.rect(
            window,
            Color.RED.value,
            (self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
        )

        # Draw score and speed
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, Color.WHITE.value)
        speed_text = font.render(f"Speed: {self.speed:.1f}", True, Color.WHITE.value)
        window.blit(score_text, (10, 10))
        window.blit(speed_text, (10, 50))

        pygame.display.update()


# Main game loop
def main():
    game = SnakeGame()
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and game.direction != Direction.DOWN:
                    action = Direction.UP.value - 1
                elif event.key == pygame.K_DOWN and game.direction != Direction.UP:
                    action = Direction.DOWN.value - 1
                elif event.key == pygame.K_LEFT and game.direction != Direction.RIGHT:
                    action = Direction.LEFT.value - 1
                elif event.key == pygame.K_RIGHT and game.direction != Direction.LEFT:
                    action = Direction.RIGHT.value - 1
                else:
                    continue

                state, reward, game_over = game.step(action)

                if game_over:
                    print(f"Game Over! Score: {game.score}")
                    game.reset()

        state, reward, game_over = game.move_snake()
        if game_over:
            print(f"Game Over! Score: {game.score}")
            game.reset()

        game.draw(WINDOW)
        clock.tick(game.speed)  # Use the current game speed

    pygame.quit()


if __name__ == "__main__":
    main()
