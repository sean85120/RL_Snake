import json
import os
import random
from enum import Enum, auto
from typing import List
from uuid import uuid4

import gdsfactory as gf
import numpy as np
import pygame
from gdsfactory.component import Component

from rl_cell.src.layer_mapping import LayerMapping
from rl_cell.utils.gds_action import shift_layer_by_edge, shift_layer_by_object
from rl_cell.utils.gds_tools import get_error_edge_pairs, run_drc_on_gds_file

# Initialize Pygame
pygame.init()


# Colors
class Color(Enum):
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    BLACK = (0, 0, 0)
    PINK = (255, 192, 203)
    BLUE = (0, 0, 255)
    PURPLE = (255, 0, 255)


# Layout and window size configuration
GRID_WIDTH = 450
GRID_HEIGHT = 800
CELL_WIDTH = 4
CELL_HEIGHT = 4
CELL_SIZE = 30

# Calculate window dimensions based on the cell dimensions and grid size
WINDOW_WIDTH = int(GRID_WIDTH)
WINDOW_HEIGHT = int(GRID_HEIGHT)
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("RL Layout Game - Continuous Movement")


class Action(Enum):
    Extend = auto()
    Move = auto()


class StandardCellGame:
    """
    Try to fix DRC error of the GDS layout file
    by moving and extending the layout cells.
    """

    def __init__(self, gds_filepath: str, drc_filepath: str):
        self.gds_filepath = gds_filepath
        self.drc_filepath = drc_filepath
        self.hearts = 5
        self.current_gds_filepath = gds_filepath
        self.reset()
        self.last_drc_errors = []

    def reset(self):
        self.score: int = 0
        self.steps: int = 0
        self.hearts: int = 5
        self.game_over: bool = False
        self.component: Component = self.load_initial_layers()
        self.specified_layer: LayerMapping = LayerMapping.li
        self.last_drc_errors = self.check_drc()
        return self.get_state()

    def load_initial_layers(self) -> Component:
        # Import GDS file and extract polygons
        component = gf.import_gds(self.gds_filepath)
        return component

    def get_state(self):
        # Return the current state for the game, could be the layout details or a representation
        return self.component

    def save_gds_file(self) -> str:
        dir_path = os.path.dirname(self.gds_filepath)
        gds_filename = f"{self.specified_layer.name}_{uuid4()}.gds"
        self.current_gds_filepath = os.path.join(dir_path, gds_filename)
        return self.current_gds_filepath

    def calculate_shift_amount(self):
        """Calculate the possible shift amount for the layer."""

        def edge_pairs_distance(edge1, edge2):
            """
            Calculate the distance between two edge pairs.

            Args:
            edge1: List of two points defining the first edge, each point is a list of [x, y].
            edge2: List of two points defining the second edge, each point is a list of [x, y].

            Returns:
            List of distances between the corresponding points of the edges.
            """
            edge1 = np.array(edge1)
            edge2 = np.array(edge2)

            distances = np.linalg.norm(edge1 - edge2, axis=1)
            return distances.tolist()

        def count_most_common_category(error_list: List[dict]):
            category_count = {}
            for edge in error_list:
                category = edge["category"]
                if category in category_count:
                    category_count[category] += 1
                else:
                    category_count[category] = 1

            return max(category_count, key=category_count.get)

        error_edge_pairs_list = self.last_drc_errors

        most_common_category = count_most_common_category(error_edge_pairs_list)

        common_category_edges = [
            edge["edge_pairs"]
            for edge in error_edge_pairs_list
            if edge["category"] == most_common_category
        ]

        # Calculating distances for edge pairs in the most common category
        distances_list = []
        for pairs in common_category_edges:
            for pair in pairs:
                distances = edge_pairs_distance(pair[0], pair[1])
                distances_list.append(distances)

        return distances_list

    @staticmethod
    def combine_distances(distances):

        # Convert to numpy array
        distances = np.array(distances)
        # Ensure the distances are two-dimensional
        if distances.ndim == 1:
            # If it's a 1D array, reshape to 2D with one row
            distances = distances.reshape(1, -1)
        # Calculate Euclidean distance
        combined_distance = np.sqrt(np.sum(distances**2, axis=1))
        return combined_distance[0]

    def action_extend(self):
        """Extend the given layer in random width or length."""
        original_component = self.component
        possible_shift_amount = self.calculate_shift_amount()
        combined_distances = [self.combine_distances(d) for d in possible_shift_amount]

        # Randomly select a combined distance
        pure_distance = random.choice(combined_distances)

        # random_distance = random.choice(possible_shift_amount)
        # pure_distance = self.combine_distances(random_distance)
        self.component = shift_layer_by_edge(
            self.component, self.specified_layer.value, pure_distance
        )

        # Check the impact on score before saving the GDS file
        original_score = self.score
        self.update_score_and_screen()
        if self.score < original_score:
            self.hearts -= 1
            self.component = original_component  # Revert the component change
            if self.hearts == 0:
                self.game_over = True
                self.reset()
        else:
            self.save_gds_file()

        return self.component

    def action_move(self):
        """Move the given layer node by dx, dy."""
        original_component = self.component

        possible_shift_amount = self.calculate_shift_amount()
        random_distance = random.choice(possible_shift_amount)

        self.component = shift_layer_by_object(
            self.component, self.specified_layer.value, random_distance
        )

        # Check the impact on score before saving the GDS file
        original_score = self.score
        self.update_score_and_screen()
        if self.score < original_score:
            self.hearts -= 1
            self.component = original_component  # Revert the component change
            if self.hearts == 0:
                self.game_over = True
                self.reset()
        else:
            self.save_gds_file()

        return self.component

    def update_score_and_screen(self):
        """Update the screen and score based on DRC errors."""
        current_drc_errors = self.check_drc()
        error_diff = len(self.last_drc_errors) - len(current_drc_errors)
        self.score += 1 if error_diff > 0 else -1
        self.last_drc_errors = current_drc_errors
        self.draw(WINDOW)

    def step(self, action: Action):
        """Steps of the games"""
        if action == Action.Extend:
            self.action_extend()
        elif action == Action.Move:
            if not self.game_over:
                self.action_move()

        self.steps += 1
        return self.get_state(), self.score, self.game_over

    def check_drc(self) -> List[dict]:
        """
        Check DRC errors of the current layout.

        Returns:
            - List[float, float]: The edge pairs of DRC errors.
        """
        error_report_filepath = run_drc_on_gds_file(
            self.current_gds_filepath, self.drc_filepath
        )
        error_edge_pairs_filepath = get_error_edge_pairs(error_report_filepath)

        with open(error_edge_pairs_filepath, "r") as f:
            json_content = json.load(f)

        return json_content

    def draw(self, window: pygame.Surface):
        window.fill(Color.BLACK.value)
        component = self.component

        # Layer to color mapping based on LayerMapping
        layer_color_mapping = {
            LayerMapping.nwell: Color.YELLOW.value,
            LayerMapping.diff: Color.RED.value,
            LayerMapping.poly: Color.PINK.value,
            LayerMapping.li_ct: Color.WHITE.value,
            LayerMapping.li: Color.BLUE.value,  # Same color for li and metal1
            LayerMapping.metal_ct: Color.GREEN.value,
            LayerMapping.metal1: Color.BLUE.value,  # Same color for li and metal1
            LayerMapping.via1: Color.PURPLE.value,
        }

        # Scale factors for the coordinates
        scale_x = CELL_WIDTH * CELL_SIZE
        scale_y = CELL_HEIGHT * CELL_SIZE

        # Calculate the bounding box of the entire layout
        all_x = []
        all_y = []
        for polygons in component.get_polygons(by_spec=True).values():
            for polygon in polygons:
                all_x.extend([point[0] for point in polygon])
                all_y.extend([point[1] for point in polygon])
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Calculate the total layout size
        layout_width = (max_x - min_x) * scale_x
        layout_height = (max_y - min_y) * scale_y

        # Calculate the offset to center the layout
        offset_x = (WINDOW_WIDTH - layout_width) / 2 - min_x * scale_x
        offset_y = (WINDOW_HEIGHT - layout_height) / 2 - min_y * scale_y

        # Draw GDS layers
        for layer, polygons in component.get_polygons(by_spec=True).items():
            if layer in [mapping.value for mapping in LayerMapping]:
                # Get the color for the layer
                color = layer_color_mapping[LayerMapping(layer)]

                for polygon in polygons:
                    # Scale and offset the coordinates to center the layout
                    scaled_points = [
                        (int(x * scale_x + offset_x), int(y * scale_y + offset_y))
                        for x, y in polygon
                    ]

                    # Draw the filled polygon
                    pygame.draw.polygon(window, color, scaled_points)

        # Draw buttons
        self.extend_button = pygame.Rect(10, WINDOW_HEIGHT - 40, 100, 30)
        self.move_button = pygame.Rect(120, WINDOW_HEIGHT - 40, 100, 30)
        pygame.draw.rect(window, Color.GREEN.value, self.extend_button)
        pygame.draw.rect(window, Color.BLUE.value, self.move_button)

        font = pygame.font.Font(None, 24)
        extend_text = font.render("Extend", True, Color.BLACK.value)
        move_text = font.render("Move", True, Color.BLACK.value)
        window.blit(extend_text, (15, WINDOW_HEIGHT - 35))
        window.blit(move_text, (125, WINDOW_HEIGHT - 35))

        # Draw score
        score_text = font.render(f"Score: {self.score}", True, Color.WHITE.value)
        window.blit(score_text, (10, 10))

        # Draw hearts (lives) in the top center
        hearts_x_start = (WINDOW_WIDTH - (self.hearts * 30)) // 2
        for i in range(self.hearts):
            pygame.draw.ellipse(
                window,
                Color.RED.value,
                pygame.Rect(hearts_x_start + 30 * i, 40, 20, 20),
            )

        if self.game_over:
            game_over_text = font.render("Game Over", True, Color.WHITE.value)
            window.blit(
                game_over_text,
                (
                    WINDOW_WIDTH // 2 - game_over_text.get_width() // 2,
                    WINDOW_HEIGHT // 2,
                ),
            )
            pygame.display.update()
            pygame.time.wait(2000)  # Wait for 2 seconds before resetting the game
            self.reset()

        pygame.display.update()


if __name__ == "__main__":
    gds_filepath = "./generated_gds/test_data/li/li_0b760201-1951-4bea-ac0d-e48e973a979d/li_0b760201-1951-4bea-ac0d-e48e973a979d.gds"
    drc_rule_file = "./drc_sky130.lydrc"
    game = StandardCellGame(gds_filepath, drc_rule_file)

    run = True
    while run:
        pygame.time.delay(100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game.extend_button.collidepoint(event.pos):
                    game.step(Action.Extend)
                elif game.move_button.collidepoint(event.pos):
                    game.step(Action.Move)

        game.draw(WINDOW)

    pygame.quit()
