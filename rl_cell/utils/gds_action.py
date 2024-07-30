import random
from typing import List, Tuple

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component

LayerMetal1 = (67, 20)


def get_gds_wl(component: Component) -> List[str]:
    # Get the bounding box of the component
    bbox = component.bbox
    width = bbox[1][0] - bbox[0][0]
    height = bbox[1][1] - bbox[0][1]

    return [width, height]


def random_shift(component: Component, shift_percentage: float = 0.1) -> List[float]:

    wl_list = get_gds_wl(component)

    # Calculate maximum shifts
    max_dx = wl_list[0] * shift_percentage
    max_dy = wl_list[1] * shift_percentage

    # Generate random shift
    dx = random.uniform(-max_dx, max_dx)
    dy = random.uniform(-max_dy, max_dy)

    return [dx, dy]


import random
from typing import Tuple

import numpy as np
from gdsfactory.component import Component


def shift_layer_by_edge(
    component: Component, layer_mapping: Tuple[int, int], shift_amount: float = None
) -> Component:
    """
    Shift an edge between two neighboring points of a polygon by a random value.

    Args:
        - component (Component): The component containing the polygon to be modified.
        - layer_mapping (Tuple[int, int]): The layer mapping information.

    Returns:
        - Component: A new component with the modified layer.
    """
    # Copy the component
    shifted_component = component.copy()

    # Retrieve the polygons to modify based on the specified layer mapping
    polygons_to_modify = shifted_component.get_polygons(by_spec=True)[layer_mapping]

    # Select a random polygon to modify
    polygon_index = random.randint(0, len(polygons_to_modify) - 1)
    polygon_to_modify = polygons_to_modify[polygon_index]

    # Check if the polygon has at least two points
    if len(polygon_to_modify) < 2:
        modified_polygons = polygons_to_modify
    else:
        # Select a random starting point index within the polygon
        start_index = random.randint(0, len(polygon_to_modify) - 2)

        # Retrieve the two neighboring points
        point1 = np.array(polygon_to_modify[start_index])
        point2 = np.array(polygon_to_modify[start_index + 1])

        # Determine the common coordinate axis to modify (x or y)
        modify_x = point1[0] == point2[0]

        # Generate a random float for the shift
        random_shift = (
            shift_amount if shift_amount else random.uniform(-2, 2)
        )  # NOTE: 1.82 is the width of the inv standard cell

        # Create a modified polygon with the shifted edge
        modified_polygon = []
        for i, point in enumerate(polygon_to_modify):
            new_point = np.array(point)
            if i == start_index or i == start_index + 1:
                if modify_x:
                    new_point[0] += random_shift
                else:
                    new_point[1] += random_shift
            modified_polygon.append(new_point.tolist())

        # Replace the modified polygon in the list of polygons
        modified_polygons = polygons_to_modify[:]
        modified_polygons[polygon_index] = modified_polygon

    # Remove the original layer from the component
    shifted_component.remove_layers(layers=[layer_mapping])

    # Add the modified polygons to the component
    for modified_polygon in modified_polygons:
        shifted_component.add_polygon(modified_polygon, layer_mapping)

    return shifted_component


def shift_layer_by_object(
    component: Component,
    layer_mapping: Tuple[int, int],
    shift_amount: List[float] = None,
) -> Component:
    """
    Shift an object from a layer by a random value.

    Args:
        - component (Component): The component containing the polygon to be modified.
        - layer_mapping (Tuple[int, int]): The layer mapping information.

    Returns:
        - Component: A new component with the modified layer.
    """

    shifted_component = component.copy()
    polygons_to_modify = shifted_component.get_polygons(by_spec=True)[layer_mapping]

    random_shift_amount = (
        shift_amount if shift_amount else random_shift(shifted_component)
    )

    modified_polygons = []
    for polygon in polygons_to_modify:
        modified_polygon = [
            (point[0] + random_shift_amount[0], point[1] + random_shift_amount[1])
            for point in polygon
        ]
        modified_polygons.append(modified_polygon)

    # Remove the original layer
    shifted_component.remove_layers(layers=[layer_mapping])

    # Add the modified positions of the layer
    for modified_polygon in modified_polygons:

        shifted_component.add_polygon(modified_polygon, layer_mapping)

    return shifted_component
