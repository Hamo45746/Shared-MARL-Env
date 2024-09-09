import numpy as np
from jammer import Jammer


def create_jammers(n_jammers, map_matrix, randomiser, jam_radius, pos_list=None, constraints=None):
    """
    Initializes jammers on a map (map_matrix) at random positions.
    REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/agent_utils
    """
    jammers = []
    for i in range(n_jammers):
        if pos_list and i < len(pos_list):
            x, y = pos_list[i]
        else:
            x, y = feasible_position(randomiser, map_matrix, constraints=constraints)
        jammer = Jammer(jam_radius)
        jammer.set_position(x, y)
        jammers.append(jammer)
    return jammers

def feasible_position(randomiser, map_matrix, constraints=None):
    """
    Returns a feasible position on map (map_matrix).
    REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/agent_utils
    """
    xs, ys = map_matrix.shape
    building_positions = np.argwhere(map_matrix == 0) 
    while True:
        if constraints is None:
            x = randomiser.integers(0, xs)
            y = randomiser.integers(0, ys)
        else:
            xl, xu = constraints[0]
            yl, yu = constraints[1]
            x = randomiser.integers(xl, xu)
            y = randomiser.integers(yl, yu)
        if map_matrix[x, y] != 0 and is_near_building((x,y), building_positions, max_distance=2):
            return (x, y)

def is_near_building(position, building_positions, max_distance=3):
    """
    Checks if the given position is within the specified max_distance from any building.
    
    Parameters:
    - position: Tuple (x, y) of the jammer's position.
    - building_positions: Array of building coordinates (x, y).
    - max_distance: Maximum allowed distance to a building (in pixels).
    
    Returns:
    - True if within max_distance of any building, otherwise False.
    """
    x, y = position
    # Calculate Manhattan (city block) distance for grid-based checking
    distances = np.abs(building_positions - np.array([x, y])).sum(axis=1)
    return np.any(distances <= max_distance)