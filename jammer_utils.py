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
    while True:
        if constraints is None:
            x = randomiser.integers(0, xs)
            y = randomiser.integers(0, ys)
        else:
            xl, xu = constraints[0]
            yl, yu = constraints[1]
            x = randomiser.integers(xl, xu)
            y = randomiser.integers(yl, yu)
        if map_matrix[x, y] != 0:
            return (x, y)
