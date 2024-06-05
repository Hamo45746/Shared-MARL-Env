from target import Target
import numpy as np

def create_targets(ntargets, map_matrix, obs_range, randomizer, pos_list=None, goal_pos_list=None, flatten=False, randinit=False, constraints=None):
    """Initializes the targets on a map (map_matrix)."""
    xs, ys = map_matrix.shape
    targets = []

    # Precompute feasible positions
    feasible_positions = get_feasible_positions(map_matrix)

    for i in range(ntargets):
        xinit, yinit = (0, 0)
        if pos_list and i < len(pos_list):
            xinit, yinit = pos_list[i]
        elif randinit and feasible_positions:
            idx = randomizer.integers(0, len(feasible_positions))
            xinit, yinit = feasible_positions.pop(idx)  # Remove to avoid reuse
        else:
            xinit, yinit = 0, 0

        xgoal, ygoal = (0, 0)
        if goal_pos_list and i < len(goal_pos_list):
            xgoal, ygoal = goal_pos_list[i]
        elif randinit and feasible_positions:
            idx = randomizer.integers(0, len(feasible_positions))
            xgoal, ygoal = feasible_positions.pop(idx)  # Remove to avoid reuse
        else:
            xgoal, ygoal = (0, 0)
        
        target = Target(xs, ys, map_matrix, randomizer, start_pos=[xinit, yinit], goal_pos=[xgoal, ygoal], obs_range=obs_range, flatten=flatten)
        target.set_position(xinit, yinit)
        targets.append(targets)
    return targets

def get_feasible_positions(map_matrix):#, expanded_mat):
    feasible_positions = []
    xs, ys = map_matrix.shape
    for x in range(xs):
        for y in range(ys):
            if map_matrix[x, y] != 0:
                feasible_positions.append((x, y))
    return feasible_positions