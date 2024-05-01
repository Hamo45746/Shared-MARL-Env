import numpy as np
from agent import DiscreteAgent
from target import Target

def create_agents(nagents, map_matrix, obs_range, randomizer, pos_list=None, flatten=False, randinit=False, constraints=None):
    """Initializes the agents on a map (map_matrix).

     -nagents: the number of agents to put on the map
     -randinit: if True will place agents in random, feasible locations
                if False will place all agents at 0
     expanded_mat: This matrix is used to spawn non-adjacent agents
     """
    xs, ys = map_matrix.shape
    agents = []

    # Precompute feasible positions
    feasible_positions = get_feasible_positions(map_matrix)#, expanded_mat)

    for i in range(nagents):
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
            xgoal, ygoal = 0, 0

        agent = Target(xs, ys, map_matrix, randomizer, start_pos=[xinit,yinit], goal_pos= [xgoal, ygoal], obs_range=obs_range, flatten=flatten)
        agent.set_position(xinit, yinit)
        agents.append(agent)
    return agents

def create_targets(nagents, map_matrix, obs_range, randomizer, pos_list=None, goal_pos_list=None, flatten=False, randinit=False, constraints=None):
    """Initializes the targets on a map (map_matrix)."""
    xs, ys = map_matrix.shape
    agents = []

    # Precompute feasible positions
    feasible_positions = get_feasible_positions(map_matrix)#, expanded_mat)

    for i in range(nagents):
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
        
        agent = Target(xs, ys, map_matrix, randomizer, start_pos=[xinit, yinit], goal_pos=[xgoal, ygoal], obs_range=obs_range, flatten=flatten)
        agent.set_position(xinit, yinit)
        agents.append(agent)
    return agents

def get_feasible_positions(map_matrix):#, expanded_mat):
    feasible_positions = []
    xs, ys = map_matrix.shape
    for x in range(xs):
        for y in range(ys):
            if map_matrix[x, y] != 0:
                feasible_positions.append((x, y))
    return feasible_positions

def feasible_position_exp(randomizer, map_matrix, expanded_mat, constraints=None):
    """Returns a feasible position on map (map_matrix).
       DEPRECIATED """
    xs, ys = map_matrix.shape
    while True:
        if constraints is None:
            x = randomizer.integers(0, xs)
            y = randomizer.integers(0, ys)
        else:
            xl, xu = constraints[0]
            yl, yu = constraints[1]
            x = randomizer.integers(xl, xu)
            y = randomizer.integers(yl, yu)
        if map_matrix[x, y] != 0 and expanded_mat[x + 1, y + 1] != 0: # Changed to 0 for my purposes (-1 for PettingZoo)
            return (x, y)


def set_agents(agent_matrix, map_matrix):
    # check input sizes
    if agent_matrix.shape != map_matrix.shape:
        raise ValueError("Agent configuration and map matrix have mis-matched sizes")

    agents = []
    xs, ys = agent_matrix.shape
    for i in range(xs):
        for j in range(ys):
            n_agents = agent_matrix[i, j]
            if n_agents > 0:
                if map_matrix[i, j] != 0:
                    raise ValueError(
                        "Trying to place an agent into a building: check map matrix and agent configuration"
                    )
                agent = DiscreteAgent(xs, ys, map_matrix)
                agent.set_position(i, j)
                agents.append(agent)
    return agents
