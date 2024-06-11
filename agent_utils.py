from discrete_agent import DiscreteAgent
from continuous_agent import ContinuousAgent
from target import Target
import numpy as np

def create_agents(nagents, map_matrix, obs_range, randomiser, pos_list=None, agent_type='discrete', flatten=False, randinit=False, constraints=None):
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
    if agent_type == 'discrete':
        agent_class = DiscreteAgent  
    else: 
        agent_class = ContinuousAgent

    for i in range(nagents):
        xinit, yinit = (0, 0)
        if pos_list and i < len(pos_list):
            xinit, yinit = pos_list[i]
        elif randinit and feasible_positions:
            idx = randomiser.integers(0, len(feasible_positions))
            xinit, yinit = feasible_positions.pop(idx)  # Remove to avoid reuse
        else:
            xinit, yinit = 0, 0
        agent = agent_class(xs, ys, map_matrix, randomiser, obs_range=obs_range, flatten=flatten)
        agent.set_position(xinit, yinit)
        agents.append(agent)
    return agents

def create_targets(ntargets, map_matrix, obs_range, randomiser, pos_list=None, goal_pos_list=None, flatten=False, randinit=False, constraints=None):
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
            idx = randomiser.integers(0, len(feasible_positions))
            xinit, yinit = feasible_positions.pop(idx)  # Remove to avoid reuse
        else:
            xinit, yinit = 0, 0

        xgoal, ygoal = (0, 0)
        if goal_pos_list and i < len(goal_pos_list):
            xgoal, ygoal = goal_pos_list[i]
        elif randinit and feasible_positions:
            idx = randomiser.integers(0, len(feasible_positions))
            xgoal, ygoal = feasible_positions.pop(idx)  # Remove to avoid reuse
        else:
            xgoal, ygoal = (0, 0)
        
        target = Target(xs, ys, map_matrix, randomiser, start_pos=[xinit, yinit], goal_pos=[xgoal, ygoal], obs_range=obs_range, flatten=flatten)
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
