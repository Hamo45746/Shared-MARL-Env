from discrete_agent import DiscreteAgent
from continuous_agent import ContinuousAgent
from task_allocation_agent import TaskAllocationAgent
# from target import Target
# import numpy as np

def create_agents(nagents, map_matrix, obs_range, randomiser, path_preprocessor, pos_list=None, agent_type='discrete', flatten=False, randinit=False, constraints=None):
    """Initializes the agents on a map (map_matrix).

     -nagents: the number of agents to put on the map
     -randinit: if True will place agents in random, feasible locations
                if False will place all agents at 0
     expanded_mat: This matrix is used to spawn non-adjacent agents
     """
    xs, ys = map_matrix.shape
    agents = []

    # Precompute feasible positions
    feasible_positions = get_feasible_positions(map_matrix)
    if agent_type == 'discrete':
        agent_class = DiscreteAgent
    elif agent_type == 'task_allocation':
        agent_class = TaskAllocationAgent  
    else: 
        agent_class = ContinuousAgent

    for i in range(nagents):
        xinit, yinit = (0, 0)
        if pos_list and i < len(pos_list):
            xinit, yinit = pos_list[i]
        elif randinit and feasible_positions:
            idx = randomiser.integers(0, len(feasible_positions))
            xinit, yinit = feasible_positions.pop(idx)  # Remove to avoid reuse
            
        if agent_type == 'task_allocation':
            agent = agent_class(xs, ys, map_matrix, randomiser, path_preprocessor, obs_range=obs_range, flatten=flatten, max_steps_per_action=15)
        else:
            agent = agent_class(xs, ys, map_matrix, randomiser, obs_range=obs_range, flatten=flatten)
            
        agent.set_position(xinit, yinit)
        agents.append(agent)
        #this is for lunch and learn 
        agent.origin = agent.current_pos
    return agents

def get_feasible_positions(map_matrix):
    feasible_positions = []
    xs, ys = map_matrix.shape
    for x in range(xs):
        for y in range(ys):
            if map_matrix[x, y] != 0:
                feasible_positions.append((x, y))
    return feasible_positions