import numpy as np

def calculate_continuous_reward(agent, env):
    reward = 0

    # Check if the agent hits a building
    if agent.inbuilding(agent.current_position()[0], agent.current_position()[1]):
         reward -= 50

    # Check if the agent hits another agent or target
    for other_agent in env.agent_layer.agents:
         if other_agent != agent and np.array_equal(agent.current_position, other_agent.current_position):
             reward -= 50

    for target in env.target_layer.targets:
         if np.array_equal(agent.current_position, target.current_position):
             reward -= 10

    percentage_new_information = agent.gains_information() 
    #print("percentage_new_information", percentage_new_information)
    # percentage_new_information = int(percentage_new_information)
    # percentage_new_information = percentage_new_information/2
    reward += (percentage_new_information*1.5)

    # # Check if the agent gains new information
    # if agent.gains_information():
    #     reward += 10

    obs_half_range = agent._obs_range // 2
    agent_pos = agent.current_position()

    # Check if the agent finds a target
    for target in env.target_layer.targets:
        target_pos = target.current_position()
        if (agent_pos[0] - obs_half_range <= target_pos[0] <= agent_pos[0] + obs_half_range and 
            agent_pos[1] - obs_half_range <= target_pos[1] <= agent_pos[1] + obs_half_range):
            reward += 50

    # # Check if the agent finds a target
    # for target in env.target_layer.targets:
    #     if target.current_position() in agent.get_observation_state():
    #         reward += 50

    # Check if the agent communicates information to other drones
    if agent.communicates_information():
        reward += 5

    # Check if the agent calls the obstacle avoidance method
    if agent.calls_obstacle_avoidance():
        reward -= 20

    # check if the agent has to change it's angle choice 
    if agent.angle_change():
        reward -= 8

    if agent.goal_area is not None:
        current_distance_to_goal = agent.calculate_distance_to_goal()
        if current_distance_to_goal is not None and agent.previous_distance_to_goal is not None:
            if current_distance_to_goal < agent.previous_distance_to_goal:
                reward += 10  # Reward for moving closer to the goal
            else:
                reward -= 10  # Penalty for moving away from the goal
            agent.previous_distance_to_goal = current_distance_to_goal

    #Jammer reward for removing it - If its near the same zone as a jammer it will remove it 

    return reward
