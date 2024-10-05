import numpy as np

def calculate_continuous_reward(agent, env):
    reward = 0
    # Check if the agent hits another agent or target
    for other_agent in env.agent_layer.agents:
         if other_agent != agent and np.array_equal(agent.current_position, other_agent.current_position):
             reward -= 1

    for target in env.target_layer.targets:
         if np.array_equal(agent.current_position, target.current_position):
             reward -= 1

    if not agent.valid_move:
        reward -= 5

    #velocity reward - not going to use this - or maybe i should??
    velocity_norm = np.linalg.norm(agent.observation_state["velocity"])
    if velocity_norm > 2:
        #print('here')
        reward -= 10

    percentage_new_information = agent.gains_information() 
    if percentage_new_information > 0:
        reward += (percentage_new_information*2)
    else:
        reward -= 5

    obs_half_range = agent._obs_range // 2
    agent_pos = agent.current_position()

    # # Check if the agent finds a target - maybe give it a big bonus reward if it finds all targets 
    # for target in env.target_layer.targets:
    #     target_pos = target.current_position()
    #     if (agent_pos[0] - obs_half_range <= target_pos[0] <= agent_pos[0] + obs_half_range and 
    #         agent_pos[1] - obs_half_range <= target_pos[1] <= agent_pos[1] + obs_half_range):
    #         reward += 60

    # Check if the agent communicates information to other drones
    # Agents only get this reward for one time step when they communicate and then they don't get it 
    # again for another 30 time steps - so it needs to be larger 
    # if agent.communicates_information():
    #     reward += 8

    # # Check if the agent calls the obstacle avoidance method
    # if agent.calls_obstacle_avoidance():
    #     print("obs avoidance")
    #     reward -= 10

    # check if the agent has to0 large angle choice 
    # if agent.angle_change():
    #     reward -= 2

    if env.using_goals and agent.goal_area is not None:
        current_distance_to_goal = agent.calculate_distance_to_goal()
        if current_distance_to_goal is not None and agent.previous_distance_to_goal is not None:
            if agent.goal_step_counter < 60:
                if current_distance_to_goal < 40:
                    reward += 100 
                    print("within 40")
                    agent.goal_step_counter += 1
                elif current_distance_to_goal < agent.previous_distance_to_goal:
                    print("getting closer")
                    reward += 40
                elif current_distance_to_goal >= agent.previous_distance_to_goal:
                    print("getting futher way")
                    reward -= 3
            else:
                print("post goal reward")
                reward += 5
            agent.previous_distance_to_goal = current_distance_to_goal

    #Jammer reward for removing it - If its near the same zone as a jammer it will remove it 

    return reward
