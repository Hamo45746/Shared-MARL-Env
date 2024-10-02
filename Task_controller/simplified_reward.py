import numpy as np

class RewardCalculator:
    def __init__(self, env):
        self.env = env
        self.num_agents = len(env.agents)
        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in env.jammers])
        self.prev_agent_positions = {i: env.agents[i].current_position() for i in range(self.num_agents)}
        self.prev_target_positions = [target.current_position() for target in env.targets]

    def calculate_final_rewards(self, actions_dict):
        rewards = {}
        for agent_id in range(self.num_agents):
            if self.env.agents[agent_id].is_terminated():
                rewards[agent_id] = 0.0
                continue

            agent = self.env.agents[agent_id]
            reward = 0.0
            
            # Penalty for invalid action
            if agent_id in actions_dict:
                reward += self.invalid_action_penalty(agent_id, actions_dict[agent_id])

            # Reward for observed targets and jammers in full state and local observation
            reward += self.observation_reward(agent)

            # Reward for jammer destruction
            reward += self.jammer_destruction_reward(agent_id)

            # Reward for being in a communication network
            reward += self.communication_network_reward(agent_id)

            # Reward for moving closer to jammers
            reward += self.jammer_proximity_reward(agent_id)

            # Reward for target tracking (movement-based, not observation-based)
            reward += self.target_tracking_reward(agent_id)

            rewards[agent_id] = reward

        # Update previous states for next step
        self.update_previous_states()
        return rewards
    
    def invalid_action_penalty(self, agent_id, action):
        agent = self.env.agents[agent_id]
        valid_actions = agent.get_valid_actions()
        if action not in valid_actions:
            return -50.0  # Significant penalty for invalid action
        return 0.0

    def observation_reward(self, agent):
        full_state = agent.get_observation()['full_state']
        local_obs = agent.get_observation()['local_obs']

        # Reward for targets and jammers in full state (including those known through communication)
        targets_visible_full = np.sum(full_state[2] == 0.0)  # Layer 2 for targets
        jammers_visible_full = np.sum(full_state[3] == 0.0)  # Layer 3 for jammers
        full_state_reward = (targets_visible_full * 5) + (jammers_visible_full * 7.5)
        
        # Additional reward for targets and jammers in local observation range
        targets_visible_local = np.sum(local_obs[2] == 0.0)
        jammers_visible_local = np.sum(local_obs[3] == 0.0)
        local_reward = (targets_visible_local * 5) + (jammers_visible_local * 7.5)
        
        return full_state_reward + local_reward

    def jammer_destruction_reward(self, agent_id):
        reward = 0.0
        for jammer_id, (prev_state, current_state) in enumerate(zip(self.prev_jammer_states,
                                                                    [j.get_destroyed() for j in self.env.jammers])):
            if not prev_state and current_state and self.env.jammers[jammer_id].destroyed_by == agent_id:
                reward += 100  # 100 points for destroying a jammer
        return reward

    def communication_network_reward(self, agent_id):
        agent = self.env.agents[agent_id]
        for network in self.env.networks:
            if agent_id in network:
                base_reward = len(network) * 3
                if agent.battery < 30:
                    battery_factor = 1 + (30 - agent.battery) / 30
                    return base_reward * battery_factor
                return base_reward
        return 0

    def jammer_proximity_reward(self, agent_id):
        agent = self.env.agents[agent_id]
        prev_pos = np.array(self.prev_agent_positions[agent_id])
        current_pos = np.array(agent.current_position())
        
        active_jammers = [j for j in self.env.jammers if not j.get_destroyed()]
        if not active_jammers:
            return 0.0

        prev_dist = min(np.linalg.norm(prev_pos - np.array(j.current_position())) for j in active_jammers)
        current_dist = min(np.linalg.norm(current_pos - np.array(j.current_position())) for j in active_jammers)

        if current_dist < prev_dist:
            return 5.0  # Reward for moving closer to a jammer
        return 0.0

    def target_tracking_reward(self, agent_id):
        agent = self.env.agents[agent_id]
        current_pos = np.array(agent.current_position())
        
        reward = 0.0
        for prev_target_pos, current_target_pos in zip(self.prev_target_positions, 
                                                       [t.current_position() for t in self.env.targets]):
            prev_dist = np.linalg.norm(np.array(self.prev_agent_positions[agent_id]) - np.array(prev_target_pos))
            current_dist = np.linalg.norm(current_pos - np.array(current_target_pos))
            
            if current_dist < prev_dist:
                reward += 2.0  # Reward for moving closer to a target

        return reward

    def update_previous_states(self):
        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in self.env.jammers])
        self.prev_agent_positions = {i: agent.current_position() for i, agent in enumerate(self.env.agents)}
        self.prev_target_positions = [target.current_position() for target in self.env.targets]

    def reset(self):
        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in self.env.jammers])
        self.prev_agent_positions = {i: agent.current_position() for i, agent in enumerate(self.env.agents)}
        self.prev_target_positions = [target.current_position() for target in self.env.targets]