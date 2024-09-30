import numpy as np

class RewardCalculator:
    def __init__(self, env):
        self.env = env
        self.num_agents = len(env.agents)
        self.prev_agent_positions = {i: env.agents[i].current_position() for i in range(self.num_agents)}
        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in env.jammers])

    def calculate_step_rewards(self):
        step_rewards = {}
        for agent_id in range(self.num_agents):
            if self.env.agents[agent_id].is_terminated():
                step_rewards[agent_id] = 0.0
                continue

            agent = self.env.agents[agent_id]
            current_pos = agent.current_position()

            reward = 0.0
            reward += self.observation_reward(agent)
            reward += self.jammer_destruction_reward(agent_id)
            reward += self.movement_reward(agent_id, current_pos)

            # Calculate communication network reward separately
            comm_reward = self.communication_network_reward(agent_id)
            
            # Apply battery scaling to communication reward if battery is below 30%
            if agent.battery < 30:
                battery_factor = 1 + (30 - agent.battery) / 30  # Higher factor at lower battery
                comm_reward *= battery_factor

            reward += comm_reward

            step_rewards[agent_id] = reward

            # Update previous states
            self.prev_agent_positions[agent_id] = current_pos

        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in self.env.jammers])
        return step_rewards

    def observation_reward(self, agent):
        # Reward for all currently visible targets and jammers
        targets_visible = np.sum(agent.full_state[2] == 0.0)
        jammers_visible = np.sum(agent.full_state[3] == 0.0)
        return (targets_visible + jammers_visible) * 10  # 10 points for each visible target or jammer

    def jammer_destruction_reward(self, agent_id):
        reward = 0.0
        for jammer_id, (prev_state, current_state) in enumerate(zip(self.prev_jammer_states, 
                                                                    [j.get_destroyed() for j in self.env.jammers])):
            if not prev_state and current_state and self.env.jammers[jammer_id].destroyed_by == agent_id:
                reward += 100  # 100 points for destroying a jammer
        return reward

    def communication_network_reward(self, agent_id):
        for network in self.env.networks:
            if agent_id in network:
                return len(network) * 5  # 5 points per agent in the network
        return 0

    def movement_reward(self, agent_id, current_pos):
        prev_pos = self.prev_agent_positions[agent_id]
        targets_pos = [t.current_position() for t in self.env.targets]
        jammers_pos = [j.current_position() for j in self.env.jammers if not j.get_destroyed()]

        prev_dist_to_targets = min(self.distance(prev_pos, t) for t in targets_pos) if targets_pos else float('inf')
        current_dist_to_targets = min(self.distance(current_pos, t) for t in targets_pos) if targets_pos else float('inf')

        prev_dist_to_jammers = min(self.distance(prev_pos, j) for j in jammers_pos) if jammers_pos else float('inf')
        current_dist_to_jammers = min(self.distance(current_pos, j) for j in jammers_pos) if jammers_pos else float('inf')

        reward = 0.0
        if current_dist_to_targets < prev_dist_to_targets:
            reward += 5  # 5 points for moving closer to a target
        if current_dist_to_jammers < prev_dist_to_jammers:
            reward += 5  # 5 points for moving closer to a jammer

        return reward

    @staticmethod
    def distance(pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def reset(self):
        self.prev_agent_positions = {i: self.env.agents[i].current_position() for i in range(self.num_agents)}
        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in self.env.jammers])