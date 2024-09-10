import numpy as np

class RewardCalculator:
    def __init__(self, env):
        self.env = env
        self.accumulated_rewards = np.zeros(len(env.agents), dtype=np.float32)
        self.prev_observed_cells = [np.full(env.global_state[1].shape, -20, dtype=np.float16) for _ in range(len(env.agents))]
        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in env.jammers], dtype=bool)
        self.step_rewards = np.zeros(len(env.agents), dtype=np.float32)

    def pre_step_update(self):
        self.step_rewards.fill(0)

    def update_exploration_reward(self, agent_id):
        agent_state = self.env.agents[agent_id].full_state[1:]
        new_observed = np.sum((agent_state > -20) & (self.prev_observed_cells[agent_id] == -20))
        self.step_rewards[agent_id] += new_observed * 0.1
        self.prev_observed_cells[agent_id] = np.maximum(self.prev_observed_cells[agent_id], agent_state)

    def update_target_reward(self, agent_id):
        observed_targets = np.sum((self.env.agents[agent_id].full_state[2] >= -0.5) & (self.env.agents[agent_id].full_state[2] <= 0))
        self.step_rewards[agent_id] += observed_targets * 5

    def update_jammer_reward(self):
        current_jammer_states = np.array([jammer.get_destroyed() for jammer in self.env.jammers], dtype=bool)
        newly_destroyed = current_jammer_states & ~self.prev_jammer_states
        if np.any(newly_destroyed):
            jammer_positions = np.array([jammer.current_position() for jammer in self.env.jammers])[newly_destroyed]
            agent_positions = np.array([agent.current_position() for agent in self.env.agents])
            distances = np.linalg.norm(agent_positions[:, np.newaxis] - jammer_positions, axis=2)
            closest_agents = np.argmin(distances, axis=0)
            self.step_rewards[closest_agents] += 10
        self.prev_jammer_states = current_jammer_states

    def update_communication_reward(self):
        for network in self.env.networks:
            network_size_reward = len(network) * 0.5
            self.step_rewards[list(network)] += network_size_reward

    def post_step_update(self):
        battery_factors = 1 + (100 - np.array([agent.get_battery() for agent in self.env.agents])) / 200
        self.accumulated_rewards += self.step_rewards * battery_factors

    def get_rewards(self):
        return self.accumulated_rewards.copy()

    def reset(self):
        self.accumulated_rewards.fill(0)
        for cells in self.prev_observed_cells:
            cells.fill(-20)
        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in self.env.jammers], dtype=bool)
        self.step_rewards.fill(0)