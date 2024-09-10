import numpy as np


class RewardCalculator:
    def __init__(self, env):
        self.env = env
        self.num_agents = len(env.agents)
        self.accumulated_rewards = {i: 0.0 for i in range(self.num_agents)}
        self.prev_observed_cells = [np.full(env.global_state[1].shape, -20, dtype=np.float16) for _ in range(self.num_agents)]
        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in env.jammers], dtype=bool)
        self.step_rewards = {i: 0.0 for i in range(self.num_agents)}

    def pre_step_update(self):
        for i in range(self.num_agents):
            self.step_rewards[i] = 0.0

    def update_exploration_reward(self, agent_id):
        if self.env.agents[agent_id].is_terminated():
            return
        agent_state = self.env.agents[agent_id].full_state[1:]
        new_observed = np.sum((agent_state > -20) & (self.prev_observed_cells[agent_id] == -20))
        self.step_rewards[agent_id] += new_observed * 0.1
        self.prev_observed_cells[agent_id] = np.maximum(self.prev_observed_cells[agent_id], agent_state)

    def update_target_reward(self, agent_id):
        if self.env.agents[agent_id].is_terminated():
            return
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
            for agent_id in closest_agents:
                self.step_rewards[agent_id] += 10
        self.prev_jammer_states = current_jammer_states

    def update_communication_reward(self):
        for network in self.env.networks:
            network_size_reward = len(network) * 0.5
            for agent_id in network:
                if not self.env.agents[agent_id].is_terminated():
                    self.step_rewards[agent_id] += network_size_reward

    def post_step_update(self):
        for i in range(self.num_agents):
            if not self.env.agents[i].is_terminated():
                battery_factor = 1 + (100 - self.env.agents[i].get_battery()) / 200
                self.accumulated_rewards[i] += self.step_rewards[i] * battery_factor
            else:
                self.accumulated_rewards[i] = 0  # Ensure terminated agents have zero reward

    def get_rewards(self):
        return self.accumulated_rewards.copy()

    def reset(self):
        for i in range(self.num_agents):
            self.accumulated_rewards[i] = 0.0
            self.step_rewards[i] = 0.0
        for cells in self.prev_observed_cells:
            cells.fill(-20)
        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in self.env.jammers], dtype=bool)