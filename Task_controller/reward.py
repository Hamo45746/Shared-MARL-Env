import numpy as np


class RewardCalculator:
    """
    Class designed to calculate the reward for all agents over a single task allocation step.
    """
    def __init__(self, env, destroyed_jammers=set(), prev_observed_cells=None):
        self.env = env
        self.num_agents = len(env.agents)
        self.accumulated_rewards = {i: 0.0 for i in range(self.num_agents)}
        if prev_observed_cells == None:
            self.prev_observed_cells = [np.full(env.global_state[1].shape, -20, dtype=np.float16) for _ in range(self.num_agents)]
        else:
            self.prev_observed_cells = prev_observed_cells
        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in env.jammers], dtype=bool)
        self.step_rewards = {i: 0.0 for i in range(self.num_agents)}
        self.destroyed_jammers = destroyed_jammers

    def pre_step_update(self):
        """
        Zero the current step rewards for all agents in prev step.
        """
        for i in range(self.num_agents):
            self.step_rewards[i] = 0.0

    def update_exploration_reward(self, agent_id):
        """
        Reward an agent proportionally the number of cells in its state that are not -20.0 (no info).
        Rewarding information gain.
        
        Args:
            agent_id (int): The index of the agent in the agents list maintianed by Environment and AgentLayer.
        """
        if self.env.agents[agent_id].is_terminated():
            return
        agent_state = self.env.agents[agent_id].full_state[1:]
        new_observed = np.sum((agent_state > -20) & (self.prev_observed_cells[agent_id] == -20))
        self.step_rewards[agent_id] += new_observed * 0.1
        self.prev_observed_cells[agent_id] = np.maximum(self.prev_observed_cells[agent_id], agent_state)

    def update_target_reward(self, agent_id):
        """
        Rewards an agent 50 for each target within its immediate observation range. I.e. agent rewarded
        for each target it has a real time view of.

        Args:
            agent_id (int): The index of the agent in the agents list maintianed by Environment and AgentLayer.
        """
        if self.env.agents[agent_id].is_terminated():
            return
        obs_half_range = self.env.obs_range // 2
        agent_pos = self.env.agent_layer.agents[agent_id].current_position()
        for target in self.env.target_layer.targets:
            target_pos = target.current_position()
            if (agent_pos[0] - obs_half_range <= target_pos[0] <= agent_pos[0] + obs_half_range and 
                agent_pos[1] - obs_half_range <= target_pos[1] <= agent_pos[1] + obs_half_range):
                self.step_rewards[agent_id] += 50

    def update_jammer_reward(self):
        """
        Reward agents for finding jammers (within immediate observation range) and destroying them.
        """
        obs_half_range = self.env.obs_range // 2
        current_jammer_states = np.array([jammer.get_destroyed() for jammer in self.env.jammers], dtype=bool)
        
        # Reward for finding jammers
        for agent_id, agent in enumerate(self.env.agents):
            if agent.is_terminated():
                continue
            agent_pos = agent.current_position()
            for jammer_id, jammer in enumerate(self.env.jammers):
                jammer_pos = jammer.current_position()
                if (agent_pos[0] - obs_half_range <= jammer_pos[0] <= agent_pos[0] + obs_half_range and 
                    agent_pos[1] - obs_half_range <= jammer_pos[1] <= agent_pos[1] + obs_half_range):
                    self.step_rewards[agent_id] += 30  # Reward for finding a jammer

        # Reward for destroying jammers
        newly_destroyed = current_jammer_states & ~self.prev_jammer_states
        if np.any(newly_destroyed):
            for jammer_id, destroyed in enumerate(newly_destroyed):
                if destroyed and jammer_id not in self.destroyed_jammers:
                    destroyer_id = self.env.jammers[jammer_id].destroyed_by
                    if destroyer_id is not None:
                        self.step_rewards[destroyer_id] += 100  # Reward for destroying a jammer
                    self.destroyed_jammers.add(jammer_id)

        self.prev_jammer_states = current_jammer_states

    def update_communication_reward(self):
        """
        Reward all agents for being part of larger multi-hop communication networks.
        """
        for network in self.env.networks:
            network_size_reward = len(network) * 3
            for agent_id in network:
                if not self.env.agents[agent_id].is_terminated():
                    self.step_rewards[agent_id] += network_size_reward

    def post_step_update(self):
        """
        Scale rewards for each agent based on battery percentage. Favor exploration/info gain 
        on high battery and maintaining large communication networks on lower battery.
        """
        for i in range(self.num_agents):
            if not self.env.agents[i].is_terminated():
                battery_level = self.env.agents[i].get_battery()
                exploration_factor = 1 + battery_level / 100  # Higher reward for exploration at high battery
                communication_factor = 1 + (100 - battery_level) / 100  # Higher reward for communication at low battery
                
                # Separate exploration and communication rewards
                exploration_reward = self.step_rewards[i] * exploration_factor
                communication_reward = self.env.networks[i] * 3 * communication_factor if i in self.env.networks else 0
                
                total_reward = exploration_reward + communication_reward
                self.accumulated_rewards[i] += total_reward
            else:
                self.accumulated_rewards[i] = 0  # Ensure terminated agents have zero reward

    def get_rewards(self):
        """
        Returns:
            dict: key: agent_id, value: float accumulated rewards for that agent.
        """
        return {i: float(reward) if np.isfinite(reward) else 0.0 for i, reward in self.accumulated_rewards.items()}

    def reset(self):
        """
        Reset the reward calculator. Should be called at step end. CURRENTLY UNNEEDED
        """
        for i in range(self.num_agents):
            self.accumulated_rewards[i] = 0.0
            self.step_rewards[i] = 0.0
        for cells in self.prev_observed_cells:
            cells.fill(-20)
        self.prev_jammer_states = np.array([jammer.get_destroyed() for jammer in self.env.jammers], dtype=bool)
        
    def episode_reset(self):
        self.reset()
        self.destroyed_jammers.clear()
        
    def get_observed_cells(self):
        return self.prev_observed_cells