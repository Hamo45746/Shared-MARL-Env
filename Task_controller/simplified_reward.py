import numpy as np

class RewardCalculator:
    def __init__(self, env):
        self.env = env
        self.num_agents = len(env.agents)
        self.prev_team_metrics = None
        self.prev_agent_contributions = None
        self.prev_destroyed_jammers = set()

    def calculate_final_rewards(self, actions_dict):
        rewards = {i: 0.0 for i in range(self.num_agents)}
        current_team_metrics = self.env.get_metrics()
        current_agent_contributions = self.env.agent_contributions.copy()
        current_destroyed_jammers = set(j for j in self.env.jammer_layer.jammers if j.get_destroyed())
        
        for agent_id in range(self.num_agents):
            if self.env.agents[agent_id].is_terminated():
                continue

            # Penalty for invalid action
            if agent_id in actions_dict:
                invalid_action_penalty = self.check_invalid_action(agent_id, actions_dict[agent_id])
                rewards[agent_id] += invalid_action_penalty

            # Calculate improvements in metrics
            if self.prev_team_metrics is not None and self.prev_agent_contributions is not None:
                # Individual contribution rewards
                map_contribution = current_agent_contributions[agent_id, 0] - self.prev_agent_contributions[agent_id, 0]
                rewards[agent_id] += map_contribution * 0.1 * (1 + current_team_metrics['map_seen'] / 100)

                target_contribution = current_agent_contributions[agent_id, 1] - self.prev_agent_contributions[agent_id, 1]
                rewards[agent_id] += target_contribution * 10

                jammer_contribution = current_agent_contributions[agent_id, 2] - self.prev_agent_contributions[agent_id, 2]
                rewards[agent_id] += jammer_contribution * 15

                # Team-wide improvement rewards
                # map_improvement = current_team_metrics['map_seen'] - self.prev_team_metrics['map_seen']
                # rewards[agent_id] += map_improvement * 0.05

                # target_improvement = current_team_metrics['targets_seen'] - self.prev_team_metrics['targets_seen']
                # rewards[agent_id] += target_improvement * 5

                # jammer_seen_improvement = current_team_metrics['jammers_seen'] - self.prev_team_metrics['jammers_seen']
                # rewards[agent_id] += jammer_seen_improvement * 7.5

            # Jammer destruction reward (agent-specific)
            for jammer in current_destroyed_jammers - self.prev_destroyed_jammers:
                if jammer.destroyed_by == agent_id:
                    rewards[agent_id] += 300  # Significant reward for destroying a jammer

            # Communication network reward
            network_size = len(self.env.networks[agent_id]) if agent_id in self.env.networks else 0
            rewards[agent_id] += network_size 

        # Completion bonus (team-wide)
        completion_bonus = 0
        if current_team_metrics['map_seen'] >= 90:
            completion_bonus += 1000
        if current_team_metrics['targets_seen'] == 100:
            completion_bonus += 1000
        if current_team_metrics['jammers_destroyed'] == 100:
            completion_bonus += 3000
        
        # Distribute completion bonus equally among active agents
        active_agents = [i for i in range(self.num_agents) if not self.env.agents[i].is_terminated()]
        if active_agents:
            for agent_id in active_agents:
                rewards[agent_id] += completion_bonus / len(active_agents)

        self.prev_team_metrics = current_team_metrics
        self.prev_agent_contributions = current_agent_contributions
        self.prev_destroyed_jammers = current_destroyed_jammers
        return rewards

    def check_invalid_action(self, agent_id, action):
        agent = self.env.agents[agent_id]
        valid_actions = agent.get_valid_actions()
        if action not in valid_actions:
            return -50.0  # Significant penalty for invalid action
        return 0.0

    def reset(self):
        self.prev_team_metrics = None
        self.prev_agent_contributions = None
        self.prev_destroyed_jammers = set()