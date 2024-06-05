# Constants for rewards and penalties
        # self.JAMMER_DISCOVERY_REWARD = self.config['jammer_discovery_reward']
        # self.TARGET_DISCOVERY_REWARD = self.config['target_discovery_reward']
        # self.TRACKING_REWARD = self.config['tracking_reward']
        # self.DESTRUCTION_REWARD = self.config['destruction_reward']
        # self.MOVEMENT_PENALTY = self.config['movement_penalty']
        # self.EXPLORATION_REWARD = self.config['exploration_reward']


def compute_path_reward(self, agent_id, chosen_location, path_steps):
        """
        Compute the reward based on the agent's path and the encounters along it, including exploration of outdated areas.

        Args:
        - agent_id (int): ID of the agent.
        - chosen_location (tuple): The final destination chosen by the agent.
        - path_steps (list): A list of tuples representing the path coordinates.

        Returns:
        - float: The computed reward for the path taken.
        """
        reward = 0

        target_identified = False
        #TODO: Fix this up - move to reward file
        for step in path_steps:
            # Check for jammer destruction
            if step == chosen_location and self.is_jammer_location(step):
                reward += self.DESTRUCTION_REWARD
                self.destroy_jammer(step) #

            # Check for target identification and tracking
            if self.is_target_in_observation(agent_id, step):
                if not target_identified:
                    reward += self.TARGET_DISCOVERY_REWARD
                    target_identified = True
                else:
                    reward += self.TRACKING_REWARD

            # Reward for exploring outdated regions
            if self.is_information_outdated(step, self.OUTDATED_INFO_THRESHOLD):
                reward += self.EXPLORATION_REWARD
                self.update_global_state(agent_id, step)

        return reward