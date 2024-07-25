import numpy as np
from discrete_agent import DiscreteAgent
# from gymnasium import spaces
from gym import spaces # for MARLlib

class TaskAllocationAgent(DiscreteAgent):
    def __init__(
        self,
        xs,
        ys,
        map_matrix,
        randomiser,
        path_preprocessor,
        obs_range=3,
        n_layers=4,
        seed=10,
        flatten=False,
        max_steps_per_action=10,
    ):
        super().__init__(xs, ys, map_matrix, randomiser, obs_range, n_layers, seed, flatten)
        self.max_distance = max_steps_per_action
        self.path = []
        self.path_index = 0
        self.steps_taken = 0
        self.path_preprocessor = path_preprocessor
        self.randomiser = randomiser
        self._action_space = spaces.Discrete((2 * self.max_distance + 1) ** 2)
    
    @property
    def action_space(self):
        return self._action_space
    
    @property
    def observation_space(self):
        return spaces.Dict({
            'local_obs': spaces.Box(low=-20, high=1, shape=self._obs_shape, dtype=np.float16),
            'full_state': spaces.Box(low=-20, high=1, shape=(self.n_layers, self.X, self.Y), dtype=np.float16)
        })

    def get_observation(self):
        return {
            'local_obs': self.observation_state,
            'full_state': self.local_state
        }

    def step(self, action):
        # print(f"TaskAllocationAgent step called with action: {action}")
        waypoint = self.action_to_waypoint(action)
        if not self.path:
            self.path = self.compute_path(tuple(self.current_pos), waypoint)
        
        if self.path:
            next_pos = self.path.pop(0)
            discrete_action = self.determine_action(tuple(self.current_pos), next_pos)
            self.current_pos = super().step(discrete_action)
        
        return self.current_pos

    def compute_path(self, start, goal):
        return self.path_preprocessor.get_path(start, goal)

    def determine_action(self, current_pos, next_pos):
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        if dx == -1 and dy == 0:
            return 0  # Move left
        elif dx == 1 and dy == 0:
            return 1  # Move right
        elif dx == 0 and dy == 1:
            return 2  # Move up
        elif dx == 0 and dy == -1:
            return 3  # Move down
        elif dx == 1 and dy == 1:
            return 5  # Cross up right
        elif dx == -1 and dy == 1:
            return 6  # Cross up left
        elif dx == -1 and dy == -1:
            return 7  # Cross down left
        elif dx == 1 and dy == -1:
            return 8  # Cross down right
        else:
            return 4  # Stay (in case current_pos == next_pos)

    def action_to_waypoint(self, action):
        dx = (action % (2 * self.max_distance + 1)) - self.max_distance
        dy = (action // (2 * self.max_distance + 1)) - self.max_distance
        x = np.clip(self.current_pos[0] + dx, 0, self.xs - 1)
        y = np.clip(self.current_pos[1] + dy, 0, self.ys - 1)
        return (x, y)

    def get_valid_actions(self):
        valid_actions = []
        for action in range(self._action_space.n):
            waypoint = self.action_to_waypoint(action)
            if not self.inbuilding(*waypoint):
                valid_actions.append(action)
        return valid_actions

    def get_next_action(self):
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            # print("No valid actions available!")
            return self._action_space.sample()  # Return a random action if no valid actions

        action = self.randomiser.choice(valid_actions)
        waypoint = self.action_to_waypoint(action)
        # print(f"TaskAllocationAgent get_next_action returned: {action}, waypoint: {waypoint}")
        return action

    def inbuilding(self, x, y):
        return self.local_state[0][x, y] == 0

    def reset(self):
        super().reset()
        self.path = []
        return self.get_observation()