import numpy as np
from discrete_agent import DiscreteAgent

class Target(DiscreteAgent):
    def __init__(
        self, 
        xs, 
        ys, 
        map_matrix, 
        randomiser,
        path_processor,
        start_pos, 
        obs_range=0, 
        n_channels=3, 
        seed=1, 
        flatten=False
        ):
        super().__init__(xs, ys, map_matrix, randomiser, obs_range, n_channels, seed, flatten)
        self.path_processor = path_processor
        self.randomiser = randomiser
        self.current_goal = self.select_new_goal()
        self.target_path = self.compute_path(start_pos, self.current_goal)
        self.path_index = 0
        self.debug_counter = 0

    def compute_path(self, start, goal):
        path = self.path_processor.get_path(start, goal)
        while not path:
            # If no path is found, select a new goal
            self.current_goal = self.select_new_goal()
            path = self.path_processor.get_path(start, self.current_goal)
        return path

    def select_new_goal(self):
        attempts = 0
        while attempts < 100:  # Limit attempts to prevent infinite loop
            x = self.randomiser.integers(0, self.xs)
            y = self.randomiser.integers(0, self.ys)
            if not self.inbuilding(x, y):
                return (x, y)
            attempts += 1
        # If no valid position found, return current position
        return self.current_position()

    def get_next_action(self):
        current_pos = self.current_position()
        
        if np.array_equal(current_pos, self.current_goal) or self.path_index >= len(self.target_path) - 1 or not self.target_path:
            self.current_goal = self.select_new_goal()
            self.target_path = self.compute_path(current_pos, self.current_goal)
            self.path_index = 0

        if self.target_path and self.path_index < len(self.target_path):
            next_pos = self.target_path[self.path_index]
            action = self.determine_action(current_pos, next_pos)
            self.path_index += 1
            return action
        
        return self.eactions[4]  # Default to 'stay' if path ended or not valid
    
    def step(self, action):
        old_pos = self.current_position()
        new_pos = super().step(action)
        
        return new_pos

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

    def inbuilding(self, x, y):
        return self.full_state[0][x, y] == 0