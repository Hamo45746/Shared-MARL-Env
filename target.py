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
        self.path = self.compute_path(start_pos, self.current_goal)
        self.path_index = 0

    def compute_path(self, start, goal):
        return self.path_processor.get_path(start, goal)

    def select_new_goal(self):
        while True:
            x = self.randomiser.integers(0, self.xs)
            y = self.randomiser.integers(0, self.ys)
            if not self.inbuilding(x, y):
                return (x, y)

    def get_next_action(self):
        if self.path_index >= len(self.path) - 1:
            # We've reached the current goal, select a new one
            current_pos = self.current_position()
            self.current_goal = self.select_new_goal()
            self.path = self.compute_path(current_pos, self.current_goal)
            self.path_index = 0

        if self.path_index < len(self.path):
            current_pos = self.current_position()
            next_pos = self.path[self.path_index]
            self.path_index += 1
            return self.determine_action(current_pos, next_pos)
        
        return self.eactions[4]  # Default to 'stay' if path ended or not valid

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
        return self.map_matrix[x, y] == 0