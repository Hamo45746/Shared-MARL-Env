import heapq
from discrete_agent import DiscreteAgent
from gymnasium import spaces
import numpy as np

class TaskAllocationAgent(DiscreteAgent):
    def __init__(
        self,
        xs,
        ys,
        map_matrix,
        randomizer,
        obs_range=3,
        n_layers=4,
        seed=10,
        flatten=False,
        max_steps_per_action=5,
    ):
        super().__init__(xs, ys, map_matrix, randomizer, obs_range, n_layers, seed, flatten)
        self.max_steps_per_action = max_steps_per_action
        self.action_space = spaces.Discrete(xs * ys)  # Update the action space to the entire map matrix
        self.path = []
        self.path_index = 0
        self.steps_taken = 0

    def step(self, waypoint):
        if not self.path:
            self.path = self.compute_path(self.current_pos, waypoint)
            self.path_index = 0
            self.steps_taken = 0

        reached_waypoint = False
        while self.steps_taken < self.max_steps_per_action and self.path_index < len(self.path):
            next_pos = self.path[self.path_index]
            action = self.determine_action(self.current_pos, next_pos)
            self.current_pos = super().step(action)
            self.path_index += 1
            self.steps_taken += 1

            if np.array_equal(self.current_pos, waypoint):
                reached_waypoint = True
                break

        if reached_waypoint or self.steps_taken == self.max_steps_per_action:
            self.path = []
            self.path_index = 0
            self.steps_taken = 0

        return self.current_pos

    def compute_path(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)
        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for dx, dy in self.motion_range:
                neighbor = (current[0] + dx, current[1] + dy)
                if self.inbounds(*neighbor) and not self.inbuilding(*neighbor):
                    tentative_g_score = current_cost + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))
        return []

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def determine_action(self, current_pos, next_pos):
        delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
        if delta == (-1, 0):  # Move left
            return self.eactions[0]
        elif delta == (1, 0):  # Move right
            return self.eactions[1]
        elif delta == (0, 1):  # Move up
            return self.eactions[2]
        elif delta == (0, -1):  # Move down
            return self.eactions[3]
        elif delta == (1, 1):  # Cross up right
            return self.eactions[5]
        elif delta == (-1, 1):  # Cross up left
            return self.eactions[6]
        elif delta == (-1, -1):  # Cross down left
            return self.eactions[7]
        elif delta == (1, -1):  # Cross down right
            return self.eactions[8]
        return self.eactions[4]  # Stay (in case current_pos == next_pos)

    def inbuilding(self, x, y):
        if self.map_matrix[x, y] == 0:
            return True
        return False
    
