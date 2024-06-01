import numpy as np
from agent import DiscreteAgent
import heapq
import random

class Target(DiscreteAgent):
    def __init__(
        self, 
        xs, 
        ys, 
        map_matrix, 
        randomizer, 
        start_pos, 
        goal_pos,
        obs_range=0, 
        n_channels=3, 
        seed=1, 
        flatten=False
        ):
        super().__init__(xs, ys, map_matrix, randomizer, obs_range, n_channels, seed, flatten)
        self.goal_pos = np.array(goal_pos, dtype=np.int32)
        self.path = self.compute_path(start_pos, goal_pos)
        self.path_index = 0

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

    def heuristic(self, a, b): # manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]
            
    def determine_action(self, current_pos, next_pos):
        delta = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
        if delta == (-1, 0): # Move left
            return self.eactions[0]
        elif delta == (1, 0): # Move right
            return self.eactions[1]  
        elif delta == (0, 1): # Move up
            return self.eactions[2]  
        elif delta == (0, -1): # Move down
            return self.eactions[3]  
        return self.eactions[4]  # Stay (in case current_pos == next_pos)
    
    def get_next_action(self):
        # Calculate the next action from the path
        if self.path_index < len(self.path):
            current_pos = self.current_position()
            next_pos = self.path[self.path_index]
            self.path_index += 1
            #if self.path_index >= len(self.path):
                #self.path_index = 0  # Optionally loop the path
            return self.determine_action(current_pos, next_pos)
        return self.eactions[4]  # Default to 'stay' if path ended or not valid