import numpy as np
from heapq import heappush, heappop

class PathProcessor:
    def __init__(self, map_matrix, width, height):
        self.map_matrix = map_matrix
        self.height = height
        self.width = width
        self.path_cache = {}
        # print(f"PathProcessor initialized with map shape: {self.height}x{self.width}")


    def get_path(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        
        if start == goal:
            return [start]

        if not self.is_valid_position(start) or not self.is_valid_position(goal):
            return []

        path = self._cache_aware_a_star(start, goal)
        if path:
            self._cache_subpaths(path)
      
        return path
    
    def is_valid_position(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height and self.map_matrix[x, y] != 0

    def _cache_aware_a_star(self, start, goal):
        # print(f"Running A* from {start} to {goal}")
        heap = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while heap:
            current_f, current = heappop(heap)
            
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                # print(f"Path found with length: {len(path)}")
                return path
            
            for neighbor in self._get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    heappush(heap, (f_score[neighbor], neighbor))
        
        # print("No path found")
        return []

    def _get_neighbors(self, pos):
        x, y = pos
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [n for n in neighbors if self.is_valid_position(n)]

    def _heuristic(self, a, b):
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))  # Chebyshev distance for 8-directional movement

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def _get_longest_cached_subpath(self, start, goal):
        longest_path = None
        longest_length = 0
        for cached_start, cached_end in self.path_cache.keys():
            if cached_start == start and self._heuristic(cached_end, goal) < self._heuristic(start, goal):
                path = self.path_cache[(cached_start, cached_end)]
                if len(path) > longest_length:
                    longest_path = path
                    longest_length = len(path)
        return longest_path

    def _cache_subpaths(self, path):
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                subpath = path[i:j+1]
                cache_key = (subpath[0], subpath[-1])
                if cache_key not in self.path_cache:
                    self.path_cache[cache_key] = subpath

    def clear_cache(self):
        self.path_cache.clear()