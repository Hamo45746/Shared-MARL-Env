import numpy as np
from heapq import heappush, heappop
from collections import OrderedDict

class PathProcessor:
    def __init__(self, map_matrix, width, height, max_cache_size=5000, min_cache_length=3):
        self.map_matrix = map_matrix
        self.height = height
        self.width = width
        self.max_cache_size = max_cache_size
        self.min_cache_length = min_cache_length
        self.path_cache = OrderedDict()

    def get_path(self, start, goal):
        start, goal = tuple(start), tuple(goal)
        
        if start == goal:
            return [start]

        if not self.is_valid_position(start) or not self.is_valid_position(goal):
            return []

        cache_key = (start, goal)
        if cache_key in self.path_cache:
            self.path_cache.move_to_end(cache_key)  # Move to the end (most recently used)
            return self.path_cache[cache_key]

        path = self._cache_aware_a_star(start, goal)
        if path:
            self._cache_subpaths(path)
      
        return path

    def _cache_subpaths(self, path):
        path_length = len(path)
        for i in range(path_length):
            for j in range(i + self.min_cache_length, path_length + 1):
                subpath = path[i:j]
                if len(subpath) > self.min_cache_length:
                    cache_key = (subpath[0], subpath[-1])
                    if cache_key not in self.path_cache:
                        if len(self.path_cache) >= self.max_cache_size:
                            self.path_cache.popitem(last=False)  # Remove the oldest item
                        self.path_cache[cache_key] = subpath
                    self.path_cache.move_to_end(cache_key)  # Move to the end (most recently used)

    def _cache_aware_a_star(self, start, goal):
        heap = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while heap:
            current_f, current = heappop(heap)
            
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return path
            
            # Check if there's a cached path from current to goal
            cached_path = self._get_longest_cached_subpath(current, goal)
            if cached_path:
                for i in range(1, len(cached_path)):
                    came_from[cached_path[i]] = cached_path[i-1]
                return self._reconstruct_path(came_from, goal)
            
            for neighbor in self._get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    heappush(heap, (f_score[neighbor], neighbor))
        
        return []

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

    def is_valid_position(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height and self.map_matrix[x, y] != 0

    def clear_cache(self):
        self.path_cache.clear()