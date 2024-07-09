import numpy as np
from gymnasium import spaces
from base_agent import BaseAgent

class ContinuousAgent(BaseAgent):
    def __init__(self, xs, ys, map_matrix, randomiser, obs_range=3, n_layers=5, seed=10, flatten=False):
        self.random_state = randomiser
        self.xs = xs
        self.ys = ys
        self.current_pos = np.zeros(2, dtype=np.float32)
        self.last_pos = np.zeros(2, dtype=np.float32)
        self.temp_pos = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)  # Add velocity
        self.map_matrix = map_matrix
        self.terminal = False
        self._obs_range = obs_range
        self.X, self.Y = self.map_matrix.shape
        self.observation_state = np.full((n_layers, obs_range, obs_range), fill_value=-20)
        self.local_state = np.full((n_layers, self.X, self.Y), fill_value=-20)
        self._obs_shape = (n_layers, obs_range, obs_range)  # Update observation shape to include velocity and position
        self.observed_areas = set()
        self.path = []
        self.communicated = False
        self.initial_position = None

    @property
    def observation_space(self):
        return spaces.Box(low=-20, high=1, shape=self._obs_shape, dtype=np.float32)

    @property
    def action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def step(self, action):
        # Convert action to acceleration (assuming action is in range [-1, 1] and maps to [-2, 2] km/h)
        acceleration = action #acceleration = action * 2.0
        # Adjust velocity
        self.velocity += acceleration
        # Clamp velocity to the desired range
        self.velocity = np.clip(self.velocity, -1.0, 1.0)  # Adjust as per your requirements

        # Determine the new direction based on the constraints
        if np.linalg.norm(self.velocity) > 10.0:
            current_direction = np.arctan2(self.velocity[1], self.velocity[0])
            max_angle_change = np.deg2rad(75)
            desired_direction = np.arctan2(acceleration[1], acceleration[0])
            angle_diff = desired_direction - current_direction
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # Normalize angle to [-pi, pi]

            if np.abs(angle_diff) > max_angle_change:
                angle_diff = np.sign(angle_diff) * max_angle_change

            new_direction = current_direction + angle_diff
            speed = np.linalg.norm(self.velocity)
            self.velocity = np.array([speed * np.cos(new_direction), speed * np.sin(new_direction)])

        # Update position based on velocity
        self.temp_pos = self.current_pos + self.velocity

        if self.inbounds(self.temp_pos[0], self.temp_pos[1]) and not self.inbuilding(self.temp_pos[0], self.temp_pos[1]):
            self.last_pos[:] = self.current_pos
            self.current_pos[:] = self.temp_pos
            self.path.append((self.current_pos[0], self.current_pos[1]))

        return self.current_pos

    def get_state(self):
        return np.concatenate([self.velocity, self.current_pos])
    
    def get_observation_state(self):
        return self.observation_state

    def inbounds(self, x, y):
        return 0 <= x < self.xs and 0 <= y < self.ys

    def inbuilding(self, x, y):
        return self.map_matrix[int(x), int(y)] == 0

    def set_position(self, x, y):
        self.current_pos[:] = x, y
        if self.initial_position is None:
            self.initial_position = np.array([x, y], dtype=np.float32)

    def current_position(self):
        return self.current_pos

    def update_local_state(self, observed_state, observer_position):
        observer_x, observer_y = observer_position
        obs_half_range = self._obs_range // 2
        for layer in range(observed_state.shape[0]):
            for dx in range(-obs_half_range, obs_half_range + 1):
                for dy in range(-obs_half_range, obs_half_range + 1):
                    global_x = observer_x + dx
                    global_y = observer_y + dy
                    global_x1 = int(global_x)
                    global_y1 = int(global_y)
                    obs_x = obs_half_range + dx
                    obs_y = obs_half_range + dy
                    if self.inbounds(global_x, global_y):
                        if layer == 0:
                            self.local_state[layer, global_x1, global_y1] = observed_state[layer, obs_x, obs_y]
                        else:
                            if observed_state[layer, obs_x, obs_y] == 0:
                                self.local_state[layer, global_x1, global_y1] = 0
                            elif self.local_state[layer, global_x1, global_y1] > -20:
                                self.local_state[layer, global_x1, global_y1] -= 1

    def get_next_action(self):
        action = self.random_state.uniform(-1.0, 1.0, size=(2,))
        return action
    
    def set_observation_state(self, observation):
        self.observation_state = observation

    def gains_information(self):
        new_information_count = 0
        total_cells = self.observation_state.shape[1] * self.observation_state.shape[2]

        for x in range(self.observation_state.shape[1]):
            for y in range(self.observation_state.shape[2]):
                pos = (int(self.current_pos[0] - self._obs_range // 2 + x), int(self.current_pos[1] - self._obs_range // 2 + y))
                if pos not in self.observed_areas:
                    self.observed_areas.add(pos)
                    new_information_count += 1

        percentage_new_information = (new_information_count / total_cells) * 100
        return percentage_new_information
    
    def communicates_information(self):
        if self.communicated:
            self.communicated = False
            return True
        return False
    
    def calls_obstacle_avoidance(self):
        threshold = 2
        x, y = self.current_pos
        for dx in range(-threshold, threshold + 1):
            for dy in range(-threshold, threshold + 1):
                nx, ny = int(x + dx), int(y + dy)
                if self.inbounds(nx, ny) and self.inbuilding(nx, ny):
                    return True
        return False

