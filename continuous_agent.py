import numpy as np
from gymnasium import spaces
from base_agent import BaseAgent

class ContinuousAgent(BaseAgent):
    def __init__(self, xs, ys, map_matrix, randomiser, obs_range=3, n_layers=4, seed=10, flatten=False):
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
        self.observation_state = {
            "map": np.full((n_layers, obs_range, obs_range), fill_value=-20),
            "velocity": np.zeros(2, dtype=np.float32),
            "goal": np.zeros(2, dtype=np.float32)
        }
        self.local_state = np.full((n_layers, self.X, self.Y), fill_value=-20)
        self._obs_shape = (n_layers, obs_range, obs_range)  # Update observation shape to include velocity and position
        self.observed_areas = set()
        self.path = []
        self.communicated = False
        self.initial_position = None
        self.change_angle = False
        self.goal_area = None
        self.previous_distance_to_goal = None

    @property
    def observation_space(self):
        return spaces.Dict({
            "map": spaces.Box(low=-20, high=1, shape=self._obs_shape, dtype=np.float32),
            "velocity": spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
            "goal": spaces.Box(low=-2000, high=2000, shape=(2,), dtype=np.float32),
        })
    @property
    def action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def step(self, action):
        # Convert action to acceleration (assuming action is in range [-1, 1] and maps to [-2, 2] km/h)
        acceleration = action #acceleration = action * 2.0
        # Adjust velocity
        self.velocity += acceleration
        # Clamp velocity to the desired range
        self.velocity = np.clip(self.velocity, -2.0, 2.0)  # Adjust as per your requirements

        # Determine the new direction based on the constraints
        # Determine the number of sub-steps based on the current velocity
        speed = np.linalg.norm(self.velocity)
        num_sub_steps = max(1, int((speed // 2) * 2))

        sub_step_velocity = self.velocity / num_sub_steps

        valid_move = True

        for _ in range(num_sub_steps):
            # Update position based on sub-step velocity
            self.temp_pos = self.current_pos + sub_step_velocity

            # Check bounds and obstacles for each sub-step
            if self.inbounds(self.temp_pos[0], self.temp_pos[1]) and not self.inbuilding(self.temp_pos[0], self.temp_pos[1]):
                self.last_pos[:] = self.current_pos
                self.current_pos[:] = self.temp_pos
                self.path.append((self.current_pos[0], self.current_pos[1]))
            else:
                valid_move = False
                break  # Exit the loop if an invalid move is detected

        # If the final position after all sub-steps is invalid, reset to the last valid position
        if not valid_move:
            self.current_pos[:] = self.last_pos

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
        observed_map = observed_state["map"]

        for layer in range(observed_map.shape[0]):
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
                            self.local_state[layer, global_x1, global_y1] = observed_map[layer, obs_x, obs_y]
                        else:
                            if observed_map[layer, obs_x, obs_y] == 0:
                                self.local_state[layer, global_x1, global_y1] = 0
                            elif self.local_state[layer, global_x1, global_y1] > -20:
                                self.local_state[layer, global_x1, global_y1] -= 1

    def get_next_action(self):
        action = self.random_state.uniform(-1, 1, size=(2,))
        return action
    
    def set_observation_state(self, observation):
        self.observation_state = observation

    def set_goal_area(self, goal_area):
        self.goal_area = goal_area
        self.previous_distance_to_goal = self.calculate_distance_to_goal()

    def calculate_distance_to_goal(self):
        if self.goal_area is not None:
            return np.linalg.norm(self.current_pos - self.goal_area)
        return None

    def gains_information(self):
        new_information_count = 0
        total_cells = self.observation_state["map"].shape[1] * self.observation_state["map"].shape[2]

        for x in range(self.observation_state["map"].shape[1]):
            for y in range(self.observation_state["map"].shape[2]):
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
        threshold = 5
        x, y = self.current_pos
        for dx in range(-threshold, threshold + 1):
            for dy in range(-threshold, threshold + 1):
                nx, ny = int(x + dx), int(y + dy)
                if self.inbounds(nx, ny) and self.inbuilding(nx, ny):
                    return True
        return False

    def angle_change(self):
        if self.change_angle:
            self.change_angle = False
            return True
        return False
