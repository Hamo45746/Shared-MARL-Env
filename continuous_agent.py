import numpy as np
from gymnasium import spaces
from base_agent import BaseAgent

class ContinuousAgent(BaseAgent):
    def __init__(self, xs, ys, map_matrix, randomiser, real_world_pixel_scale, obs_range=3, n_layers=4, seed=10, flatten=False):
        self.random_state = randomiser
        self.xs = xs
        self.ys = ys
        self.current_pos = np.zeros(2, dtype=np.float32)
        self.last_pos = np.zeros(2, dtype=np.float32)
        self.temp_pos = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)  # Add velocity
        self.real_velocity = np.zeros(2, dtype=np.float32)
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
        self.goal_step_counter = 0
        self.valid_move = True
        self.velocity_cap = False
        self.stuck_steps = 0
        self.max_stuck_steps = 20
        self.communication_timer = 30
        self.real_world_pixel_scale = real_world_pixel_scale
        self.max_velocity = 10.0
        self.total_invalid_moves = 0

    @property
    def observation_space(self):
        return spaces.Dict({
            "map": spaces.Box(low=-20, high=1, shape=self._obs_shape, dtype=np.float32),
            "velocity": spaces.Box(low=-16.0, high=16.0, shape=(2,), dtype=np.float32),
            "goal": spaces.Box(low=-2000, high=2000, shape=(2,), dtype=np.float32),
        })
    @property
    def action_space(self):
        return spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)

    def step(self, action):
        acceleration = action #acceleration = action * 2.0
        # Adjust velocity
        proposed_velocity = self.real_velocity + acceleration

        if np.linalg.norm(proposed_velocity) > self.max_velocity:
            proposed_velocity = (proposed_velocity / np.linalg.norm(proposed_velocity)) * self.max_velocity
            self.velocity_cap = True
        else:
            self.velocity_cap = False

        self.real_velocity = proposed_velocity 

        real_displacement = self.real_velocity + (0.5*acceleration)

        self.velocity = self.real_velocity / self.real_world_pixel_scale
        displacement = real_displacement / self.real_world_pixel_scale
        # Determine the new direction based on the constraints
        # Determine the number of sub-steps based on the current velocity
        speed = np.linalg.norm(self.velocity)
        num_sub_steps = max(1, int((speed // 2) * 2))
        sub_step_displacement = displacement / num_sub_steps
        self.valid_move = True

        # Check all sub-steps before making the move
        for _ in range(num_sub_steps):
            self.temp_pos = self.current_pos + sub_step_displacement

            if not (self.inbounds(self.temp_pos[0], self.temp_pos[1]) and not self.inbuilding(self.temp_pos[0], self.temp_pos[1])):
                self.valid_move = False
                break

        # If move is invalid, revert and penalize or inform the agent
        if not self.valid_move:
            #print("invalid")
            self.stuck_steps += 1
            self.total_invalid_moves +=1
            # The following check is just for the very first move, it's so that the last_pos isn't (0,0)
            lx, ly = self.last_pos
            if lx == 0: 
                self.current_pos[:] = self.current_pos
            else:
                if self.stuck_steps > self.max_stuck_steps:
                    #print(f"Agent stuck for {self.stuck_steps} steps. Resetting velocity.")
                    self.velocity = np.zeros(2)
                    self.real_velocity = np.zeros(2)
                    self.stuck_steps = 0 
                    #print("let free")
                return self.current_pos
        else:
            self.stuck_steps = 0


        # Update only if all sub-steps are valid
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

    # # #this is to share full_state
    # def update_local_state(self, observed_state, observer_position):
    #     observer_x, observer_y = observer_position
    #     for layer in range(observed_state.shape[0]):
    #         for x in range(self.X):
    #             for y in range(self.Y):
    #                 # Check if the current position is within bounds
    #                 if self.inbounds(x, y):
    #                     # If it's the map layer (layer 0)
    #                     if layer == 0:
    #                         # Update only if the local state is unknown (-20) and the observed state is known (not -20)
    #                         if self.local_state[layer, x, y] == -20 and observed_state[layer, x, y] != 20:
    #                             self.local_state[layer, x, y] = observed_state[layer, x, y]
    #                             # Add the coordinate to the observed areas
    #                             self.observed_areas.add((x, y))
    #                     else:
    #                         observed_value = observed_state[layer, x, y]
    #                         current_value = self.local_state[layer, x, y]
    #                         # Update if observed value is newer (greater) or if it's the most recent information (0)
    #                         if observed_value != -20 and (observed_value > current_value or observed_value == 0):
    #                             self.local_state[layer, x, y] = observed_value

    def update_local_state(self, observed_state, observer_position):
        """
        Updates the agent's local state based on observed data from another agent.
        Handles map, jammer, agent, and target layers differently based on sharing strategy.
        """
        observer_x, observer_y = observer_position

        for layer in range(observed_state.shape[0]):
            # Full state sharing for the map layer
            if layer == 0:
                for x in range(self.X):
                    for y in range(self.Y):
                        if self.inbounds(x, y):
                            if self.local_state[layer, x, y] == -20 and observed_state[layer, x, y] != -20:
                                self.local_state[layer, x, y] = observed_state[layer, x, y]
                                # Add the coordinate to the observed areas - This is for reward 
                                self.observed_areas.add((x, y))

            # Share only observation space *2 for agent and target and jammer layers
            else:
                obs_half_range = self._obs_range 
                for dx in range(-obs_half_range, obs_half_range + 1):
                    for dy in range(-obs_half_range, obs_half_range + 1):
                        global_x = observer_x + dx
                        global_y = observer_y + dy
                        global_x = int(global_x)
                        global_y = int(global_y)
                        obs_x = obs_half_range + dx
                        obs_y = obs_half_range + dy

                        if self.inbounds(global_x, global_y):
                            observed_value = observed_state[layer, obs_x, obs_y]
                            current_value = self.local_state[layer, global_x, global_y]

                            if observed_value != -20 and (observed_value > current_value or observed_value == 0):
                                self.local_state[layer, global_x, global_y] = observed_value

    def get_next_action(self):
        action = self.random_state.uniform(-5, 5, size=(2,))
        return action
    
    def set_observation_state(self, observation):
        self.observation_state = observation

    def set_goal_area(self, goal_area):
        self.goal_area = goal_area
        self.previous_distance_to_goal = self.calculate_distance_to_goal()
        self.goal_step_counter = 0

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
                if pos not in self.observed_areas and self.inbounds(pos[0], pos[1]) and not self.inbuilding(pos[0], pos[1]):
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
        threshold = 4
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
