import numpy as np
from gymnasium import spaces
# from gym import spaces
from base_agent import BaseAgent

class ContinuousAgent(BaseAgent):
    def __init__(self, xs, ys, map_matrix, randomiser, obs_range=3, n_layers=4, seed=10, flatten=False):
        self.random_state = randomiser
        self.xs = xs
        self.ys = ys
        self.current_pos = np.zeros(2, dtype=np.float32)
        self.last_pos = np.zeros(2, dtype=np.float32)
        self.temp_pos = np.zeros(2, dtype=np.float32)
        self.map_matrix = map_matrix
        self.terminal = False
        self._obs_range = obs_range
        self.X, self.Y = self.map_matrix.shape
        self.observation_state = np.full((n_layers, obs_range, obs_range), fill_value=-20)
        self.local_state = np.full((n_layers, self.X, self.Y), fill_value=-20)
        self._obs_shape = (n_layers * obs_range**2 + 1,) if flatten else (obs_range, obs_range, n_layers)
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
        cpos = self.current_pos
        lpos = self.last_pos
        if self.inbuilding(cpos[0], cpos[1]):
            return cpos
        tpos = self.temp_pos
        tpos[0] = cpos[0]
        tpos[1] = cpos[1]
        tpos += action
        x, y = tpos
        if not self.inbounds(x, y):
            return cpos
        if self.inbuilding(x,y):
            lpos[:] = cpos
            return cpos
        lpos[:] = cpos
        cpos[:] = tpos
        self.path.append((cpos[0], cpos[1]))
        return cpos

    def get_state(self):
        return self.current_pos
    
    def get_observation_state(self):
        return self.observation_state

    def inbounds(self, x, y):
        return 0 <= x < self.xs and 0 <= y < self.ys

    def inbuilding(self, x, y):
        return self.map_matrix[int(x), int(y)] == 0

    def set_position(self, x, y):
        self.current_pos[:] = x, y
        if self.initial_position is None:
            self.initial_position = np.array([x,y], dtype = np.float32)

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
                                # self.local_state[layer, global_x, global_y] = max(self.local_state[layer, global_x, global_y] - 1, -20) 

    def get_observation_state(self):
        return self.observation_state
    
    def set_observation_state(self, observation):
        self.observation_state = observation
    
    def get_next_action(self):
        min_radius = 1.0
        while True:
            action = self.random_state.uniform(-1.0, 1.0, size=(2,))
            #print(action)
            if np.linalg.norm(action) >= min_radius:
                break
        return action

    # #this is for lunch and learn - sending agents away from origin
    # def get_next_action(self): 
    #     potential_actions = self.random_state.uniform(-1.0, 1.0, size=(100,2)) 
    #     max_distance = -1 
    #     best_action = None 
    #     print(self.origin)
    #     for action in potential_actions: 
    #         potential_position = self.current_pos + action 
    #         if self.inbounds(potential_position[0], potential_position[1]) and not self.inbuilding(potential_position[0], potential_position[1]):
    #             distance = np.linalg.norm(potential_position - self.initial_position) 
    #             if distance > max_distance: 
    #                 max_distance = distance 
    #                 best_action = action 
    #     # If all potential actions are invalid 
    #     if best_action is None: 
    #         best_action = self.random_state.uniform(-1.0, 1.0, size=(2,)) 
    #         print("here")
    #     return best_action
    
    def gains_information(self):
        new_information_count = 0
        total_cells = self.observation_state.shape[1] * self.observation_state.shape[2]

        # Iterate over the observation space
        for x in range(self.observation_state.shape[1]):
            for y in range(self.observation_state.shape[2]):
                pos = (int(self.current_pos[0] - self._obs_range // 2 + x), int(self.current_pos[1] - self._obs_range // 2 + y))
                if pos not in self.observed_areas:
                    self.observed_areas.add(pos)
                    new_information_count += 1

        # Calculate the percentage of new information
        percentage_new_information = (new_information_count / total_cells) * 100
        #print(f"Agent at {self.current_position()} gained {percentage_new_information}% new information.")
        return percentage_new_information
    
    def communicates_information(self):
        # Logic to check if the agent successfully shares new information with another agent
        if self.communicated:
            self.communicated = False
            return True
        return False
    
    def calls_obstacle_avoidance(self):
        # Define the obstacle avoidance threshold
        threshold = 2
        x, y = self.current_pos
        for dx in range(-threshold, threshold + 1):
            for dy in range(-threshold, threshold + 1):
                nx, ny = int(x + dx), int(y + dy)
                if self.inbounds(nx, ny) and self.inbuilding(nx, ny):
                    return True
        return False