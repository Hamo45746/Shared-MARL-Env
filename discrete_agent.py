import numpy as np
import random
from base_agent import BaseAgent
from gymnasium import spaces

class DiscreteAgent(BaseAgent):
    # constructor
    def __init__(
        self,
        xs,
        ys,
        map_matrix,
        randomiser,
        obs_range=3,
        n_layers=4,
        seed=10,
        flatten=False,
    ):

        self.random_state = randomiser
        self.xs = xs
        self.ys = ys
        self.eactions = [
            0,  # move left
            1,  # move right
            2,  # move up
            3,  # move down 
            4,  # stay
            5, #cross up right
            6, #cross up left
            7, #cross down left
            8, #cross down right 
        ] 
        self.motion_range = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0], [1, 1], [-1, 1], [-1, -1], [1, -1]]
        self.current_pos = np.zeros(2, dtype=np.int32)  # x and y position
        self.last_pos = np.zeros(2, dtype=np.int32)
        self.temp_pos = np.zeros(2, dtype=np.int32)
        self.map_matrix = map_matrix
        self.terminal = False
        self._obs_range = obs_range # Initialise the local observation state
        self.X, self.Y = self.map_matrix.shape
        self.observation_state = np.full((n_layers, obs_range, obs_range), fill_value=-20)
        self.local_state = np.full((n_layers, self.X, self.Y), fill_value=-20, dtype=np.int32)
        
        if flatten:
            self._obs_shape = (n_layers * obs_range**2 + 1,)
        else:
            self._obs_shape = (obs_range, obs_range, n_layers)

    @property
    def observation_space(self):
        return spaces.Box(low=-20, high=1, shape=self._obs_shape, dtype=np.float32)

    @property
    def action_space(self):
        return spaces.Discrete(len(self.eactions))

    # Dynamics Functions
    def step(self, a):
        cpos = self.current_pos
        lpos = self.last_pos
        # if dead or reached goal dont move
        # if self.terminal:
        #     print('here1')
        #     return cpos
        # # if in building, dead, and stay there
        if self.inbuilding(cpos[0], cpos[1]):
            return cpos
        tpos = self.temp_pos
        tpos[0] = cpos[0]
        tpos[1] = cpos[1]
        # transition is deterministic
        tpos += self.motion_range[a]
        x = tpos[0]
        y = tpos[1]
        # check bounds
        if not self.inbounds(x, y):
            return cpos
        # if bumped into building, then stay
        if self.inbuilding(x, y):
            return cpos
        #deleted else statement
        lpos[0] = cpos[0]
        lpos[1] = cpos[1]
        cpos[0] = x
        cpos[1] = y
        self.current_pos = cpos
        self.last_pos = lpos
        return cpos

    def get_state(self):
        return self.local_state

    def get_observation_state(self):
        return self.observation_state

    # Helper Functions
    def inbounds(self, x, y):
        if 0 <= x < self.xs and 0 <= y < self.ys:
            return True
        return False
    
    def inbuilding(self, x, y):
        # if self.observation_state[0, x - self.current_pos[0], y - self.current_pos[1]] == 0: # Maybe incorrect?
        if self.map_matrix[x, y] == 0:
            return True
        return False

    def nactions(self):
        return len(self.eactions)

    def set_position(self, xs, ys):
        self.current_pos[0] = xs
        self.current_pos[1] = ys

    def current_position(self):
        return self.current_pos

    def last_position(self):
        return self.last_pos
    
    def update_local_state(self, observed_state, observer_position):
        observer_x, observer_y = observer_position
        obs_half_range = self._obs_range // 2

        for layer in range(observed_state.shape[0]):
            for dx in range(-obs_half_range, obs_half_range + 1):
                for dy in range(-obs_half_range, obs_half_range + 1):
                    global_x = observer_x + dx
                    global_y = observer_y + dy
                    obs_x = obs_half_range + dx
                    obs_y = obs_half_range + dy
                    
                    if (self.inbounds(global_x, global_y) and
                        0 <= obs_x < self._obs_range and
                        0 <= obs_y < self._obs_range):
                        
                        if layer == 0:  # Map layer
                            self.local_state[layer, global_x, global_y] = observed_state[layer, obs_x, obs_y]
                        else:
                            observed_value = observed_state[layer, obs_x, obs_y]
                            current_value = self.local_state[layer, global_x, global_y]
                            
                            if observed_value == 0:
                                self.local_state[layer, global_x, global_y] = 0
                            elif observed_value > current_value:
                                # Update only if the observed value is more recent
                                self.local_state[layer, global_x, global_y] = observed_value
                            
    def set_observation_state(self, observation):
        """Update the observation_state based on the input observation"""
        self.observation_state = observation.copy()
        # for layer in range(observation.shape[0]):
        #     if layer == 0:  # Layer 0 (map matrix) 
        #         # In map layer 0 is obstacle, 1 is empty space
        #         self.observation_state[layer] = observation[layer]
        #     else:
        #         for i in range(observation.shape[1]):
        #             for j in range(observation.shape[2]):
        #                 if observation[layer, i, j] == 0:
        #                     self.observation_state[layer, i, j] = 0
        #                 elif self.observation_state[layer, i, j] > -20:
        #                     self.observation_state[layer, i, j] -= 1

    def get_next_action(self):
        return random.choice(self.eactions)