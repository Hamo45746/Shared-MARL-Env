import numpy as np
from gymnasium import spaces
import random
# Reference: This is largely copied from PettingZoos DiscreteAgent and Agent classes.

class Agent:
    def __new__(cls, *args, **kwargs):
        agent = super().__new__(cls)
        return agent

    @property
    def observation_space(self):
        raise NotImplementedError()

    @property
    def action_space(self):
        raise NotImplementedError()

    def __str__(self):
        return f"<{type(self).__name__} instance>"
    
class DiscreteAgent(Agent):
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
        # map_matrix is the map of the environment (!0 are buildings)
        # n channels is the number of observation channels

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
        
        # Initialise the local observation state
        self._obs_range = obs_range
        self.X, self.Y = self.map_matrix.shape
        self.observation_state = np.full((n_layers, obs_range, obs_range), fill_value=-20)
        self.local_state = np.full((n_layers, self.X, self.Y), fill_value=-20)
        
        if flatten:
            self._obs_shape = (n_layers * obs_range**2 + 1,)
        else:
            self._obs_shape = (obs_range, obs_range, 4)
            # self._obs_shape = (4, obs_range, obs_range)

    @property
    def observation_space(self):
        return spaces.Box(low=-20, high=1, shape=self._obs_shape)

    @property
    def action_space(self):
        return spaces.Discrete(9)

    # Dynamics Functions
    def step(self, a):
        cpos = self.current_pos
        lpos = self.last_pos
        # if dead or reached goal dont move
        if self.terminal:
            return cpos
        # if in building, dead, and stay there
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
            print("here3")
            return cpos
        # if bumped into building, then stay
        if self.inbuilding(x, y):
            print("here")
            return cpos
        #deleted else statement
        lpos[0] = cpos[0]
        lpos[1] = cpos[1]
        cpos[0] = x
        cpos[1] = y
        return cpos

    def get_state(self):
        return self.current_pos

    # Helper Functions
    def inbounds(self, x, y):
        if 0 <= x < self.xs and 0 <= y < self.ys:
            return True
        return False

    def inbuilding(self, x, y):
        if self.observation_state[0, x, y] == 0: # Maybe incorrect?
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
        """Update the agent's global representation of the environment state based on another agent's observations."""
        observer_x, observer_y = observer_position
        obs_half_range = self._obs_range // 2

        # Iterate through each layer in the observed_state
        for layer in range(observed_state.shape[0]):
            if layer == 0:  # Layer 0 (map matrix)
                # Directly assign the observed map matrix to the local state
                for dx in range(-obs_half_range, obs_half_range + 1):
                    for dy in range(-obs_half_range, obs_half_range + 1):
                        global_x = observer_x + dx
                        global_y = observer_y + dy
                        if self.inbounds(global_x, global_y):
                            obs_x = obs_half_range + dx
                            obs_y = obs_half_range + dy
                            self.local_state[layer, global_x, global_y] = observed_state[layer, obs_x, obs_y]
            else:
                # Update the remaining layers with decrement
                for dx in range(-obs_half_range, obs_half_range + 1):
                    for dy in range(-obs_half_range, obs_half_range + 1):
                        global_x = observer_x + dx
                        global_y = observer_y + dy
                        if self.inbounds(global_x, global_y):
                            obs_x = obs_half_range + dx
                            obs_y = obs_half_range + dy
                            if observed_state[layer, obs_x, obs_y] == 0:
                                self.local_state[layer, global_x, global_y] = 0
                            elif self.local_state[layer, global_x, global_y] > -20:
                                self.local_state[layer, global_x, global_y] = max(self.local_state[layer, global_x, global_y] - 1, -20)
                            
    def set_observation_state(self, observation):
        self.observation_state = observation
        
    def get_observation_state(self):
        return self.observation_state

    def get_next_action(self):
        random_actions = self.eactions
        temp = []
        for action in self.eactions:
                temp.append(action)
        a = random.choice(random_actions)
        x, y = self.step(a)
        while self.inbuilding(x,y) or not self.inbounds(x,y):
            self.current_pos = self.last_pos
            self.current_position()
            return 4
        # while self.inbuilding(x,y) or not self.inbounds(x,y):
        #     temp.remove(a)
        #     b = a
        #     a = random.choice(temp)
        #     self.current_pos = self.last_pos
        #     x, y = self.step(a)
        #     temp.append(b)
        if self.inbuilding(x,y):
            print("grrrrrr")
        return a