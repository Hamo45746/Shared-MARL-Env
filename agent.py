import numpy as np
from gymnasium import spaces

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
        randomizer,
        obs_range=3,
        n_channels=4,
        seed=1,
        flatten=False,
    ):
        # map_matrix is the map of the environment (!0 are buildings)
        # n channels is the number of observation channels

        self.random_state = randomizer

        self.xs = xs
        self.ys = ys

        self.eactions = [
            0,  # move left
            1,  # move right
            2,  # move up
            3,  # move down
            4,  # stay
        ] 

        self.motion_range = [[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]]

        self.current_pos = np.zeros(2, dtype=np.int32)  # x and y position
        self.last_pos = np.zeros(2, dtype=np.int32)
        self.temp_pos = np.zeros(2, dtype=np.int32)

        self.map_matrix = map_matrix

        self.terminal = False
        
        # Initialize the local observation state
        self._obs_range = obs_range
        self.X, self.Y = self.map_matrix.shape
        self.observation_state = np.full((n_channels, obs_range, obs_range), fill_value=-np.inf)
        self.local_state = np.full((n_channels, self.X, self.Y), fill_value=-np.inf)
        
        if flatten:
            self._obs_shape = (n_channels * obs_range**2 + 1,)
        else:
            self._obs_shape = (obs_range, obs_range, 4)
            # self._obs_shape = (4, obs_range, obs_range)

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=self._obs_shape)

    @property
    def action_space(self):
        return spaces.Discrete(5)

    # Dynamics Functions
    def step(self, a):
        cpos = self.current_pos
        lpos = self.last_pos
        # if dead or reached goal dont move
        if self.terminal:
            return cpos
        # if in building, dead, and stay there
        if self.inbuilding(cpos[0], cpos[1]):
            self.terminal = True
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
        else:
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
        if self.map_matrix[x, y] != 0:
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
        """Update agents local representation of the environment state based on an allies observation."""
        observer_x, observer_y = observer_position
        agent_x, agent_y = self.current_position()

        # Calculate the relative offset from the observer to the agent
        rel_x = observer_x - agent_x + self._obs_range // 2
        rel_y = observer_y - agent_y + self._obs_range // 2

        # Update local state using the observed state from another agent
        for dx in range(-self._obs_range // 2, self._obs_range // 2 + 1):
            for dy in range(-self._obs_range // 2, self._obs_range // 2 + 1):
                global_x = rel_x + dx
                global_y = rel_y + dy
                # Update the local state at the observing agent’s location
                if 0 <= global_x < self._obs_range and 0 <= global_y < self._obs_range:
                    self.local_state[:, global_x, global_y] = observed_state[:, self._obs_range//2 + dx, self._obs_range//2 + dy]
                    
    def set_observation_state(self, observation):
        self.observation_state = observation
