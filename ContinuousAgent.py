import numpy as np
from gymnasium import spaces
from BaseAgent import BaseAgent

class ContinuousAgent(BaseAgent):
    def __init__(self, xs, ys, map_matrix, randomizer, obs_range=3, n_layers=4, seed=10, flatten=False):
        self.random_state = randomizer
        self.xs = xs
        self.ys = ys
        self.current_pos = np.zeros(2, dtype=np.float32)
        self.last_pos = np.zeros(2, dtype=np.float32)
        self.temp_pos = np.zeros(2, dtype=np.float32)
        self.map_matrix = map_matrix
        self.terminal = False
        self._obs_range = obs_range
        self.X, self.Y = self.map_matrix.shape
        self.observation_state = np.full((n_layers, obs_range, obs_range), fill_value=-np.inf)
        self.local_state = np.full((n_layers, self.X, self.Y), fill_value=-np.inf)
        self._obs_shape = (n_layers * obs_range**2 + 1,) if flatten else (obs_range, obs_range, n_layers)

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=self._obs_shape, dtype=np.float32)

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
        if not self.inbounds(x, y) or self.inbuilding(x, y):
            return cpos
        lpos[:] = cpos
        cpos[:] = tpos
        return cpos

    def get_state(self):
        return self.current_pos

    def inbounds(self, x, y):
        return 0 <= x < self.xs and 0 <= y < self.ys

    def inbuilding(self, x, y):
        return self.map_matrix[int(x), int(y)] == 0

    def set_position(self, x, y):
        self.current_pos[:] = x, y

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
                    if self.inbounds(global_x, global_y):
                        obs_x = obs_half_range + dx
                        obs_y = obs_half_range + dy
                        self.local_state[layer, global_x, global_y] = observed_state[layer, obs_x, obs_y]

    def set_observation_state(self, observation):
        self.observation_state = observation

    def get_next_action(self):
        action = self.random_state.uniform(-1.0, 1.0, size=(2,))
        return action