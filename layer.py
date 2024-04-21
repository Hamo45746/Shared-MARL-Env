import numpy as np

class AgentLayer:
    def __init__(self, xs, ys, allies, seed=1):
        """Initializes the AgentLayer class.

        xs: x size of map
        ys: y size of map
        allies: list of ally agents
        seed: seed

        Each ally agent must support:
        - move(action)
        - current_position()
        - nactions()
        - set_position(x, y)
        
        REF: PettingZoo Pursuit by Farama Foundation
        """
        self.allies = allies
        self.nagents = len(allies)
        self.layer_state = np.zeros((xs, ys), dtype=np.int32)

    def n_agents(self):
        return self.nagents

    def move_agent(self, agent_idx, action):
        return self.allies[agent_idx].step(action)

    def set_position(self, agent_idx, x, y):
        self.allies[agent_idx].set_position(x, y)

    def get_position(self, agent_idx):
        """Returns the position of the given agent."""
        return self.allies[agent_idx].current_position()

    def get_nactions(self, agent_idx):
        return self.allies[agent_idx].nactions()

    def remove_agent(self, agent_idx):
        # idx is between zero and nagents
        self.allies.pop(agent_idx)
        self.nagents -= 1

    def get_state_matrix(self):
        """Returns a matrix representing the positions of all allies.

        Example: matrix contains the number of allies at give (x,y) position
        0 0 0 1 0 0 0
        0 2 0 2 0 0 0
        0 0 0 0 0 0 1
        1 0 0 0 0 0 5
        TODO: FIX THIS
        """
        gs = self.layer_state
        gs.fill(0)
        for ally in self.allies:
            x, y = ally.current_position()
            gs[x, y] += 1
        return gs

    def get_state(self):
        pos = np.zeros(2 * len(self.allies))
        idx = 0
        for ally in self.allies:
            pos[idx : (idx + 2)] = ally.get_state()
            idx += 2
        return pos
    
    
class JammerLayer(AgentLayer):
    def __init__(self, xs, ys, jammers, activation_times=None, seed=1):
        super().__init__(xs, ys, jammers, seed)
        self.activation_times = activation_times or [0] * len(jammers)  # Default to immediate activation

    def activate_jammers(self, current_time):
        """Activate jammers based on the current time and their respective activation times."""
        for i, jammer in enumerate(self.jammers):
            if current_time >= self.activation_times[i] and jammer.active == 0:
                self.activate_jammer(jammer)

    def activate_jammer(self, jammer):
        """Activates the given jammer."""
        jammer.active = 1

    def set_position(self, jammer_idx, x, y):
        self.jammers[jammer_idx].set_position(x, y)
        self.update_layer_state()

    def update_layer_state(self):
        """Updates the layer state matrix with current positions of all jammers."""
        self.layer_state.fill(0)
        for jammer in self.jammers:
            x, y = jammer.position
            if jammer.active == 1:
                self.layer_state[x, y] += 1

    def get_state_matrix(self):
        """Returns a matrix representing the positions of active jammers."""
        self.update_layer_state()
        return self.layer_state
    
class TargetLayer:
    def __init__(self, targets, map_matrix):
        self.targets = targets
        self.layer_state = map_matrix

    def update(self, a_star_search_func=None):
        for target in self.targets:
            if target.policy == 'a_star':
                if not hasattr(target, 'path') or not target.path:
                    start = target.current_position()
                    target.path = a_star_search_func(start, target.goal, self.layer_state)
                # Move along the calculated path
                if target.path:
                    next_pos = target.path.pop(0)
                    if self.layer_state.is_valid_position(*next_pos):
                        target.set_position(*next_pos)
            elif target.policy == 'random':
                self.move_randomly(target)

    def move_randomly(self, target):
        x, y = target.current_position()
        possible_moves = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx != 0 or dy != 0]
        valid_moves = [move for move in possible_moves if self.layer_state.is_valid_position(*move)]
        if valid_moves:
            target.set_position(*valid_moves[np.random.randint(len(valid_moves))])

