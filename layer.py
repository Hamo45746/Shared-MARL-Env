import numpy as np
import random

class AgentLayer:
    def __init__(self, xs, ys, agents, seed=1):
        """Initializes the AgentLayer class.

        xs: x size of map
        ys: y size of map
        agents: list of agents
        seed: seed

        Each ally agent must support:
        - move(action)
        - current_position()
        - nactions()
        - set_position(x, y)
        
        REF: PettingZoo Pursuit by Farama Foundation
        """
        self.agents = agents
        self.nagents = len(agents)
        self.layer_state = np.full((xs, ys), -np.inf)

    def n_agents(self):
        return self.nagents

    def move_agent(self, agent_idx, action):
        """Moves the agent according to the defined action and updates the layer state.
           This is where the policies should come in, providing the action for each agent."""
        o_pos = self.agents[agent_idx].current_position()
        n_pos = self.agents[agent_idx].step(action)
        # Update the layer state for old and new positions
        self.update_positions(o_pos, n_pos)
        return n_pos

    def update_positions(self, old_position, new_position):
        """Clear the old position and set the new position in the layer state."""
        ox, oy = old_position
        nx, ny = new_position
        if self.layer_state[ox, oy] == 0:  # Only reset if it was the current position of the agent
            self.layer_state[ox, oy] = -1  # Start decay from -1
        self.layer_state[nx, ny] = 0  # Refresh the new position to 0

    def set_position(self, agent_idx, x, y):
        self.agents[agent_idx].set_position(x, y)

    def get_position(self, agent_idx):
        """Returns the position of the given agent."""
        return self.agents[agent_idx].current_position()

    def get_nactions(self, agent_idx):
        return self.agents[agent_idx].nactions()

    def remove_agent(self, agent_idx):
        """Removes an agent from the layer and updates the state."""
        if 0 <= agent_idx < self.nagents:
            pos = self.agents[agent_idx].current_position()
            self.agents.pop(agent_idx)
            self.nagents -= 1
            self.layer_state[pos[0], pos[1]] = -np.inf  # Clear the position in the layer state

    def get_state_matrix(self):
        """Returns a matrix representing the positions of all allies."""
        return self.layer_state[:]

    def get_state(self):
        pos = np.zeros(2 * len(self.agents))
        idx = 0
        for agent in self.agents:
            pos[idx : (idx + 2)] = agent.get_state()
            idx += 2
        return pos
    
    def update(self):
        # Decay previous positions
        mask = self.layer_state > -np.inf
        self.layer_state[mask] -= 1  # Decrement the state of previously occupied positions
        # Reset positions that were more than 20 time steps ago
        self.layer_state[self.layer_state < -20] = -np.inf
        # Update positions based on current agent locations
        for agent in self.agents:
            x, y = agent.current_position()
            self.layer_state[x, y] = 0  # Set current agent positions to 0
    
class TargetLayer(AgentLayer):
    def __init__(self, xs, ys, targets, map_matrix, seed=None):
        super()._init_(xs, ys, targets, seed)
        self.targets = targets
        self.map_matrix = map_matrix
        #self.layer_state = np.full((xs, ys), -np.inf)
        self.ntargets = len(targets)
        self.goal = None

    def get_position(self, target_idx):
        """Returns the position of the given target."""
        return self.targets[target_idx].current_position()
    
    def move_targets(self, target_idx, action):
        """Moves the agent according to the defined action and updates the layer state.
           This is where the policies should come in, providing the action for each agent."""
        o_pos = self.targets[target_idx].current_position()
        n_pos = self.targets[target_idx].step(action)

        # Update the layer state for old and new positions
        self.update_positions(o_pos, n_pos)
        return n_pos

    def n_targets(self):
        return self.ntargets

class JammerLayer(AgentLayer):
    def __init__(self, xs, ys, jammers, activation_times=None, seed=1):
        self.jammers = jammers
        self.nagents = len(jammers)
        self.layer_state = np.full((xs, ys), -np.inf)
        self.activation_times = activation_times or [0] * len(jammers)  # Default to immediate activation

    def activate_jammers(self, current_time):
        """Activate jammers based on the current time and their respective activation times."""
        for i, jammer in enumerate(self.jammers):
            if current_time >= self.activation_times[i] and jammer.active == 0:
                self.activate_jammer(jammer)

    def activate_jammer(self, jammer):
        if not jammer.is_active() and not jammer.get_destroyed():
            jammer.activate()
            self.update_layer_state()

    def set_position(self, jammer_idx, new_position):
        jammer = self.agents[jammer_idx]
        if new_position != jammer.current_position():
            jammer.set_position(new_position[0], new_position[1])
            self.update_layer_state()

    def update_layer_state(self):
        """
        Updates the layer state matrix with current positions of all jammers. 
        0 in the matrix is a current jammer position.
        """
        self.layer_state.fill(-np.inf)
        for jammer in self.jammers:
            x, y = jammer.current_position()
            if jammer.is_active():
                self.layer_state[x, y] = 0

    def get_state_matrix(self):
        """Returns a matrix representing the positions of active jammers."""
        self.update_layer_state()
        return self.layer_state[:]