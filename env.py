#import os
##os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import yaml
import agent_utils
import jammer_utils
import target_utils
import heapq
import pygame
from skimage.transform import resize
from layer import AgentLayer, JammerLayer, TargetLayer
from gymnasium.utils import seeding
from gymnasium import spaces
#from stable_baselines3.common.env_checker import check_env

class Environment:
    def __init__(self, config_path, render_mode="human"):
        # Load configuration from YAML
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialise from config
        self.D = self.config['grid_size']['D']
        self.obs_range = self.config['obs_range']
        self.pixel_scale = self.config['pixel_scale'] # Size in pixels of each map cell
        self.map_scale = self.config['map_scale'] # Scaling factor of map resolution
        self.seed = self.config['seed']
        self.comm_range = self.config['comm_range']
        self._seed(self.seed)
        
        # Load the map
        original_map = np.load(self.config['map_path'])[:, :, 0]
        original_map = original_map.transpose() 
 
        original_x, original_y = original_map.shape
        # Scale map according to config
        self.X = int(original_x * self.map_scale)
        self.Y = int(original_y * self.map_scale)
        # Resizing the map, using nearest interpolation
        resized_map = resize(original_map, (self.X, self.Y), order=0, preserve_range=True, anti_aliasing=False)
        # Assuming obstacles are any non-zero value - convert to binary map
        obstacle_map = (resized_map != 0).astype(int)
        self.map_matrix = obstacle_map
        # Global state includes layers for map, agents, targets, and jammers
        self.global_state = np.zeros((self.D,) + self.map_matrix.shape, dtype=np.float32)
        
        # Initialise agents, targets, jammers
        # Created jammers, targets and agents at random positions if not given a position from config
        if 'agent_positions' in self.config:
            agent_positions = [tuple(pos) for pos in self.config['agent_positions']]
        else:
            agent_positions = None
        
        if 'target_positions' in self.config:
            target_positions = [tuple(pos) for pos in self.config['target_positions']]
        else:
            target_positions = None

        if 'target_goals' in self.config:
            target_goals = [tuple(pos) for pos in self.config['target_goals']]
        else:
            target_goals = None
        
        self.num_agents = self.config['n_agents']
        self.agent_type = self.config.get('agent_type', 'discrete')
        self.agents = agent_utils.create_agents(self.num_agents, self.map_matrix, self.obs_range, self.np_random, agent_positions, agent_type=self.agent_type, randinit=True)
        self.agent_layer = AgentLayer(self.X, self.Y, self.agents)
        
        # get agent id for class instance
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        self.num_targets = self.config['n_targets']
        self.targets = target_utils.create_targets(self.num_targets, self.map_matrix, self.obs_range, self.np_random, target_positions, target_goals, randinit=True)
        self.target_layer = TargetLayer(self.X, self.Y, self.targets, self.map_matrix)

        self.num_jammers = self.config['n_jammers']
        self.jammers = jammer_utils.create_jammers(self.num_jammers, self.map_matrix, self.np_random, self.config['jamming_radius'])
        self.jammer_layer = JammerLayer(self.X, self.Y, self.jammers)
        self.jammed_positions = None
        #self.update_jammed_areas()
        
        # Define action and observation spaces
        self.action_spaces = [spaces.Discrete(len(self.agents[0].eactions)) for _ in range(self.num_agents)]
        self.observation_spaces = [spaces.Box(low=0, high=np.inf, shape=(self.obs_range, self.obs_range, self.D), dtype=np.float32) for _ in range(self.num_agents)]
                
        # Set global state layers
        self.global_state[0] = self.map_matrix
        self.global_state[1] = self.agent_layer.get_state_matrix()
        self.global_state[2] = self.target_layer.get_state_matrix()
        self.global_state[3] = self.jammer_layer.get_state_matrix()
        
        # Pygame for rendering
        self.render_mode = render_mode
        self.screen = None
        pygame.init()


    def reset(self):
        """ Reset the environment for a new episode"""
        
        # Reinitialise the map and entities
        # original_map = np.load(self.config['map_path'])[:, :, 0]
        # resized_map = resize(original_map, (self.X, self.Y), order=0, preserve_range=True, anti_aliasing=False)
        # self.map_matrix = (resized_map != 0).astype(int)
        
        # Reset global state
        self.global_state.fill(0)
        self.global_state[0] = self.map_matrix # Uncomment above code if map_matrix is changed by sim

        # Reinitialise agent positions
        if 'agent_positions' in self.config:
            agent_positions = [tuple(pos) for pos in self.config['agent_positions']]
        else:
            agent_positions = None

        self.agents = agent_utils.create_agents(self.num_agents, self.map_matrix, self.obs_range, self.np_random, agent_positions, agent_type=self.agent_type, randinit=True)
        self.agent_layer = AgentLayer(self.X, self.Y, self.agents)
        
        # Reinitialise target positions
        if 'target_positions' in self.config:
            target_positions = [tuple(pos) for pos in self.config['target_positions']]
        else:
            target_positions = None

        self.targets = target_utils.create_targets(self.num_targets, self.map_matrix, self.obs_range, self.np_random, target_positions, randinit=True)
        self.target_layer = TargetLayer(self.X, self.Y, self.targets, self.map_matrix)

        # Reinitialise jammers
        self.jammers = jammer_utils.create_jammers(self.num_jammers, self.map_matrix, self.np_random, self.config['jamming_radius'])
        self.jammer_layer = JammerLayer(self.X, self.Y, self.jammers)
        
        self.jammed_positions = set()

        # Update layers in global state
        self.global_state[1] = self.agent_layer.get_state_matrix()
        self.global_state[2] = self.target_layer.get_state_matrix()
        self.global_state[3] = self.jammer_layer.get_state_matrix()

        # Return all agent observations
        return {agent: self.safely_observe(i) for i, agent in enumerate(self.agents)}
    
    def is_comm_blocked(self, agent_id):
        """
        Determine if an agent's communication is currently blocked by any active jammers.

        Args:
        - agent_id (int): ID of the agent to check.

        Returns:
        - bool: True if communication is blocked, False otherwise.
        """
        agent_pos = self.agents[agent_id].position
        for jammer in self.jammers:
            if jammer.active and np.linalg.norm(np.array(agent_pos) - np.array(jammer.position)) <= self.config['jamming_radius']:
                return True
        return False
            
    def step(self, actions_dict):
        # Need to update target position in target_layer, and target class itself
        # Ensure three variables receive update: self.targets for each target class, self.target_layer state, target objects in target list within target_layer 
        # Update current_position in each target instance, should be same target instance stored in self.targets and self.target_layer.targets.
        # Then do target_layer.update() to update its representation of the current state. - same for agent
        # Update target positions and layer state
       
        for i, target in enumerate(self.target_layer.targets):
            action = target.get_next_action()
            self.target_layer.move_targets(i, action)

        #for target_id, target in enumerate(env.targets):
            #action = target.get_next_action()
            #self.target_layer.move_targets(target_id, action)

        # Update agent positions and layer state based on the provided actions
        for agent_id, action in actions_dict.items():
            self.agent_layer.move_agent(agent_id, action)

        # Update the global state with the new agent and target layer states
        self.global_state[1] = self.agent_layer.get_state_matrix()
        self.global_state[2] = self.target_layer.get_state_matrix()

        # Update jammed areas based on the current state of jammers
        self.update_jammed_areas()

        # Collect observations for each agent
        observations = {}
        for agent_id in range(self.num_agents):
            obs = self.safely_observe(agent_id)
            self.agents[agent_id].set_observation_state(obs)
            observations[agent_id] = obs
            
            # DEBUGGING: Print the agent's observation and corresponding section of map matrix
            # agent_pos = self.agents[agent_id].current_position()
            # obs_range = self.obs_range
            # obs_half_range = obs_range // 2
            # x_start, x_end = agent_pos[0] - obs_half_range, agent_pos[0] + obs_half_range + 1
            # y_start, y_end = agent_pos[1] - obs_half_range, agent_pos[1] + obs_half_range + 1

            # print(f"Agent {agent_id} Observation:")
            # print(self.agents[agent_id].get_observation_state()[0])
            # print(f"Agent {agent_id} Position: {agent_pos}")
            # print("Corresponding Map Matrix Section:")
            # print(self.map_matrix[x_start:x_end, y_start:y_end])
            # print("---")

        # Share and update observations among agents within communication range
        self.share_and_update_observations() # TODO: Does this func work?

        # Calc rewards for each agent
        rewards = {}
        for agent_id in range(self.num_agents):
            agent = self.agents[agent_id]
            reward = self.calculate_reward(agent)  # Implement the reward calculation logic in a separate function, depends on agent type
            rewards[agent_id] = reward

        # Determine if the episode is done (implement termination conditions here)
        done = self.is_episode_done()
        # Create the info dictionary (idk if needed?)
        info = {}
        return observations, rewards, done, info
    
    # Getter functions for action and observation space
    def action_space(self, agent):
            return self.action_spaces[self.agent_name_mapping[agent]]

    def observation_space(self, agent):
        return self.observation_spaces[self.agent_name_mapping[agent]]

    def draw_model_state(self):
        """
        Use pygame to draw environment map.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        # 0 is clear space flag
        x_len, y_len = self.global_state[0].shape
        for x in range(x_len):
            for y in range(y_len):
                pos = pygame.Rect(
                    self.pixel_scale * x,
                    self.pixel_scale * y,
                    self.pixel_scale,
                    self.pixel_scale,
                )
                col = (0, 0, 0) 
                if self.global_state[0][x][y] != 0:
                    col = (255, 255, 255)
                pygame.draw.rect(self.screen, col, pos)

    # need to delete this, just doing it for testing 
    def calculate_reward(self, agent):
        reward = 1
        return reward 
    
    def is_episode_done(self):
        return False 

    def draw_agents(self):
        """
        Use pygame to draw agents.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        for i in range(self.agent_layer.n_agents()):
            x, y = self.agent_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (0, 0, 255)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))
            
            
    def draw_targets(self):
        """
        Use pygame to draw targets.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        for i in range(self.target_layer.n_targets()):
            x, y = self.target_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (255, 0, 0)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))


    def draw_jammers(self):
        """
        Use pygame to draw jammers and jamming regions.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        # Where self.jammers is a list of jammer class objects
        for jammer in self.jammers:
            x = jammer.position[0]
            y = jammer.position[1]
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            # Green for jammers
            col = (0, 255, 0)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 3))
            # Draw jamming radius
            jamming_radius_pixels = jammer.radius * self.pixel_scale  # Converting jamming radius to pixels
            # Semi-transparent green ellipse for jamming radius
            jamming_area = pygame.Rect(center[0] - jamming_radius_pixels / 2,
                                       center[1] - jamming_radius_pixels / 2,
                                       jamming_radius_pixels,
                                       jamming_radius_pixels
                                    )
            pygame.draw.ellipse(self.screen, (0, 255, 0, 128), jamming_area, width=1)


    def draw_agents_observations(self):
        """
        Use pygame to draw agents observation regions.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        for i in range(self.agent_layer.n_agents()):
            x, y = self.agent_layer.get_position(i)
            patch = pygame.Surface(
                (self.pixel_scale * self.obs_range, self.pixel_scale * self.obs_range)
            )
            patch.set_alpha(128)
            patch.fill((72, 152, 255))
            ofst = self.obs_range / 2.0
            self.screen.blit(
                patch,
                (
                    self.pixel_scale * (x - ofst + 1 / 2),
                    self.pixel_scale * (y - ofst + 1 / 2),
                ),
            )
    
    
    def render(self, mode="human") -> None | np.ndarray | str | list:
        """ 
        Basic render of environment using matplotlib scatter plot.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.pixel_scale * self.X, self.pixel_scale * self.Y)
                )
                pygame.display.set_caption("Search & Track Task Assign")
            else:
                self.screen = pygame.Surface(
                    (self.pixel_scale * self.X, self.pixel_scale * self.Y)
                )

        self.screen.fill((0, 0, 0))  # Clears the screen with black
        self.draw_model_state()
        self.draw_targets()
        self.draw_agents()
        self.draw_agents_observations()
        self.draw_jammers()

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
        return (new_observation,
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def state(self) -> np.ndarray:
        return self.global_state
    
    
    def close(self):
        """
        Closes any resources that should be released.

        Closes the rendering window, subprocesses, network connections,
        or any other resources that should be released.
        """
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            
            
    ## OBSERVATION FUNCTIONS ##
    def observe(self, agent):
        return self.safely_observe(self.agent_name_mapping[agent])
     
     
    def safely_observe(self, agent_id):
        obs = self.collect_obs(self.agent_layer, agent_id)
        return obs
    

    def collect_obs(self, agent_layer, agent_id):
        return self.collect_obs_by_idx(agent_layer, agent_id)


    def collect_obs_by_idx(self, agent_layer, agent_idx):
        # Initialise the observation array for all layers, ensuring no information loss
        obs = np.full((self.global_state.shape[0], self.obs_range, self.obs_range), fill_value=-np.inf, dtype=np.float32)
        # Get the current position of the agent
        xp, yp = agent_layer.get_position(agent_idx)
        # Calculate bounds for the observation
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)
        # Populate the observation array with data from all layers
        for layer in range(self.global_state.shape[0]):
            obs[layer, xolo:xohi, yolo:yohi] = self.global_state[layer, xlo:xhi, ylo:yhi]

        return obs

    def obs_clip(self, x, y):
        xld = x - self.obs_range // 2
        xhd = x + self.obs_range // 2
        yld = y - self.obs_range // 2
        yhd = y + self.obs_range // 2
        xlo, xhi, ylo, yhi = (
            np.clip(xld, 0, self.X - 1),
            np.clip(xhd, 0, self.X - 1),
            np.clip(yld, 0, self.Y - 1),
            np.clip(yhd, 0, self.Y - 1),
        )
        xolo, yolo = abs(np.clip(xld, -self.obs_range // 2, 0)), abs(np.clip(yld, -self.obs_range // 2, 0))
        xohi, yohi = xolo + (xhi - xlo), yolo + (yhi - ylo)
        return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1


    def share_and_update_observations(self):
        """
        Updates each agent classes internal observation state and internal local (entire env) state.
        Will merge current observations of agents within communication range into each agents local state.
        This function should be run in the step function.
        """
        for i, agent in enumerate(self.agents):
            # safely_observe returns the current observation of agent i
            current_obs = self.safely_observe(i)
            current_pos = agent.current_position()
            agent.set_observation_state(current_obs)
            
            for j, other_agent in enumerate(self.agents):
                if i != j:
                    other_pos = other_agent.current_position()
                    if self.within_comm_range(current_pos, other_pos):
                        other_agent.update_local_state(current_obs, current_pos)
    
    
    def within_comm_range(self, agent1, agent2):
        """Checks two agents are within communication range. Assumes constant comm range for all agents."""
        distance = np.linalg.norm(np.array(agent1) - np.array(agent2))
        return distance <= self.comm_range
    
    
    # JAMMING HELPER FUNCTIONS #
    
    def activate_jammer(self, jammer_index):
        jammer = self.jammer_layer.agents[jammer_index]
        if not jammer.is_active():
            self.jammer_layer.activate_jammer(jammer)
            self.update_jammed_areas()


    def deactivate_jammer(self, jammer_index):
        jammer = self.jammer_layer.agents[jammer_index]
        if jammer.is_active():
            jammer.deactivate()
            self.update_jammed_areas()


    def move_jammer(self, jammer_index, new_position):
        self.jammer_layer.set_position(jammer_index, *new_position)
        self.update_jammed_areas()


    def update_jammed_areas(self):
        """
        Recalculate the jammed areas based on the current state of all jammers.
        Stores the jammed grid positions in a cache. This function should be called any time a 
        jammers position, activation, and destruction status changes.
        """
        if self.jammed_positions is not None:
            self.jammed_positions.clear()
        #for jammer in self.jammer_layer.agents:
        for jammer in self.jammer_layer.jammers:
            if jammer.is_active() and not jammer.get_destroyed():
                jammed_area = self.calculate_jammed_area(jammer.current_position(), jammer.radius)
                self.jammed_positions.update(jammed_area)

    def calculate_jammed_area(self, position, radius):
        """
        Calculate the set of grid coordinates affected by a jammer's radius given its position.
        
        Parameters:
        - position (tuple): The (x, y) coordinates of the jammer on the grid.
        - radius (int): The radius within which the jammer affects other units.
        
        Returns:
        - set of tuples: Set of (x, y) coordinates representing the jammed area.
        """
        center_x, center_y = position
        jammed_area = set()

        # Determine the grid bounds of the jammed area
        x_lower_bound = max(center_x - radius, 0)
        x_upper_bound = min(center_x + radius, self.X - 1)
        y_lower_bound = max(center_y - radius, 0)
        y_upper_bound = min(center_y + radius, self.Y - 1)

        # Iterate over the range defined by the radius around the jammer's position
        for x in range(x_lower_bound, x_upper_bound + 1):
            for y in range(y_lower_bound, y_upper_bound + 1):
                # Calculate the distance from the center to ensure it's within the circle defined by the radius
                distance = np.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
                if distance <= radius:
                    jammed_area.add((x, y))

        return jammed_area


    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)

    def run_simulation(env, max_steps=100):
        running = True
        step_count = 0
        while running and step_count < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Create the action_dict for all agents
            action_dict = {agent_id: agent.get_next_action() for agent_id, agent in enumerate(env.agents)}

            # Update environment states with the action_dict
            observations, rewards, done, info = env.step(action_dict)

            #env.step()  # Update environment states
            env.render()  # Render the current state to the screen

            pygame.display.flip()  # Update the full display Surface to the screen
            pygame.time.wait(100)  # Wait some time so it's visually comprehensible

            step_count += 1

        pygame.quit()

config_path = 'config.yaml' 
env = Environment(config_path)
Environment.run_simulation(env)