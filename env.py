import gymnasium as gym 
import numpy as np
import random
import yaml
import agent_utils
import jammer_utils
import target_utils
import heapq
import pygame
from skimage.transform import resize
from layer import AgentLayer, JammerLayer, TargetLayer
from gymnasium.utils import seeding
#from Continuous_controller.agent_controller import AgentController
from Discrete_controller.agent_controller import DiscreteAgentController
from Continuous_controller.reward import calculate_continuous_reward
from gymnasium import spaces
from path_processor import PathProcessor


class Environment(gym.Env):
    def __init__(self, config_path, render_mode="human"):
        super(Environment, self).__init__()
        # Load configuration from YAML
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialise from config
        self.D = self.config['grid_size']['D']
        self.obs_range = self.config['obs_range']
        self.pixel_scale = self.config['pixel_scale'] # Size in pixels of each map cell
        self.map_scale = self.config['map_scale'] # Scaling factor of map resolution
        self.seed_value = self.config['seed']
        self.comm_range = self.config['comm_range']
        self._seed(self.seed_value)

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
        
        if self.agent_type == 'task_allocation':
            self.agent_paths = {agent_id: [] for agent_id in range(self.num_agents)}
            self.current_waypoints = {agent_id: None for agent_id in range(self.num_agents)}
        
        # Assumes static environment map
        self.path_processor = PathProcessor(self.map_matrix, self.X, self.Y)

        self.agents = agent_utils.create_agents(self.num_agents, self.map_matrix, self.obs_range, self.np_random, self.path_processor, agent_positions, agent_type=self.agent_type, randinit=True)
        self.agent_layer = AgentLayer(self.X, self.Y, self.agents)

        # get agent id for class instance
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        self.num_targets = self.config['n_targets']
        self.targets = target_utils.create_targets(self.num_targets, self.map_matrix, self.obs_range, self.np_random, self.path_processor, target_positions, randinit=True)
        self.target_layer = TargetLayer(self.X, self.Y, self.targets, self.map_matrix)

        self.num_jammers = self.config['n_jammers']
        self.jammers = jammer_utils.create_jammers(self.num_jammers, self.map_matrix, self.np_random, self.config['jamming_radius'])
        self.jammer_layer = JammerLayer(self.X, self.Y, self.jammers)
        self.jammed_positions = set()

        # Define action and observation spaces
        if self.agent_type == 'discrete':
            self.action_space = spaces.Discrete(len(self.agents[0].eactions))
        else:
            #self.action_space = spaces.Dict({agent_id: agent.action_space for agent_id, agent in enumerate(self.agents)})
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_agents * 2,), dtype=np.float32) #changed it to this to work with stable baselines

        if self.agent_type == 'task_allocation' or 'discrete':
            self.observation_space = spaces.Dict({agent_id: spaces.Box(low=-20, high=1, shape=(self.global_state.shape[0], self.obs_range, self.obs_range), dtype=np.float32) for agent_id in range(self.num_agents)})
        else: 
            self.observation_space = spaces.Dict({agent_id: spaces.Box(low=-30, high=2, shape=(self.global_state.shape[0] + 1, self.obs_range, self.obs_range), dtype=np.float32) for agent_id in range(self.num_agents)})

        # Set global state layers
        self.global_state[0] = self.map_matrix
        self.global_state[1] = self.agent_layer.get_state_matrix()
        self.global_state[2] = self.target_layer.get_state_matrix()
        self.global_state[3] = self.jammer_layer.get_state_matrix()
        
        self.current_step = 0
        self.render_modes = render_mode
        self.screen = None
        pygame.init()


    def reset(self, seed= None, options: dict = None):
        """ Reset the environment for a new episode"""
        super().reset(seed=self.seed_value)
        info = {}
        if seed is not None:
            self.seed_value = seed
        self._seed(self.seed_value)
        # Reset global state
        self.global_state.fill(0)
        self.global_state[0] = self.map_matrix # Uncomment above code if map_matrix is changed by sim

        # Reinitialise agent positions
        if 'agent_positions' in self.config:
            agent_positions = [tuple(pos) for pos in self.config['agent_positions']]
        else:
            agent_positions = None

        self.agents = agent_utils.create_agents(self.num_agents, self.map_matrix, self.obs_range, self.np_random, self.path_processor, agent_positions, agent_type=self.agent_type, randinit=True)
        self.agent_layer = AgentLayer(self.X, self.Y, self.agents)
        
        # Reinitialise target positions
        if 'target_positions' in self.config:
            target_positions = [tuple(pos) for pos in self.config['target_positions']]
        else:
            target_positions = None

        self.targets = target_utils.create_targets(self.num_targets, self.map_matrix, self.obs_range, self.np_random, self.path_processor ,target_positions, randinit=True)
        self.target_layer = TargetLayer(self.X, self.Y, self.targets, self.map_matrix)

        # Reinitialise jammers
        self.jammers = jammer_utils.create_jammers(self.num_jammers, self.map_matrix, self.np_random, self.config['jamming_radius'])
        self.jammer_layer = JammerLayer(self.X, self.Y, self.jammers)
        
        self.jammed_positions = set()
        
        self.target_layer.update()
        self.agent_layer.update()
        # Update layers in global state
        self.global_state[1] = self.agent_layer.get_state_matrix()
        self.global_state[2] = self.target_layer.get_state_matrix()
        self.global_state[3] = self.jammer_layer.get_state_matrix()
        self.current_step = 0

        observations = {}
        #observations = np.zeros(self.observation_space.shape, dtype=np.float32)
        for agent_id in range(self.num_agents):
            obs = self.safely_observe(agent_id)
            self.agents[agent_id].set_observation_state(obs)
            observations[agent_id] = obs

        return observations, info
    

    def step(self, actions_dict):
        if self.agent_type == 'task_allocation':
            return self.task_allocation_step(actions_dict)
        else:
            return self.regular_step(actions_dict)
        
    
    def task_allocation_step(self, actions_dict):
        # print("\nStarting task_allocation_step")
        # print(f"Current agent positions: {[tuple(self.agent_layer.get_position(i)) for i in range(self.num_agents)]}")
    
        for agent_id, action in actions_dict.items():
            agent = self.agents[agent_id]
            start = tuple(self.agent_layer.get_position(agent_id))
            goal = agent.action_to_waypoint(action)  # Use the agent's method to convert action to waypoint
            self.current_waypoints[agent_id] = goal
            # print(f"\nAgent {agent_id}:")
            # print(f"  Current position: {start}")
            # print(f"  Goal position: {goal}")
            
            new_path = self.path_processor.get_path(start, goal)
            self.agent_paths[agent_id] = new_path
            
            # print(f"  Path length: {len(new_path)}")
            # if len(new_path) > 0:
            #     print(f"  First few steps: {new_path[:min(5, len(new_path))]}")
            # # else:
            #     print("  Empty path!")

        # Find the maximum path length
        max_path_length = max(len(path) for path in self.agent_paths.values())

        # Move agents and targets for max_path_length steps
        for step in range(max_path_length):
            # Move agents
            for agent_id in actions_dict.keys():
                if self.agent_paths[agent_id]:
                    next_pos = self.agent_paths[agent_id].pop(0)
                    self.agent_layer.set_position(agent_id, next_pos[0], next_pos[1])
                    # self.agents[agent_id].reset_action_space()

            # Move all targets
            for target in self.target_layer.targets:
                action = target.get_next_action()
                self.target_layer.move_targets(self.target_layer.targets.index(target), action)
            
            self.target_layer.update()
            self.agent_layer.update()
            
            # Update observations and communicate
            observations = {}
            observations = self.update_observations()
            self.share_and_update_observations()

            # Update jammed areas and global state
            self.jammer_layer.activate_jammers(self.current_step)
            self.update_jammed_areas()
            self.update_global_state()
            
            # Check for jammer destruction
            self.check_jammer_destruction()
            self.render()
            self.current_step += 1

        # Collect final local states and calculate rewards
        local_states, rewards = self.collect_local_states_and_rewards()
        
        terminated = self.is_episode_done()
        truncated = self.is_episode_done()
        info = {}
        
        # np.set_printoptions(threshold=np.inf)
        # print("local_states", local_states)
        # print("local_states[0]", local_states[0])
        # print("End of task_allocation_step")
        return local_states, rewards, terminated, truncated, info
    
    def regular_step(self, actions_dict):
        # Update target positions and layer state
        for i, target in enumerate(self.target_layer.targets):
            action = target.get_next_action()
            self.target_layer.move_targets(i, action)
        
        # Update agent positions and layer state based on the provided actions
        for agent_id, action in actions_dict.items():
            agent = self.agents[agent_id]
            self.agent_layer.move_agent(agent_id, action)
            
            # Check if the agent touches any jammer and destroy it
            agent_pos = self.agent_layer.agents[agent_id].current_position()
            for jammer in self.jammers:
                if not jammer.get_destroyed() and jammer.current_position() == tuple(agent_pos):
                    jammer.set_destroyed()
                    self.update_jammed_areas()  # Update jammed areas after destroying the jammer

        self.target_layer.update()
        self.agent_layer.update()
        # Update the global state with the new layer states
        self.update_global_state()
        
        # Update observations and communicate
        observations = {}
        observations = self.update_observations()
        self.share_and_update_observations()
        
        self.jammer_layer.activate_jammers(self.current_step)
        self.update_jammed_areas()
        self.check_jammer_destruction()

        # Collect final local states and calculate rewards
        local_states, rewards = self.collect_local_states_and_rewards()
        
        self.current_step += 1
        
        terminated = self.is_episode_done()
        truncated = self.is_episode_done()
        info = {}
        
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        print("observations", observations)
        # print("local_states", local_states)
        # print("local_states0", local_states[0])
        return local_states, rewards, terminated, truncated, info

    def update_observations(self): #Alex had this one
        observations = {}
        for agent_id in range(self.num_agents):
            obs = self.safely_observe(agent_id)
            # print("This should come first agent id", agent_id, obs)
            self.agent_layer.agents[agent_id].set_observation_state(obs)
            observations[agent_id] = obs
        return observations
    
    # def update_observations(self): # Hamish had this one
    #     for agent_id in range(self.num_agents):
    #         obs = self.safely_observe(agent_id)
    #         self.agent_layer.agents[agent_id].set_observation_state(obs)

    def collect_local_states_and_rewards(self):
        local_states = {}
        rewards = {}
        for agent_id in range(self.num_agents):
            agent = self.agent_layer.agents[agent_id]
            local_states[agent_id] = agent.get_state()  # This returns the local_state
            
            if self.agent_type == "discrete":
                reward = DiscreteAgentController.calculate_reward(agent)
            elif self.agent_type == "task_allocation":
                reward = DiscreteAgentController.calculate_reward(agent)  # TODO: Implement task allocation reward
            else: 
                reward = calculate_continuous_reward(agent, self)
            rewards[agent_id] = reward
        
        return local_states, rewards
    
    def action_to_waypoint(self, action):
        # Convert the action (which is now an index) to a waypoint (x, y) coordinate
        x = action // self.Y
        y = action % self.Y
        return np.array([x, y])
    
    def is_valid_action(self, agent_id, action):
        agent = self.agents[agent_id]
        if self.agent_type == 'task_allocation':
            return (tuple(action) in agent.get_valid_actions() and 
                    self.chebyshev_distance(agent.current_pos, action) <= agent.max_distance)
        else:
            return action in agent.action_space

    def chebyshev_distance(self, pos1, pos2):
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
    
    def check_jammer_destruction(self):
        for agent_id in range(self.num_agents):
            agent_pos = self.agent_layer.get_position(agent_id)
            for jammer in self.jammers:
                if not jammer.get_destroyed() and jammer.current_position() == tuple(agent_pos):
                    jammer.set_destroyed()
                    self.update_jammed_areas()
                    
    def update_global_state(self):
        self.global_state[1] = self.agent_layer.get_state_matrix()
        self.global_state[2] = self.target_layer.get_state_matrix()
        self.global_state[3] = self.jammer_layer.get_state_matrix()

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
    
    #need to update this, doing it for testing
    def is_episode_done(self):
        return False 

    def draw_agents(self):
        """
        Use pygame to draw agents and their paths.
        """
        for i, agent in enumerate(self.agent_layer.agents):
            x, y = agent.current_position()
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            col = (0, 0, 255)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 1.5))

            # Draw the agent's path
            if len(agent.path) > 1:
                for j in range(len(agent.path) - 1):
                    start_pos = (
                        int(self.pixel_scale * agent.path[j][0] + self.pixel_scale / 2),
                        int(self.pixel_scale * agent.path[j][1] + self.pixel_scale / 2)
                    )
                    end_pos = (
                        int(self.pixel_scale * agent.path[j + 1][0] + self.pixel_scale / 2),
                        int(self.pixel_scale * agent.path[j + 1][1] + self.pixel_scale / 2)
                    )
                    pygame.draw.line(self.screen, (255, 0, 0), start_pos, end_pos, 2)  # Draw red line
     
            
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
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 1.5))


    def draw_jammers(self):
        """
        Use pygame to draw jammers and jamming regions.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        # Where self.jammers is a list of jammer class objects
        for jammer in self.jammer_layer.jammers:
            x = jammer.position[0]
            y = jammer.position[1]
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            # Green for jammers
            col = (0, 255, 0)
            pygame.draw.circle(self.screen, col, center, int(self.pixel_scale / 1.5))
            # Draw jamming radius
            jamming_radius_pixels = jammer.radius*2 * self.pixel_scale  # Converting jamming radius to pixels
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
            
    def draw_waypoints(self):
        """
        Draw waypoints for task allocation agents.
        """
        if self.agent_type != 'task_allocation':
            return

        # Define a list of distinct colors for waypoints
        waypoint_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Lime
        ]

        for i, agent in enumerate(self.agents):
            agent_pos = agent.current_position()
            agent_center = (
                int(self.pixel_scale * agent_pos[0] + self.pixel_scale / 2),
                int(self.pixel_scale * agent_pos[1] + self.pixel_scale / 2),
            )

            waypoint = self.current_waypoints[i]
            if waypoint is not None:
                waypoint_center = (
                    int(self.pixel_scale * waypoint[0] + self.pixel_scale / 2),
                    int(self.pixel_scale * waypoint[1] + self.pixel_scale / 2),
                )

                # Draw a line from agent to waypoint
                color = waypoint_colors[i % len(waypoint_colors)]
                pygame.draw.line(self.screen, color, agent_center, waypoint_center, 2)

                # Draw the waypoint
                pygame.draw.circle(self.screen, color, waypoint_center, int(self.pixel_scale / 3))

                # Draw the agent number near the waypoint
                font = pygame.font.Font(None, 24)
                text = font.render(str(i), True, color)
                self.screen.blit(text, (waypoint_center[0] + 5, waypoint_center[1] + 5))

    #Still working on this function - Trying to make it work to show all observed area 
    def draw_all_agents_observations(self):

        observed_positions = set()
        for agent in self.agent_layer.agents:
            #observed_positions = (agent.local_state<=0) & (agent.local_state > -20)
            observed_positions = (agent.local_state[0]==0) | (agent.local_state[0] == -1)

        observed_positions_2 = agent.local_state[observed_positions]
        # print(observed_positions_2)

        for x, y in observed_positions_2:
            pos = pygame.Rect(
                self.pixel_scale * x, 
                self.pixel_scale * y,
                self.pixel_scale, 
                self.pixel_scale
            )
            col = (0 ,0, 225)
            pygame.draw.rect(self.screen, col, pos)
            
    def draw_agent_communication_range(self):
        """
        Use pygame to draw jammers and jamming regions.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        # Where self.jammers is a list of jammer class objects
        for i in range(self.agent_layer.n_agents()):
            x, y = self.agent_layer.get_position(i)
            center = (
                int(self.pixel_scale * x + self.pixel_scale / 2),
                int(self.pixel_scale * y + self.pixel_scale / 2),
            )
            # Draw jamming radius
            comms_radius_pixels = self.comm_range * self.pixel_scale  # Converting jamming radius to pixels
            # Semi-transparent green ellipse for jamming radius
            comms_area = pygame.Rect(center[0] - comms_radius_pixels / 2,
                                       center[1] - comms_radius_pixels / 2,
                                       comms_radius_pixels,
                                       comms_radius_pixels
                                    )
            pygame.draw.ellipse(self.screen, (72, 152, 255, 128), comms_area, width=1)
    
    
    def render(self, mode="human") -> None | np.ndarray | str | list:
        """ 
        Basic render of environment using matplotlib scatter plot.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        if self.screen is None:
            if self.render_modes == "human":
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
        self.draw_agent_communication_range()
        self.draw_jammers()
        self.draw_waypoints()

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation
        if self.render_modes == "human":
            pygame.event.pump()
            pygame.display.update()
        return (new_observation,
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_modes == "rgb_array"
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
        #obs = np.where(obs == -np.inf, -1e10, obs)
        obs = obs.transpose((0,2,1))
        obs = np.clip(obs, self.observation_space[agent_id].low, self.observation_space[agent_id].high)
        return obs

    def collect_obs(self, agent_layer, agent_id):
        return self.collect_obs_by_idx(agent_layer, agent_id)

    def collect_obs_by_idx(self, agent_layer, agent_idx):
        if self.agent_type == 'task_allocation' or 'discrete':
            # Initialize the observation array to include velocity, position, and the previous state observation
            obs = np.full((self.global_state.shape[0], self.obs_range, self.obs_range), fill_value=-20, dtype=np.float32)
        else:
            obs = np.full((self.global_state.shape[0] + 1, self.obs_range, self.obs_range), fill_value=-20, dtype=np.float32)

        # Get the current position of the agent
        xp, yp = agent_layer.get_position(agent_idx)
        # Get the current velocity of the agent
        if self.agent_type == 'continuous':
            vx, vy = self.agents[agent_idx].velocity
        
        # Calculate bounds for the observation
        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)
        
        xlo1 = int(xlo)
        xhi1 = int(xhi)
        ylo1 = int(ylo)
        yhi1 = int(yhi)
        xolo = int(xolo)
        xohi = int(xohi)
        yolo = int(yolo)
        yohi = int(yohi)
        
        # Populate the observation array with data from all layers
        for layer in range(self.global_state.shape[0]):
            obs_slice = self.global_state[layer, xlo1:xhi1, ylo1:yhi1]
            obs_shape = obs_slice.shape
            if xolo != 0 and yohi == 17: 
                pad_width = [(0,0), (self.obs_range-obs_shape[0], 0),(self.obs_range-obs_shape[1], 0)]
            elif yolo !=0 and xolo == 0:
                pad_width = [(0,0), (0, self.obs_range-obs_shape[0]),(self.obs_range-obs_shape[1], 0)]
            elif xolo == 0 and yolo == 0:
                pad_width = [(0,0), (0, self.obs_range-obs_shape[0]),(0, self.obs_range-obs_shape[1])]
            elif yolo == 0 and xohi == 17:
                pad_width = [(0,0), (self.obs_range-obs_shape[0], 0),(0, self.obs_range-obs_shape[1])]
            obs_padded = np.pad(obs_slice, pad_width[1:], mode='constant', constant_values=-21)
            obs[layer, :obs_padded.shape[0], :obs_padded.shape[1]] = obs_padded
        
        if self.agent_type == 'continuous':
            # Add the velocity and position to the observation
            obs[-1, 0, 0] = vx
            obs[-1, 0, 1] = vy
            obs[-1, 0, 2] = xp
            obs[-1, 0, 3] = yp
    
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
        for i, agent in enumerate(self.agent_layer.agents):
            # safely_observe returns the current observation of agent i - but that should be called before this function
            current_obs = agent.get_observation_state()
            current_pos = agent.current_position()
            # agent.set_observation_state(current_obs)
            for j, other_agent in enumerate(self.agent_layer.agents):
                if i != j:
                    other_pos = other_agent.current_position()
                    agent_id = self.agent_name_mapping[agent]
                    other_agent_id = self.agent_name_mapping[other_agent]
                    if self.within_comm_range(current_pos, other_pos) and not self.is_comm_blocked(agent_id) and not self.is_comm_blocked(other_agent_id):
                        other_agent.update_local_state(current_obs, current_pos)
                        agent.communicated = True 

    # def share_and_update_observations(self):
    #     """
    #     Updates each agent's internal observation state and internal local (entire env) state.
    #     Will merge current observations of agents within communication range into each agent's local state.
    #     This function should be run in the step function.
    #     """

    #     np.set_printoptions(threshold=np.inf, linewidth=np.inf) #THIS is for testing 

    #     for i, agent in enumerate(self.agents):
    #         current_obs = agent.get_observation_state()
    #         current_pos = agent.current_position()
    #         #print(f"Agent {i} local state before communication:")

    #         for j, other_agent in enumerate(self.agents):
    #             if i != j:
    #                 other_pos = other_agent.current_position()
    #                 if self.within_comm_range(current_pos, other_pos):
    #                     if not self.is_comm_blocked(i):
    #                         #print(f"Agent {i} is communicating with Agent {j}")

    #                         # Print the other agent's observation
    #                         #print(f"Agent {j}'s observation:")
    #                         #print(other_agent.get_observation_state())

    #                         # Print the section of the agent's local state before communication
    #                         #print(f"Agent {i} local state at Agent {j}'s observation location before communication:")
    #                         #self.print_local_state_section(agent, other_pos)

    #                         other_agent.update_local_state(current_obs, current_pos)
    #                         agent.communicated = True 

    #                         # Print the section of the agent's local state after communication
    #                         #print(f"Agent {i} local state at Agent {j}'s observation location after communication:")
    #                         #self.print_local_state_section(agent, other_pos)
    #                         #print("---")
    #                     # else:
    #                     #    print(f"Agent {i} is within a jammed area and cannot communicate with Agent {j}")
    #                 #else:
    #                     #print(f"Agent {i} is out of communication range with Agent {j}")
    #                 #if self.is_comm_blocked(i):
    #                     #print("comm is blocked")

    def print_local_state_section(self, agent, other_pos):
        """
        Prints the section of the agent's local state that corresponds to the other agent's observation location.
        """
        obs_range = self.obs_range
        obs_half_range = obs_range // 2
        x_start, x_end = other_pos[0] - obs_half_range, other_pos[0] + obs_half_range + 1
        y_start, y_end = other_pos[1] - obs_half_range, other_pos[1] + obs_half_range + 1

        x_start = int(x_start)
        x_end = int(x_end)
        y_start = int(y_start)
        y_end = int(y_end)
        
        local_state_section = agent.local_state[:, x_start:x_end, y_start:y_end]
        print(local_state_section)
    
    
    def within_comm_range(self, agent1, agent2):
        """Checks two agents are within communication range. Assumes constant comm range for all agents."""
        distance = np.linalg.norm(np.array(agent1) - np.array(agent2))
        return distance <= self.comm_range/2
    
    
    def is_comm_blocked(self, agent_id):
        """
        Determine if an agent's communication is currently blocked by any active jammers.

        Args:
        - agent_id (int): ID of the agent to check.

        Returns:
        - bool: True if communication is blocked, False otherwise.
        """
        x, y = self.agent_layer.agents[agent_id].current_position()
        x_int = int(x)
        y_int = int(y)
        agent_pos = x_int, y_int
        return tuple(agent_pos) in self.jammed_positions
    
    
    # JAMMING FUNCTIONS #
    def activate_jammer(self, jammer_index):
        jammer = self.jammer_layer.jammers[jammer_index]
        if not jammer.is_active():
            self.jammer_layer.activate_jammer(jammer)
            self.update_jammed_areas()


    def deactivate_jammer(self, jammer_index):
        jammer = self.jammer_layer.jammers[jammer_index]
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
        np.random.seed(seed)
        random.seed(seed)

    def run_simulation(self, max_steps=100):
        running = True
        step_count = 0

        while running and step_count < max_steps:
            print(f"Step: {step_count}")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Generate new actions for all agents in every step
            action_dict = {agent_id: agent.get_next_action() for agent_id, agent in enumerate(self.agents)}
            #print("Action_dict: ", action_dict)
            # Update environment states with the action_dict
            observations, rewards, terminated, truncated, self.info = self.step(action_dict)

            self.render()  # Render the current state to the screen

            step_count += 1

            if terminated or truncated:
                break

        pygame.image.save(self.screen, "environment_snapshot.png")
        self.reset()

        pygame.quit()


# config_path = 'config.yaml' 
# env = Environment(config_path)
# Environment.run_simulation(env, max_steps=1)