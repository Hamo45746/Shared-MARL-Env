import numpy as np
import yaml
from pettingzoo.sisl.pursuit.utils import agent_utils
import jammer_utils
import heapq
import pygame
from skimage.transform import resize
from layer import AgentLayer, JammerLayer, TargetLayer
from gymnasium.utils import seeding

class Environment:
    def __init__(self, config_path, render_mode="human"):
        # Load configuration from YAML
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize from config
        
        self.D = self.config['grid_size']['D']
        self.obs_range = self.config['obs_range']
        self.pixel_scale = self.config['pixel_scale'] # Size in pixels of each map cell
        self.map_scale = self.config['map_scale'] # Scaling factor of map resolution
        self.seed = self.config['seed']
        self.comm_range = self.config['comm_range']
        self._seed(self.seed)
        
        # Constants for rewards and penalties
        self.JAMMER_DISCOVERY_REWARD = self.config['jammer_discovery_reward']
        self.TARGET_DISCOVERY_REWARD = self.config['target_discovery_reward']
        self.TRACKING_REWARD = self.config['tracking_reward']
        self.DESTRUCTION_REWARD = self.config['destruction_reward']
        self.MOVEMENT_PENALTY = self.config['movement_penalty']
        self.EXPLORATION_REWARD = self.config['exploration_reward']
        
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
        
        # Initialize agents, targets, jammers
        # Created jammers, targets and agents at random positions if not given a position from config
        if 'agent_positions' in self.config:
            agent_positions = [tuple(pos) for pos in self.config['agent_positions']]
        else:
            agent_positions = None
        
        if 'target_positions' in self.config:
            target_positions = [tuple(pos) for pos in self.config['target_positions']]
        else:
            target_positions = None
        
        self.num_agents = self.config['n_agents']
        self.agents = agent_utils.create_agents(self.num_agents, self.map_matrix, self.obs_range, self.np_random, agent_positions, randinit=True)
        self.agent_layer = AgentLayer(self.X, self.Y, self.agents)

        self.num_targets = self.config['n_targets']
        self.targets = agent_utils.create_agents(self.num_targets, self.map_matrix, self.obs_range, self.np_random, target_positions, randinit=True)
        self.target_layer = TargetLayer(self.X, self.Y, self.targets, self.map_matrix)

        self.num_jammers = self.config['n_jammers']
        self.jammers = jammer_utils.create_jammers(self.num_jammers, self.map_matrix, self.np_random, self.config['jamming_radius'])
        self.jammer_layer = JammerLayer(self.X, self.Y, self.jammers)
        
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
        
        # Reinitialize the map and entities
        # original_map = np.load(self.config['map_path'])[:, :, 0]
        # resized_map = resize(original_map, (self.X, self.Y), order=0, preserve_range=True, anti_aliasing=False)
        # self.map_matrix = (resized_map != 0).astype(int)
        
        # Reset global state
        self.global_state.fill(0)
        self.global_state[0] = self.map_matrix # Uncomment above code if map_matrix is changed by sim

        # Reinitialize agent positions
        if 'agent_positions' in self.config:
            agent_positions = [tuple(pos) for pos in self.config['agent_positions']]
        else:
            agent_positions = None

        self.agents = agent_utils.create_agents(self.num_agents, self.map_matrix, self.obs_range, self.np_random, agent_positions, randinit=True)
        self.agent_layer = AgentLayer(self.X, self.Y, self.agents)

        # Reinitialize target positions
        if 'target_positions' in self.config:
            target_positions = [tuple(pos) for pos in self.config['target_positions']]
        else:
            target_positions = None

        self.targets = agent_utils.create_agents(self.num_targets, self.map_matrix, self.obs_range, self.np_random, target_positions, randinit=True)
        self.target_layer = TargetLayer(self.X, self.Y, self.targets, self.map_matrix)

        # Reinitialize jammers
        self.jammers = jammer_utils.create_jammers(self.num_jammers, self.map_matrix, self.np_random, self.config['jamming_radius'])
        self.jammer_layer = JammerLayer(self.X, self.Y, self.jammers)

        # Update layers in global state
        self.global_state[1] = self.agent_layer.get_state_matrix()
        self.global_state[2] = self.target_layer.get_state_matrix()
        self.global_state[3] = self.jammer_layer.get_state_matrix()


    def compute_path_reward(self, agent_id, chosen_location, path_steps):
        """
        Compute the reward based on the agent's path and the encounters along it, including exploration of outdated areas.

        Args:
        - agent_id (int): ID of the agent.
        - chosen_location (tuple): The final destination chosen by the agent.
        - path_steps (list): A list of tuples representing the path coordinates.

        Returns:
        - float: The computed reward for the path taken.
        """
        reward = 0

        target_identified = False

        for step in path_steps:
            # Check for jammer destruction
            if step == chosen_location and self.is_jammer_location(step):
                reward += self.DESTRUCTION_REWARD
                self.destroy_jammer(step)

            # Check for target identification and tracking
            if self.is_target_in_observation(agent_id, step):
                if not target_identified:
                    reward += self.TARGET_DISCOVERY_REWARD
                    target_identified = True
                else:
                    reward += self.TRACKING_REWARD

            # Reward for exploring outdated regions
            if self.is_information_outdated(step, self.OUTDATED_INFO_THRESHOLD):
                reward += self.EXPLORATION_REWARD
                self.update_global_state(agent_id, step)

        return reward
    
    
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
            if not jammer.is_destroyed and np.linalg.norm(np.array(agent_pos) - np.array(jammer.position)) <= self.config['jamming_radius']:
                return True
        return False



    def step(self, action, agent_id, is_last):
        agent_layer = self.agent_layer
        opponent_layer = self.target_layer

        # actual action application, change the pursuer layer
        agent_layer.move_agent(agent_id, action)

        # Update only the agent layer
        self.global_state[1] = self.agent_layer.get_state_matrix()

        self.latest_reward_state = self.reward() / self.num_agents # Reward not implemented

        if is_last:
            # Possibly change the evader layer
            ev_remove, pr_remove, pursuers_who_remove = self.remove_agents()

            for i in range(opponent_layer.n_agents()):
                # controller input should be an observation, but doesn't matter right now
                opponent_layer.move_agent(i, a)

            self.latest_reward_state += self.catch_reward * pursuers_who_remove
            self.latest_reward_state += self.urgency_reward
            self.frames = self.frames + 1

        # Update the remaining layers
        self.global_state[2] = self.target_layer.get_state_matrix()

        global_val = self.latest_reward_state.mean()
        local_val = self.latest_reward_state
        self.latest_reward_state = (
            self.local_ratio * local_val + (1 - self.local_ratio) * global_val
        )

        if self.render_mode == "human":
            self.render()
    

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
        for i in range(self.target_layer.ntargets):
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
            
            
    ## OBSERVATION FUNCTIONS ## - TODO: Not currently sharing agent observations
    def safely_observe(self, agent_id):
        obs = self.collect_obs(self.agent_layer, agent_id)
        return obs


    def collect_obs(self, agent_layer, agent_id):
        return self.collect_obs_by_idx(agent_layer, agent_id)


    def collect_obs_by_idx(self, agent_layer, agent_idx):
        obs = np.zeros((3, self.obs_range, self.obs_range), dtype=np.float32) # With shared observations later these may need to change
        obs[0].fill(-np.inf)
        xp, yp = agent_layer.get_position(agent_idx)

        xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = self.obs_clip(xp, yp)

        # Adjust observation data retrieval based on global state and its layers
        obs[0:3, xolo:xohi, yolo:yohi] = self.global_state[0:3, xlo:xhi, ylo:yhi]
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
        for i, agent in enumerate(self.agents):
            current_obs, _ = self.safely_observe(i)  # Gets current observations for agent i
            for j, other_agent in enumerate(self.agents):
                if i != j and self.within_comm_range(agent, other_agent):
                    self.update_global_state_from_observation(other_agent, current_obs)


    def update_global_state_from_observation(self, receiving_agent, obs):
        # Extract the current position of the receiving agent
        agent_pos = receiving_agent.current_position()
        # Determine the offset based on the agent's position and the observation range
        for dx in range(-self.obs_range // 2, self.obs_range // 2 + 1):
            for dy in range(-self.obs_range // 2, self.obs_range // 2 + 1):
                global_x = agent_pos[0] + dx
                global_y = agent_pos[1] + dy
                # Update the global state at the receiving agent’s location
                if 0 <= global_x < self.X and 0 <= global_y < self.Y:
                    receiving_agent.global_state[0:3, global_x, global_y] = obs[0:3, self.obs_range//2 + dx, self.obs_range//2 + dy]
                    
    
    def within_comm_range(self, agent1, agent2):
        distance = np.linalg.norm(np.array(agent1.current_position()) - np.array(agent2.current_position()))
        return distance <= self.comm_range

    
    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        
      

config_path = 'config.yaml' 

map_processor = Environment(config_path)

map_processor.render()
pygame.image.save(map_processor.screen, "environment_snapshot.png")
#pygame.time.delay(10000)
