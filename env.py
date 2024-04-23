import numpy as np
import yaml
from pettingzoo.sisl.pursuit.utils import agent_utils
from layer import AgentLayer, JammerLayer, TargetLayer
from gymnasium.utils import seeding
import jammer_utils
import heapq
import pygame
from skimage.transform import resize



class Environment:
    def __init__(self, config_path, render_mode="human"):
        # Load configuration from YAML
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Initialize from config
        
        self.D = config['grid_size']['D']
        self.obs_range = config['obs_range']
        self.pixel_scale = config['pixel_scale'] # Size in pixels of each map cell
        self.map_scale = config['map_scale'] # Scaling factor of map resolution
        self.seed = config['seed']
        self._seed(self.seed)
        
        # Load the map
        original_map = np.load(config['map_path'])[:, :, 0]
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
        # Created jammers, targets and agents at random positions - TODO: Update this to position from config file
        self.num_agents = config['n_agents']
        self.agents = agent_utils.create_agents(self.num_agents, self.map_matrix, self.obs_range, self.np_random, randinit=True)
        self.agent_layer = AgentLayer(self.X, self.Y, self.agents)

        self.num_targets = config['n_targets']
        self.targets = agent_utils.create_agents(self.num_targets, self.map_matrix, self.obs_range, self.np_random, randinit=True)
        self.target_layer = TargetLayer(self.X, self.Y, self.targets, self.map_matrix)

        self.num_jammers = config['n_jammers']
        self.jammers = jammer_utils.create_jammers(self.num_jammers, self.map_matrix, self.np_random, config['jamming_radius'])
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


    def draw_model_state(self):
        """
        Use pygame to draw environment map.
        REF: PettingZoo's pursuit example: PettingZoo/sisl/pursuit/pursuit_base.py
        """
        # !0 is building pixel flag - TODO: may need to change this?
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
        for i in range(self.target_layer.n_agents()):
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
        self.draw_agents_observations()

        self.draw_targets()
        self.draw_agents()
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
            
    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        
    
    def update(self):
        for i in range(self.target_layer.n_agents()):
            self.target_layer.move_agent(i, 1)
            
            

config_path = '/Users/hamishmacintosh/Uni Work/METR4911/Shared Repo/Shared-MARL-Env/config.yaml' 

map_processor = Environment(config_path)

map_processor.render()
#pygame.image.save(map_processor.screen, "/Users/hamishmacintosh/Desktop/environment_snapshot.png")
pygame.time.delay(10000)

