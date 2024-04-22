import numpy as np
import yaml
import matplotlib.pyplot as plt
import random
from pettingzoo.sisl.pursuit.utils import agent_utils
from layer import AgentLayer, JammerLayer, TargetLayer
from gymnasium.utils import seeding
import jammer_utils
import heapq

class Environment:
    def __init__(self, config_path, render_mode=None):
        # Load configuration from YAML
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Initialize from config
        self.X = config['grid_size']['X']
        self.Y = config['grid_size']['Y']
        self.D = config['grid_size']['D']
        self.obs_range = config['obs_range']
        self.seed = config['seed']
        self._seed(self.seed)
        
        # Load the map
        self.map_matrix = np.load(config['map_path'])[:, :, 0] 
        self.obstacle_map = np.load('city_image_1.npy')
        print("Shape of map_matrix:", self.map_matrix.shape)
        self.resolution = np.array(config['resolution'])
        
        # Global state includes layers for map, agents, targets, and jammers
        self.global_state = np.zeros((self.D,) + self.map_matrix.shape, dtype=np.float32)
        
        # Initialize agents, targets, jammers
        self.num_agents = config['n_agents']
        self.agents = agent_utils.create_agents(self.num_agents, self.map_matrix, self.obs_range, self.np_random)
        self.agent_layer = AgentLayer(self.X, self.Y, self.agents)

        self.num_targets = config['n_targets']
        self.targets = agent_utils.create_agents(self.num_targets, self.map_matrix, self.obs_range, self.np_random)
        self.target_layer = TargetLayer(self.X, self.Y, self.targets, self.map_matrix)

        self.num_jammers = config['n_jammers']
        # Created jammers at random positions - TODO: Update this to position jammers from config file
        self.jammers = jammer_utils.create_jammers(self.num_jammers, self.map_matrix, self.np_random, config['jamming_radius'])
        self.jammer_layer = JammerLayer(self.X, self.Y, self.jammers)
    
    def mark_obstacles(self):
        # assume the obstacles are marked by a non-zero value
        obstacles = self.obstacle_map != 0
        return obstacles

    def draw_map(self):
        # Plot the map with obstacles and overlay the grid
        plt.figure(figsize=(10, 6))

        # Overlay grid
        grid_obstacle_map = self.map_matrix
        plt.imshow(grid_obstacle_map, cmap='gray', alpha=0.5)

        # Mark targets
        for i in range(self.target_layer.n_agents()):
            #x, y = self.target_layer.get_position(i)
            target = self.add_random_targets()
            plt.scatter(target[1], target[0], s=100, c='yellow', marker='o')  # x and y are reversed in plt.scatter
            self.target_layer.set_position(i, target[0], target[1])

        # Mark jammers
        #for i in range(self.jammer_layer.n_agents()):
           # x, y = self.jammer_layer.get_position(i)
            #plt.scatter(100, 100, s=100, c='blue', marker='o')  # x and y are reversed in plt.scatter

        # Mark agents
        for i in range(self.agent_layer.n_agents()):
            x, y = self.agent_layer.get_position(i)
            plt.scatter(x, y, s=100, c='red', marker='o')  # x and y are reversed in plt.scatter

        plt.show()

    def add_random_targets(self):
        obstacles = self.mark_obstacles()
        potential_locations = np.argwhere(obstacles == True) 
        # Randomly select from the potential locations
        target = random.choice(potential_locations)
        return target
    
            
    def _seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
    
    def update(self):
        for i in range(self.target_layer.n_agents()):
            self.target_layer.move_agent(i, 1)
            y,x = self.target_layer.get_position(i)
            plt.scatter(x, y, s=10, c='yellow', marker='o') 
            

config_path = 'config.yaml' 
map_processor = Environment(config_path)

#map_processor.add_random_targets(3)
plt.ion()
map_processor.draw_map()
for i in range(100):
    map_processor.update()
plt.pause(5)
plt.ioff()
