import os
import numpy as np
import random
import yaml
import pickle
import agent_utils
import jammer_utils
import target_utils
import pygame
from skimage.transform import resize
from layer import AgentLayer, JammerLayer, TargetLayer
from gymnasium.utils import seeding
from Discrete_controller.agent_controller import DiscreteAgentController
from Continuous_controller.reward import calculate_continuous_reward
from gymnasium import spaces
from path_processor import PathProcessor
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from Continuous_autoencoder.autoencoder import EnvironmentAutoencoder
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.algorithms.algorithm import Algorithm

class Environment(MultiAgentEnv):
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
        self.comm_range = self.config['comm_range']
        self.use_task_allocation = self.config.get('use_task_allocation_with_continuous', False)

        self.seed_value = self.config.get('seed', None)
        if self.seed_value == "None":
            self.seed_value = None
        self._seed(self.seed_value)

        # Initialize autoencoders for all layers
        self.autoencoder = EnvironmentAutoencoder((self.obs_range, self.obs_range), encoded_dim=32)
        for layer in ["map_view", "agent", "target", "jammer"]:
            self.autoencoder.add_layer(layer, (self.obs_range, self.obs_range), encoded_dim=32)
            self.autoencoder.load(f'outputs/trained_autoencoder_{layer}.pth', layer)
            self.autoencoder.autoencoders[layer].eval()

        self.map_matrix = self.load_map()
        # Global state includes layers for map, agents, targets, and jammers
        self.global_state = np.zeros((self.D,) + self.map_matrix.shape, dtype=np.float32)          
        self.num_agents = self.config['n_agents']
        self.agent_type = self.config.get('agent_type', 'discrete')
        
        if self.agent_type == 'task_allocation':
            self.agent_paths = {agent_id: [] for agent_id in range(self.num_agents)}
            self.current_waypoints = {agent_id: None for agent_id in range(self.num_agents)}
        
        # Assumes static environment map - this is for task allocation
        self.path_processor = PathProcessor(self.map_matrix, self.X, self.Y)

        # Initialise environment 
        self.initialise_agents()
        self.initialise_targets()
        self.initialise_jammers()
        self.define_action_space()
        self.define_observation_space()

        # Set global state layers
        self.global_state[0] = self.map_matrix
        self.update_global_state()
        self.current_step = 0
        self.render_modes = render_mode
        self.screen = None
        pygame.init()


    def load_map(self):
        original_map = np.load(self.config['map_path'])[:, :, 0]
        original_map = original_map.transpose()
        original_x, original_y = original_map.shape
        self.X = int(original_x * self.map_scale)
        self.Y = int(original_y * self.map_scale)
        resized_map = resize(original_map, (self.X, self.Y), order=0, preserve_range=True, anti_aliasing=False)
        return (resized_map != 0).astype(int) # 0 for obstacles, 1 for free space


    def initialise_agents(self):
        agent_positions = self.config.get('agent_positions')
        self.agents = agent_utils.create_agents(self.num_agents, self.map_matrix, self.obs_range, self.np_random, 
                                                self.path_processor, agent_positions, agent_type=self.agent_type, randinit=True)
        self.agent_layer = AgentLayer(self.X, self.Y, self.agents)
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

    def initialise_targets(self):
        self.num_targets = self.config['n_targets']
        target_positions = self.config.get('target_positions')
        self.targets = target_utils.create_targets(self.num_targets, self.map_matrix, self.obs_range, self.np_random, 
                                                   self.path_processor, target_positions, randinit=True)
        self.target_layer = TargetLayer(self.X, self.Y, self.targets, self.map_matrix)


    def initialise_jammers(self):
        self.num_jammers = self.config['n_jammers']
        self.jammers = jammer_utils.create_jammers(self.num_jammers, self.map_matrix, self.np_random, self.config['jamming_radius'])
        self.jammer_layer = JammerLayer(self.X, self.Y, self.jammers)
        self.jammed_positions = set()


    def define_action_space(self):
        if self.agent_type == 'discrete':
            self.action_space = spaces.Discrete(len(self.agents[0].eactions))
        elif self.agent_type == 'task_allocation':
            self.action_space = spaces.Discrete((2 * self.agents[0].max_distance + 1) ** 2)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


    def define_observation_space(self):
        if self.agent_type == 'continuous':
            self.observation_space = spaces.Dict({
                "map": spaces.Box(low=-20, high=1, shape=(4, self.obs_range, self.obs_range), dtype=np.float32),
                "velocity": spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
                "goal": spaces.Box(low=-2000, high=2000, shape=(2,), dtype=np.float32)
        })
        elif self.agent_type == "discrete":
            self.observation_space = spaces.Dict({
                agent_id: spaces.Box(low=-20, high=1, shape=(4, self.obs_range, self.obs_range), dtype=np.float32)
                for agent_id in range(self.num_agents)})
        else: 
            self.observation_space = spaces.Dict({
                'local_obs': spaces.Box(low=-20, high=1, shape=(self.D, self.obs_range, self.obs_range), dtype=np.float32),
                'full_state': spaces.Box(low=-20, high=1, shape=(self.D, self.X, self.Y), dtype=np.float32)
            })

    def encode_observation(self, observations):
        encoded_observations = {}
        for agent_id, obs in observations.items():
            # Split the 'map' observation into individual layers and encode them
            map_layers = [obs['map'][i] for i in range(4)]
            encoded_map = []
            for i, layer in enumerate(["map_view", "agent", "target", "jammer"]):
                encoded_layer = self.autoencoder.encode_state({layer: map_layers[i]})
                encoded_map.append(encoded_layer[layer]['encoded'])
            encoded_observations[agent_id] = {
                    "encoded_map": np.stack(encoded_map),
                    "velocity": obs['velocity'],
                    "goal": obs['goal']
                }
        return encoded_observations


    def decode_observation(self, encoded_observations):
        decoded_observations = {}
        for agent_id, encoded_obs in encoded_observations.items():
            encoded_map = encoded_obs['encoded_map']
            decoded_map = []
            for i, layer in enumerate(["map_view", "agent", "target", "jammer"]):
                decoded_layer = self.autoencoder.decode_state({layer: {'encoded': encoded_map[i]}})
                decoded_map.append(decoded_layer[layer])
            decoded_observations[agent_id] = {
                "map": np.stack(decoded_map),
                "velocity": encoded_obs['velocity'],
                "goal": encoded_obs['goal']
            }
        return decoded_observations
    

    def reset(self, seed=None, options: dict = None):
        """ Reset the environment for a new episode"""
        info = {}
        super().reset(seed=self.seed_value if seed is None else seed)
        self._seed(self.seed_value if seed is None else seed)
        # Reset global state
        self.global_state.fill(0)
        self.global_state[0] = self.map_matrix # Uncomment above code if map_matrix is changed by sim

        self.path_processor = PathProcessor(self.map_matrix, self.X, self.Y) # This was added by hamish - will test with it

        self.initialise_agents()
        self.initialise_targets()
        self.initialise_jammers()
        
        self.target_layer.update()
        self.agent_layer.update()
        # Update layers in global state
        self.update_global_state()
        self.current_step = 0

        observations = {}
        for agent_id in range(self.num_agents):
            obs = self.safely_observe(agent_id)
            self.agents[agent_id].set_observation_state(obs)
            observations[agent_id] = {
                "map": obs['map'],
                "velocity": obs['velocity'],
                "goal": obs['goal']
            }
        
        encoded_observations = self.encode_observation(observations)
            
        return encoded_observations, info

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
        
        return local_states, rewards, terminated, truncated, info
    
    def regular_step(self, actions_dict):
        # Update target positions and layer state
        for i, target in enumerate(self.target_layer.targets):
            action = target.get_next_action()
            self.target_layer.move_targets(i, action)
        
        # Update agent positions and layer state based on the provided actions
        for agent_id, action in actions_dict.items():
            agent = self.agents[agent_id]
            #if there is a goal provided from task allocation 
            if self.use_task_allocation and agent_id in self.task_goals:
                agent.set_goal_area(self.task_goals[agent_id])

            current_direction = np.arctan2(agent.velocity[1], agent.velocity[0])
            desired_direction = np.arctan2(action[1], action[0])
            angle_diff = desired_direction - current_direction
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  

            # Check the angle change constraint
            if np.abs(angle_diff) > np.deg2rad(75):
                # Flag the angle change for reward penalty
                agent.change_angle = True
            else:
                agent.change_angle = False

            self.agent_layer.move_agent(agent_id, action)
            
            # Check if the agent touches any jammer and destroy it
            agent_pos = self.agent_layer.agents[agent_id].current_position()
            for jammer in self.jammers:
                if not jammer.get_destroyed() and jammer.current_position() == tuple(agent_pos):
                    jammer.set_destroyed()
                    self.update_jammed_areas()  # Update jammed areas after destroying the jammer

        self.target_layer.update()
        self.agent_layer.update()
        self.update_global_state()
        observations = self.update_observations()
        self.share_and_update_observations()
        self.jammer_layer.activate_jammers(self.current_step)
        self.update_jammed_areas()
        self.check_jammer_destruction()

        # Collect final local states and calculate rewards
        local_states, rewards = self.collect_local_states_and_rewards()
        encoded_observations = self.encode_observation(observations)
        decoded_obs = self.decode_observation(encoded_observations)
        
        self.current_step += 1
        
        terminated = {agent_id: self.is_episode_done() for agent_id in range(self.num_agents)}
        truncated = {agent_id: False for agent_id in range(self.num_agents)}
        info = {}

        terminated["__all__"] = self.is_episode_done()
        truncated["__all__"] = False
        np.set_printoptions(threshold=2000, suppress=True, precision=1, linewidth=2000)
    
        # print("Raw Observations")
        # for agent_id, obs in observations.items():
        #     print(f"Agent {agent_id}: {obs}")

        # # print("Encoded Observations")
        # encoded_obs = {}
        # for agent_id, obs in encoded_observations.items():
        #     encoded_obs[agent_id] = obs["encoded_map"]
        #     # print(f"Agent {agent_id}: {encoded_obs[agent_id]}")
        
        # print("Decoded Observations")
        # for agent_id, obs in decoded_obs.items():
        #     decoded_obs[agent_id] = obs["map"]
        #     print(f"Agent {agent_id}: {decoded_obs[agent_id]}")
        
        # If i'm collecting observations I have to return observations. Once the auto encoder is trained I can return encoded observation
        return encoded_observations, rewards, terminated, truncated, info
    

    def update_observations(self): #Alex had this one
        observations = {}
        for agent_id in range(self.num_agents):
            obs = self.safely_observe(agent_id)
            self.agent_layer.agents[agent_id].set_observation_state(obs)
            observations[agent_id] = obs
        return observations


    def collect_local_states_and_rewards(self):
        local_states = {}
        rewards = {}
        for agent_id in range(self.num_agents):
            agent = self.agent_layer.agents[agent_id]
            #local_states[agent_id] = agent.get_state()  # This returns the local_state
            local_states[agent_id] = {
                "map": agent.get_observation_state()["map"],
                "velocity": agent.velocity,
                "goal": agent.goal_area if self.use_task_allocation else np.zeros((2,))
            }
            
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
        # Example condition: end episode after a fixed number of steps
        max_steps = 500  # or any other logic to end the episode
        if self.current_step >= max_steps:
            return True
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
                pygame.display.set_caption("Search & Track Path Planning")
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

        # observation = pygame.surfarray.pixels3d(self.screen)
        # new_observation = np.copy(observation)
        # del observation
        # if self.render_modes == "human":
        #     pygame.event.pump()
        #     pygame.display.update()
        # elif mode == "human":
        #     pygame.display.flip()

        # return (new_observation,
        #     np.transpose(new_observation, axes=(1, 0, 2))
        #     if self.render_modes == "rgb_array"
        #     else None
        # )
        if mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose((1, 0, 2))  # Ensure consistent shape
        elif mode == "human":
            pygame.display.flip()
        return None


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
        obs = obs.transpose((0,2,1))
        #obs = np.clip(obs, self.observation_space[agent_id].low, self.observation_space[agent_id].high)
        obs = np.clip(obs, self.observation_space["map"].low, self.observation_space["map"].high)
        velocity = self.agents[agent_id].velocity
        goal_info = self.agents[agent_id].goal_area if self.use_task_allocation else np.zeros((2,))
        return {"map": obs, "velocity": velocity, "goal": goal_info}

    def collect_obs(self, agent_layer, agent_id):
        return self.collect_obs_by_idx(agent_layer, agent_id)

    def collect_obs_by_idx(self, agent_layer, agent_idx):
        # Initialize the observation array based on the agent type
        obs = np.full((self.global_state.shape[0], self.obs_range, self.obs_range), fill_value=-20, dtype=np.float32)
        # Get the current position of the agent
        xp, yp = agent_layer.get_position(agent_idx)
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

    #to share observations only 
    # def share_and_update_observations(self):
    #     """
    #     Updates each agent classes internal observation state and internal local (entire env) state.
    #     Will merge current observations of agents within communication range into each agents local state.
    #     This function should be run in the step function.
    #     """
    #     for i, agent in enumerate(self.agent_layer.agents):
    #         # safely_observe returns the current observation of agent i - but that should be called before this function
    #         current_obs = agent.get_observation_state()
    #         current_pos = agent.current_position()
    #         # agent.set_observation_state(current_obs)
    #         for j, other_agent in enumerate(self.agent_layer.agents):
    #             if i != j:
    #                 other_pos = other_agent.current_position()
    #                 agent_id = self.agent_name_mapping[agent]
    #                 other_agent_id = self.agent_name_mapping[other_agent]

    #                 if self.within_comm_range(current_pos, other_pos) and not self.is_comm_blocked(agent_id) and not self.is_comm_blocked(other_agent_id):
    #                     other_agent.update_local_state(current_obs, current_pos) #This is to observation only
    #                     agent.communicated = True 

    #this is to share full state
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
                        #other_agent.update_local_state(current_obs, current_pos) #This is to observation only
                        agent.update_local_state(other_agent.local_state, other_pos) #THIS CHANGE IS TO SHARE LOCAL STATE
                        agent.communicated = True 

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
        if seed is None:
            seed = np.random.randint(0, 2*32 - 1)
        self.np_random, seed_ = seeding.np_random(seed)
        np.random.seed(seed)
        random.seed(seed)

    def run_simulation(self, max_steps=10):
        running = True
        step_count = 0
        collected_data = []

        while running and step_count < max_steps:
            #print(f"Step: {step_count}")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Generate new actions for all agents in every step
            action_dict = {agent_id: agent.get_next_action() for agent_id, agent in enumerate(self.agents)}
            observations, rewards, terminated, truncated, self.info = self.step(action_dict)
            collected_data.append(observations)
            self.render()  
            step_count += 1

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                print("here")
                break

        pygame.image.save(self.screen, "outputs/environment_snapshot.png")
        self.reset()
        pygame.quit()

        return collected_data
    
    def run_simulation_with_policy(self, checkpoint_dir, params_path, max_steps=100, iteration=None):
        # Load the configuration from the params.pkl file
        with open(params_path, "rb") as f:
            config = pickle.load(f)

        # Register the custom environment
        register_env("custom_multi_agent_env", lambda config: Environment(config_path=config["config_path"], render_mode=config.get("render_mode", "human")))

        # Recreate the trainer with the loaded configuration
        trainer = PPO(config=config)

        # Restore the checkpoint from the checkpoint directory
        trainer.restore(checkpoint_dir)

        # # Get the policy from the trainer
        # policy = trainer.get_policy("policy_0") # This is for centralised learning - remove for decentralised 

        running = True
        step_count = 0
        collected_data = []

        while running and step_count < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action_dict = {}
            for agent_id in range(self.num_agents):
                obs = self.safely_observe(agent_id)
                encoded_obs = self.encode_observation({agent_id: obs})[agent_id]  # Encode observation
                obs = {
                    "encoded_map": encoded_obs["encoded_map"].reshape(1, 4, 32),  # Adjust the shape for the policy
                    "velocity": np.array(encoded_obs["velocity"]).reshape(1, -1),
                    "goal": np.array(encoded_obs["goal"]).reshape(1, -1)
                }
                policy = trainer.get_policy(f"policy_{agent_id}") # This is for decentralised learning - remove for centralised 
                action = policy.compute_single_action(obs)[0]
                action_dict[agent_id] = action

            observations, rewards, terminated, truncated, self.info = self.step(action_dict)
            collected_data.append(observations)
            self.render()
            step_count += 1

            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break
        
        if iteration is not None:
            map_filename = f"outputs/environment_snapshot_iteration_{iteration}.png"
        else:
            map_filename = "outputs/environment_snapshot.png"

        pygame.image.save(self.screen, map_filename)
        self.reset()
        pygame.quit()

        return collected_data