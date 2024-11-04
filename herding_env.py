import numpy as np
import torch
from shepherd import Shepherd
from agent import Agent
from config import config

class HerdingEnv:
    def __init__(self, config):
        self.agents = [Agent(position=[np.random.uniform(0, 100), np.random.uniform(0, 100)], # spawning in upper right quadrant
                                    heading=[np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                                    delta=config["agents"]["disp_per_timestep"],
                                    c=config["agents"]["attraction_weight"],
                                    ra=config["agents"]["agent_interaction_dist"],
                                    rhoa=config["agents"]["repulsion_weight_agents"],
                                    rs=config["agents"]["shepherd_detection_dist"],
                                    rhos=config["agents"]["repulsion_weight_shepherd"],
                                    dH=config["agents"]["heading_inertia"],
                                    noise=config["agents"]["angular_noise"]) for _ in range(config["environment"]["n_agents"])]
        
        self.shepherds = [Shepherd(position=[np.random.uniform(0, -100), np.random.uniform(0, -100)],
                                    heading=[np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                                    max_speed=config["shepherd"]["disp_per_timestep"],
                                    ra=config["shepherd"]["group_distance_factor"]*config["agents"]["shepherd_detection_dist"],
                                    dH=config["shepherd"]["heading_inertia"],
                                    n_agents=config["environment"]["n_agents"],
                                    noise=config["shepherd"]["angular_noise"]) for _ in range(config["environment"]["n_shepherds"])]
    
    def update(self):
        for agent in self.agents:
            agent.update(self.agents, self.shepherds[0])
        for shepherd in self.shepherds:
            shepherd.update_position(self.agents, torch.tensor([50, 50], dtype=torch.float32))
    
    def get_positions(self):
        return [agent.position for agent in self.agents], [shepherd.position for shepherd in self.shepherds]
    
    def get_headings(self):
        return [agent.heading for agent in self.agents], [shepherd.heading for shepherd in self.shepherds]
    
    def get_neighbors(self):
        for agent in self.agents:
            agent.neighbors = agent.get_nearest_neighbors(self.agents)
        return [agent.neighbors for agent in self.ag