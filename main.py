from config import config
from shepherd import Shepherd
from agent import Agent
import torch
import numpy as np

# Initialize agents
agents = [Agent(position=[np.random.uniform(0, 100), np.random.uniform(0, 100)],
                heading=[np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                delta=config["agents"]["disp_per_timestep"],
                c=config["agents"]["attraction_weight"],
                ra=config["agents"]["agent_interaction_dist"],
                rhoa=config["agents"]["repulsion_weight_agents"],
                rs=config["agents"]["shepherd_detection_dist"],
                rhos=config["agents"]["repulsion_weight_shepherd"],
                dH=config["agents"]["heading_inertia"],
                noise=config["agents"]["angular_noise"]) for _ in range(config["environment"]["n_agents"])]