import torch
import numpy as np
# import shepherd

class Agent:
    def __init__(self, 
                 position, 
                 heading, 
                 delta, 
                 c, 
                 ra, 
                 rhoa, 
                 rs, 
                 rhos, 
                 dH, 
                 noise,
                 grazing_probability):
        self.position = torch.tensor(position, dtype=torch.float32)
        self.heading = torch.tensor(heading, dtype=torch.float32)
        self.delta = delta   # agent displacement per time step
        self.c = c           # Weight for attraction to neighbors
        self.ra = ra         # threshold for agent repulsion
        self.rhoa = rhoa     # Weight for repulsion from agents
        self.rs = rs         # threshold for shepherd repulsion
        self.rhos = rhos     # Weight for repulsion from shepherd
        self.dH = dH         # Heading Inertia term
        self.noise = noise   # Noise term
        self.neighbors = []
        self.grazing_prob = grazing_probability

    def update(self, agents, shepherd=None):
        """ Update the agent's heading and position based on neighbors and the shepherd's position. """
        
        attraction_to_neighbors = torch.zeros(2, dtype=torch.float32)
        repulsion_from_shepherd = torch.zeros(2, dtype=torch.float32)
        repulsion_from_agents = self.get_repulsion_from_agents(agents) # get ^Ra (repulsion from other agents within distance ra)

        if shepherd is not None:
            shepherd_dist = torch.norm(shepherd.position - self.position)
            if shepherd_dist < self.rs:
                # get ^C (attraction to n nearest neighbors)
                attraction_to_neighbors = self.get_attraction_to_neighbors(agents)

                # get ^Rs (repulsion from shepherd if within distance rs)
                repulsion_from_shepherd = self.get_repulsion_from_shepherd(shepherd, shepherd_dist)
        
        # New heading as a weighted combination of C, Ra, and Rs
        self.heading = self.dH * self.heading + self.c * attraction_to_neighbors + self.rhoa * repulsion_from_agents + self.rhos * repulsion_from_shepherd + self.noise * torch.randn(2)
        self.heading = self.heading / torch.norm(self.heading)  # Normalize heading

        if shepherd is not None and shepherd_dist < self.rs:
            self.position += self.delta*self.heading  # Update position
        elif np.random.uniform() < self.grazing_prob:
            self.position += self.delta*self.heading

    def get_attraction_to_neighbors(self, agents):
        """ get attraction to the n nearest neighbors. """
        # Select n nearest neighbors and compute LCM (here taken as mean position)
        neighbors = self.get_nearest_neighbors(agents)
        C = torch.mean(torch.stack([neighbor.position for neighbor in neighbors]), dim=0) - self.position
        return C/torch.norm(C)
    
    def get_repulsion_from_shepherd(self, shepherd, shepherd_dist):
        """ get repulsion from the shepherd if within distance rs. """
        Rs = (shepherd.position - self.position) / shepherd_dist # repulsion direction (normalized)
        return Rs

    def get_repulsion_from_agents(self, agents):
        """ get repulsion from other agents within distance ra. """
        Ra = torch.zeros(2, dtype=torch.float32)
        for agent in agents:
            if agent != self:
                dist = torch.norm(agent.position - self.position)
                if dist < self.ra:
                    Ra -= (agent.position - self.position) / dist
        return Ra/torch.norm(Ra)

    def get_nearest_neighbors(self, agents, n=5):
        """ Find the n nearest neighbors among agents. """
        distances = [(agent, torch.norm(agent.position - self.position)) for agent in agents if agent != self]
        distances.sort(key=lambda x: x[1])
        return [agent for agent, dist in distances[:n]]