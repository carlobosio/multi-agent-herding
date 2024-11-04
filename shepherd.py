import torch

class Shepherd:
    def __init__(self, 
                 position, 
                 heading,
                 max_speed, 
                 ra, 
                 dH,
                 n_agents,
                 noise):
        self.position = torch.tensor(position, dtype=torch.float32)
        self.heading = torch.tensor(heading, dtype=torch.float32)
        self.ra = ra
        self.dH = dH
        self.collecting_margin = self.ra
        self.driving_margin = n_agents**(0.5)*self.ra
        self.max_speed = max_speed
        self.speed_reduced = 0.3*self.ra
        self.f = self.ra * n_agents**(2/3)
        self.noise = noise

    def update_position(self, agents, target):
        """ Update the shepherd's behavior based on agent positions. """
        GCM = torch.mean(torch.stack([agent.position for agent in agents]), dim=0)  # Global center of mass of agents
        agent_close = torch.any(torch.stack([torch.norm(agent.position - self.position) < 3*self.ra for agent in agents]))
        furthest_agent = max(agents, key=lambda a: torch.norm(a.position - GCM))
        max_dist = torch.norm(furthest_agent.position - GCM)

        if max_dist >= self.f: # collecting
            Pc = GCM + (max_dist + self.collecting_margin)*(furthest_agent.position - GCM) / torch.norm(furthest_agent.position - GCM)
            goal_position = Pc
            
        else: # driving
            Pd = GCM + self.driving_margin*(GCM - target) / torch.norm(GCM - target)
            goal_position = Pd

        if agent_close:
            speed = self.speed_reduced
        else:
            speed = self.max_speed

        self.heading = self.dH * self.heading + (goal_position - self.position) / torch.norm(goal_position - self.position) + self.noise*torch.randn(2)
        self.heading = self.heading / torch.norm(self.heading)
        self.position += speed*self.heading