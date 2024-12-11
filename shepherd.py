import numpy as np

class Shepherd:
    def __init__(self, 
                 position, 
                 heading,
                 max_speed, 
                 ra, 
                 dH,
                 n_agents,
                 noise):
        self.position = position
        self.heading = heading
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
        
        GCM = np.mean(np.stack([agent.position for agent in agents]), axis=0)
        agent_close = any(np.linalg.norm(agent.position - self.position) < 3 * self.ra for agent in agents)
        furthest_agent = max(agents, key=lambda a: np.linalg.norm(a.position - GCM))
        max_dist = np.linalg.norm(furthest_agent.position - GCM)

        if max_dist >= self.f: # collecting
            # print('collecting')
            Pc = furthest_agent.position + self.collecting_margin*(furthest_agent.position - GCM) / (max_dist + 0.0001)
            goal_position = Pc
            
        else: # driving
            # print('driving')
            # diff_vect = GCM - target 
            # Pd = diff_vect 
            Pd = GCM + self.driving_margin*(GCM - target) / (np.linalg.norm(GCM - target) + 0.0001)
            goal_position = Pd

        if agent_close:
            speed = self.speed_reduced
        else:
            speed = self.max_speed

        self.heading = self.dH * self.heading + (goal_position - self.position) / (np.linalg.norm(goal_position - self.position) + 0.0001) + self.noise*np.random.randn(2)
        self.heading = self.heading / (np.linalg.norm(self.heading) + 0.0001)
        self.position += speed*self.heading

    def update_rl(self, goal_position_and_speed):
        goal_position = goal_position_and_speed[:2]
        speed = goal_position_and_speed[2]
        # speed = torch.clamp(speed, 0, self.max_speed)
        self.heading = self.dH * self.heading + (goal_position - self.position) / (np.linalg.norm(goal_position - self.position) + 0.0001) + self.noise*np.random.randn(2)
        self.heading = self.heading / (np.linalg.norm(self.heading) + 0.0001)
        self.position += speed*self.heading
        # print("shepherd position: ", self.position)