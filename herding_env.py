import numpy as np
import torch
from shepherd import Shepherd
from agent import Agent
from config import config
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EnvWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env, max_steps=400):
        # Retrieve the action space
        action_space = env.action_space
        assert isinstance(
            action_space, gym.spaces.Box
        ), "This wrapper only works with continuous action space (spaces.Box)"
        # Retrieve the max/min values
        self.low, self.high = action_space.low, action_space.high

        # We modify the action space, so all actions will lie in [-1, 1]
        env.action_space = gym.spaces.Box(
            low=-1, high=1, shape=action_space.shape, dtype=np.float32
        )
        self.max_steps = max_steps
        # Call the parent constructor, so we can access self.env later
        super(EnvWrapper, self).__init__(env)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.low + (0.5 * (scaled_action + 1.0) * (self.high - self.low))

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float,bool, bool, dict) observation, reward, final state? truncated?, additional informations
        """
        self.current_step += 1
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self.rescale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(rescaled_action)
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info
    
    def get_agent_positions(self):
        return self.env.get_agent_positions()
    
    def get_shepherd_position(self):
        return self.env.get_shepherd_position()

class HerdingEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Initialize state
        self.state = None
        self.done = False

        self.target = config["shepherd"]["target"]
        self.distance_weight = 1.0/20.0
        self.shepherd_weight = 1.0/20.0
        self.spread_weight = 1.0/20.0

        # Define action and observation spaces
        action_low = np.array([-100, -100, 0])
        action_high = np.array([100, 100, 2])
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(3,), dtype=np.float64)

        self.observation_space = spaces.Box(low=-100, high=100, shape=(6,), dtype=np.float64)

    
    def initialize_env(self, config):
        self.agents = [Agent(position=[np.random.uniform(0, 50), np.random.uniform(0, 50)],
                            heading=[np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                            delta=config["agents"]["disp_per_timestep"],
                            c=config["agents"]["attraction_weight"],
                            ra=config["agents"]["agent_interaction_dist"],
                            rhoa=config["agents"]["repulsion_weight_agents"],
                            rs=config["agents"]["shepherd_detection_dist"],
                            rhos=config["agents"]["repulsion_weight_shepherd"],
                            dH=config["agents"]["heading_inertia"],
                            noise=config["agents"]["angular_noise"],
                            num_neighbors=config["agents"]["neighbor_count"],
                            grazing_probability=config["agents"]["grazing_probability"]) for _ in range(config["environment"]["n_agents"])]
        self.shepherd = None
        self.shepherd = Shepherd(position=[np.random.uniform(-50, 0), np.random.uniform(-50, 0)],
                                    heading=[np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                                    max_speed=config["shepherd"]["disp_per_timestep"],
                                    ra=config["shepherd"]["group_distance_factor"]*config["agents"]["agent_interaction_dist"],
                                    dH=config["shepherd"]["heading_inertia"],
                                    n_agents=config["environment"]["n_agents"],
                                    noise=config["shepherd"]["angular_noise"])
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        for agent in self.agents:
            agent.position = np.random.uniform(-30, 30, 2)
            agent.heading = np.random.uniform(-1, 1, 2)
        self.shepherd.position = np.array([np.random.uniform(30, 50), np.random.uniform(30, 50)])
        # self.shepherd.position = np.array([-20.0, -20.0])

        samples = np.random.binomial(1, 0.5, 2)
        if samples[0] == 0:
            self.shepherd.position[0] = -self.shepherd.position[0]
        if samples[1] == 0:
            self.shepherd.position[1] = -self.shepherd.position[1]

        self.shepherd.heading = np.random.uniform(-1, 1, 2)

        self.done = False
        
        return self.get_state(), {}
    
    def get_state(self):        
        # Shepherd's position
        if self.shepherd is not None:
            shepherd_position = self.get_shepherd_position()  # Should be a tensor
        else:
            shepherd_position = np.zeros((2,))
        
        # Agent positions
        gcm = self.get_gcm()
        
        # Furthest agent's position
        furthest_agent_position = self.get_furthest_agent().position  # Assume this returns a tensor
        
        # Concatenate all information into a single tensor (numpy)
        state = np.concatenate([shepherd_position, gcm, furthest_agent_position])
        
        return state
    
    def step(self, action=None):
        for agent in self.agents:
            agent.update(self.agents, self.shepherd)
        if self.shepherd is not None:
            # action is three dimensional and contains goal position and speed
            self.shepherd.update_rl(goal_position_and_speed=action)

        reward = self.get_reward()
        state = self.get_state()
        self.done = bool(np.linalg.norm(self.get_gcm() - self.target) < 3 and self.get_agent_dist_std() < 5)
        # truncated = False
        
        return state, reward, self.done, False, {}
    
    # def step(self):
    #     for agent in self.agents:
    #         agent.update(self.agents, self.shepherd)
    #     if self.shepherd is not None:
    #         # action is three dimensional and contains goal position and speed
    #         self.shepherd.update_position(self.agents, np.array([0, 0]))

    #     reward = self.get_reward()
    #     state = self.get_state()
    #     self.done = np.linalg.norm(self.get_gcm() - self.target) < 10
    #     truncated = False

    #     return state, reward, self.done, truncated, {}
    
    def render(self):
        pass

    def close(self):
        pass

    def get_shepherd_position(self):
        return self.shepherd.position.copy()
    
    def get_agent_positions(self):
        return [agent.position.copy() for agent in self.agents].copy()
    
    def get_furthest_agent(self):
        return max(self.agents, key=lambda a: np.linalg.norm(a.position - self.get_gcm()))
    
    def get_gcm(self):
        return np.mean(np.stack([agent.position for agent in self.agents]), axis=0)
    
    # def get_headings(self):
    #     return [agent.heading for agent in self.agents], [shepherd.heading for shepherd in self.shepherds]
    
    # def get_neighbors(self):
    #     for agent in self.agents:
    #         agent.neighbors = agent.get_nearest_neighbors(self.agents)
    #     return [agent.neighbors for agent in self.agents]
    
    def get_reward(self):
        # depends on the distance of the herd wrt the target position and how spread the herd is
        GCM = self.get_gcm()
        rew = - self.distance_weight*np.linalg.norm(GCM - self.target) \
                - self.spread_weight*self.get_agent_dist_std() \
                - self.shepherd_weight*np.linalg.norm(self.get_shepherd_position() - GCM)
        return rew
    
    def get_agent_dist_std(self):
        GCM = self.get_gcm()
        return np.std([np.linalg.norm(agent.position - GCM) for agent in self.agents])
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    seed = 7
    env = HerdingEnv()
    env.initialize_env(config)
    env = EnvWrapper(env)
    obs, info = env.reset(seed=seed)
    # n_steps = 165
    agent_positions_log = []
    shepherd_positions_log = []
    done = False
    while not done:
        agent_positions_log.append(env.get_agent_positions())
        # print("agent positions: ", agent_positions_log)
        shepherd_positions_log.append(env.get_shepherd_position())
        # print("shepherd position: ", shepherd_positions_log)
        random_action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(random_action)
        # print(state)
        done = terminated or truncated
        # if done:
        #     break
    print("Simulation complete.")

    # agent_positions_log = np.array(agent_positions_log)
    # shepherd_positions_log = np.array(shepherd_positions_log)
    # plot positions
    fig, ax = plt.subplots()
    ax.scatter(np.array(agent_positions_log[0])[:, 0], np.array(agent_positions_log[0])[:, 1], c='blue', marker='*')
    ax.scatter(shepherd_positions_log[0][0], shepherd_positions_log[0][1], c='red', marker='*')
    for t in range(len(shepherd_positions_log)):
        ax.scatter(np.array(agent_positions_log[t])[:, 0], np.array(agent_positions_log[t])[:, 1], c='blue', alpha=0.1)
        ax.scatter(shepherd_positions_log[t][0], shepherd_positions_log[t][1], c='red', alpha=0.1)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect('equal', adjustable='datalim')
    plt.show()