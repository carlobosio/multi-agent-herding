from stable_baselines3 import PPO
from herding_env import HerdingEnv, EnvWrapper
from config import config
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env

env = HerdingEnv()
env.initialize_env(config)
# check_env(env)
env = EnvWrapper(env, max_steps=200)

# seed = 42
# np.random.seed(seed)
# Load the model and test it
# n_steps = 100
model = PPO.load("ppo_custom_env")
obs, info = env.reset()
agent_positions_log = []
shepherd_positions_log = []
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    # action = np.array([50, 50, 1.5])
    # print(action)
    # action = action*100
    agent_positions_log.append(env.get_agent_positions())
    shepherd_positions_log.append(env.get_shepherd_position())
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    # env.render()
    # if done:
    #     obs, info = env.reset()

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