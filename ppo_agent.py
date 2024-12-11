from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from herding_env import HerdingEnv, EnvWrapper
from config import config

env = HerdingEnv()
env.initialize_env(config)
env = Monitor(env)
env = EnvWrapper(env)
env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1, device='cpu')
# model = PPO.load("ppo_custom_env")
# model.set_env(env)

# Train the model
model.learn(total_timesteps=100_000)

# Save the model
model.save("ppo_custom_env")

# Load the model and test it
# model = PPO.load("ppo_custom_env")
# obs, info = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs, deterministic=True)
#     state, reward, terminated, truncated, info = env.step(action)
#     # env.render()
#     done = terminated or truncated