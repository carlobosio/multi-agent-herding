import time
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, A2C

from stable_baselines3.common.evaluation import evaluate_policy

from herding_env import HerdingEnv, EnvWrapper
from config import config

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = HerdingEnv()
        env.initialize_env(config)
        env = EnvWrapper(env)
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    n_procs = 8
    NUM_EXPERIMENTS = 3  # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
    TRAIN_STEPS = 100000
    # Number of episodes for evaluation
    EVAL_EPS = 20
    ALGO = A2C

    # We will create one environment to evaluate the agent on
    eval_env = HerdingEnv()
    eval_env.initialize_env(config)
    eval_env = EnvWrapper(eval_env)
    eval_env = Monitor(eval_env)

    # reward_averages = []
    reward_std = []
    training_times = []

    print(f"Running for n_procs = {n_procs}")
    if n_procs == 1:
        # if there is only one process, there is no need to use multiprocessing
        env = HerdingEnv()
        env.initialize_env(config)
        env = EnvWrapper(env)
        train_env = DummyVecEnv([lambda: env])
    else:
        # Here we use the "fork" method for launching the processes, more information is available in the doc
        # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
        train_env = SubprocVecEnv(
            [make_env(i) for i in range(n_procs)],
            start_method="spawn",
        )

    rewards = []
    times = []
    model = []

    for experiment in range(NUM_EXPERIMENTS):
        # it is recommended to run several experiments due to variability in results
        train_env.reset()
        model.append(ALGO("MlpPolicy", train_env, verbose=0, device='cpu'))
        start = time.time()
        model[-1].learn(total_timesteps=TRAIN_STEPS)
        times.append(time.time() - start)
        mean_reward, _ = evaluate_policy(model[-1], eval_env, n_eval_episodes=EVAL_EPS)
        rewards.append(mean_reward)
    # Important: when using subprocesses, don't forget to close them
    # otherwise, you may have memory issues when running a lot of experiments
    train_env.close()
    reward_average = np.mean(rewards)
    reward_std = np.std(rewards)
    training_times = np.mean(times)

    # select model with highes reward and save it
    model = model[np.argmax(rewards)]
    model.save("ppo_parallel_env")

    # print reward and training time for each experiment
    print(f"Rewards: {rewards}")
    print(f"Stds: {reward_std}")
    print(f"Training times: {times}")
    print()
    print(f"Average reward: {reward_average} +/- {reward_std}")
    print(f"Training time: {training_times} s")

    # def plot_training_results(training_steps_per_second, reward_averages, reward_std):
    #     """
    #     Utility function for plotting the results of training

    #     :param training_steps_per_second: List[double]
    #     :param reward_averages: List[double]
    #     :param reward_std: List[double]
    #     """
    #     plt.figure(figsize=(9, 4))
    #     plt.subplots_adjust(wspace=0.5)
    #     plt.subplot(1, 2, 1)
    #     plt.errorbar(
    #         PROCESSES_TO_TEST,
    #         reward_averages,
    #         yerr=reward_std,
    #         capsize=2,
    #         c="k",
    #         marker="o",
    #     )
    #     plt.xlabel("Processes")
    #     plt.ylabel("Average return")
    #     plt.subplot(1, 2, 2)
    #     plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
    #     plt.xticks(range(len(PROCESSES_TO_TEST)), PROCESSES_TO_TEST)
    #     plt.xlabel("Processes")
    #     plt.ylabel("Training steps per second")