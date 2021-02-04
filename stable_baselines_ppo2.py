# TODO: alphabetize imports 
import gym
from collections import OrderedDict
import os
import tensorflow as tf
import pandas as pd
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN, A2C, PPO2
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from callback_a2c import CustomCallbackA
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import make_vec_env
import os, datetime
import argparse


# get rid of distracting TF errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# create folder and subfolders for data
dir = 'ppo2_data_100K' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'
os.makedirs(dir)
subfolder = os.path.join(dir, 'screen')
os.makedirs(subfolder)

# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment, equivalent to:
# env = make_vec_env('MountainCarContinuous-v0', n_envs=1, monitor_dir=log_dir)
env = make_atari('MsPacmanNoFrameskip-v4')
actions = make_atari('MsPacmanNoFrameskip-v4').unwrapped.get_action_meanings()
env = Monitor(env, log_dir)
env = DummyVecEnv([lambda: env])

# parse flags
parser = argparse.ArgumentParser()
parser.add_argument('--lives', help='env has lives', action='store_true', default=False)
args = parser.parse_args()
isLives = args.lives

# set num timesteps
# num_steps = int(10000 * 0.9984)
# num_steps = int(num_steps/0.9984)
num_steps = 99840
print("num steps ", num_steps)
# define callback object
step_callback = CustomCallbackA(0, actions, env,  num_steps, dir, isLives, make_atari('MsPacmanNoFrameskip-v4'))


# train:
# model = DQN(CnnPolicy, env, verbose=1)

# use pretrained model:
# model = DQN.load("deepq_pacman_300K")
# model = DQN.load("deepq_pacman_random")
# model.set_env(env)


# learning_rate set to 0 means it will act as a predict function
# model.learn(total_timesteps=num_steps, callback = step_callback) # 25000

# save model:
# model.save("deepq_pacman_300K")

# ppo2
model = PPO2('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=num_steps, callback=step_callback)
model.save("ppo2_pacman_100k")
print("done!")
