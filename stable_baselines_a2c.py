# TODO: alphabetize imports 
import gym
from collections import OrderedDict
import os
import tensorflow as tf
import pandas as pd
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN, A2C
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from callback_a2c import CustomCallback
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import make_vec_env
import os, datetime
import argparse

# create folder and subfolders for data
dir = 'ac2_data_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'
os.makedirs(dir)
subfolder = os.path.join(dir, 'screen')
os.makedirs(subfolder)


og_env = make_atari('MsPacmanNoFrameskip-v4')
actions = og_env.unwrapped.get_action_meanings()
env = DummyVecEnv([lambda:og_env])
# env = make_atari('SpaceInvadersNoFrameskip-v4')

# get rid of distracting TF errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--lives', help='env has lives', action='store_true', default=False)
args = parser.parse_args()
isLives = args.lives
# set num timesteps
num_steps = 100

# define callback object
step_callback = CustomCallback(0, actions, env,  num_steps, dir, isLives, og_env)


# train:
# model = DQN(CnnPolicy, env, verbose=1)

# use pretrained model:
# model = DQN.load("deepq_pacman_300K")
# model = DQN.load("deepq_pacman_random")
#model.set_env(env)


# learning_rate set to 0 means it will act as a predict function
# model.learn(total_timesteps=num_steps, callback = step_callback) # 25000

# save model:
# model.save("deepq_pacman_300K")

# a2c
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=num_steps, callback = step_callback)
model.save("a2c_pacman_test")

