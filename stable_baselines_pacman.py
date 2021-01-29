# TODO: alphabetize imports 
import gym
from collections import OrderedDict
import os
import tensorflow as tf
import pandas as pd
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from callback import CustomCallback
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import make_vec_env
import os, datetime
import argparse

# create folder and subfolders for data
dir = 'data_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'
os.makedirs(dir)
subfolder = os.path.join(dir, 'screen')
os.makedirs(subfolder)


env = make_atari('MsPacmanNoFrameskip-v4')
# env = make_atari('SpaceInvadersNoFrameskip-v4')

# get rid of distracting TF errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--lives', help='env has lives', action='store_true', default=False)
args = parser.parse_args()
isLives = args.lives
# set num timesteps
num_steps = 40

# define callback object
step_callback = CustomCallback(0,env.unwrapped.get_action_meanings(), env,  num_steps, dir, isLives)


# TODO: omit this and just pass in previously trained models
# learning_rate set to 0 means it will act as a predict function
# this:
# model = DQN(CnnPolicy, env, verbose=1)
# model = DQN.load("MsPacmanNoFrameSkip-v4.pkl")
# "train" aka run prediction model with callback
model = DQN.load("deepq_pacman_random")
model.set_env(env)
model.learn(total_timesteps=num_steps, callback = step_callback) # 25000
# model.save("deepq_pacman_callback")

# Moved most functions to callback.py