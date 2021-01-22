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
# env = make_atari('BreakoutNoFrameskip-v4')
env = make_atari('MsPacmanNoFrameskip-v4')

#get rid of distracting TF errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# set num timesteps
num_steps = 10

# define callback object
step_callback = CustomCallback(0,env.unwrapped.get_action_meanings(), env,  num_steps)


# TODO: omit this and just pass in previously trained models
# learning_rate set to 0 means it will act as a predict function
model = DQN(CnnPolicy, env, verbose=1, learning_rate=0)
# "train" aka run prediction model with callback
model.learn(total_timesteps=num_steps, callback = step_callback) # 25000
model.save("deepq_pacman_callback")

# Moved most functions to callback.py