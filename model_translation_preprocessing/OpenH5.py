"""
    Converting the tensorflow DQN model provided by baselines to a keras model.
"""

import tensorflow as tf
import keras
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
import stream_generator as sg

import joblib
import os
import sys
import numpy as np

import chainer
import h5py



if __name__ == '__main__':
    load_Model_with_trained_variables(best_dqn_path)
pass
