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

model_path = './models'
#An .npz file output from chainerRL DQN
best_dqn_model = "BestDQNPacman"
best_dqn_path = os.path.join(model_path,best_dqn_model)

def load_Model_with_trained_variables(load_path):

    # Keras Model
    hidden = 512
    #bias initializer to match the chainerRL one
    initial_bias = tf.keras.initializers.Constant(0.1)
    
    #matches default "channels_last" data format for Keras layers
    inputs = Input(shape=(84, 84, 4))
    
    #First call to Conv2D including all defaults for easy reference
    x = Conv2D(filters=32, kernel_size=(8, 8), strides=4, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer=initial_bias, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name='deepq/q_func/convnet/Conv')(inputs)
    x1 = Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu', padding='valid', bias_initializer=initial_bias, name='deepq/q_func/convnet/Conv_1')(x)
    x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', padding='valid', bias_initializer=initial_bias, name='deepq/q_func/convnet/Conv_2')(x1)
    #Flatten for move to linear layers
    conv_out = Flatten()(x2)
    
    action_out = Dense(hidden, activation='relu', name='deepq/q_func/action_value/fully_connected')(conv_out)
    action_scores = Dense(units = 9, name='deepq/q_func/action_value/fully_connected_1', activation='linear', use_bias=True, kernel_initializer="glorot_uniform", bias_initializer=initial_bias, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,)(action_out)  # num_actions in {4, .., 18}
    
    #Now create model using the above-defined layers
    modelArchitecture = Model(inputs, action_scores)
    
    #Pull weights from .npz file from ChainerRL
    weights = np.load(load_path)
    
    #Set so all items in weights can be seen in any output
    np.set_printoptions(threshold=sys.maxsize)
    
    # Layer 1 (layer 0 is empty)
    # shape = 32, 4, 8, 8
    layer_1_weights_array = np.array(weights["0/0/W"])
    #reshape to match Keras model's layer shape
    item = layer_1_weights_array.transpose(2,3,1,0)
    layer_1_bias_array = np.array(weights["0/0/b"])
    print("Bias:")
    print(layer_1_bias_array)
    print("Trasposed bias: ")
    print(layer_1_bias_array.transpose())
    print("Now layer 1 weights are: ")
    modelArchitecture.layers[1].set_weights([item, layer_1_bias_array])

    # Layer 2
    layer_2_weights_array = np.array(weights["0/1/W"])
    item = layer_2_weights_array.transpose(2,3,1,0)
    layer_2_bias_array = np.array(weights["0/1/b"])
    print("Layer 2 bias: ")
    print(layer_2_bias_array)
    modelArchitecture.layers[2].set_weights([item, layer_2_bias_array])

    # Layer 3
    layer_3_weights_array = np.array(weights["0/2/W"])
    item = layer_3_weights_array.transpose(2,3,1,0)
    layer_3_bias_array = np.array(weights["0/2/b"])
    print("Layer 3 bias: ")
    print(layer_3_bias_array)
    modelArchitecture.layers[3].set_weights([item, layer_3_bias_array])
        
    # Layer 5
    layer_5_weights_array = np.array(weights["0/3/W"])
    item3 = layer_5_weights_array.reshape(3136, 512)
    layer_5_bias_array = np.array(weights["0/3/b"])
    print("Layer 5 bias: ")
    print(layer_5_bias_array)
    modelArchitecture.layers[5].set_weights([item3, layer_5_bias_array])

    # Layer 6
    layer_6_weights_array = np.array(weights["1/W"])
    item3 = layer_6_weights_array.reshape(512,9)
    layer_6_bias_array = np.zeros(9)
    print("Layer 6 bias: ")
    print(layer_6_bias_array)
    modelArchitecture.layers[6].set_weights([item3, layer_6_bias_array])

#    #save as H5
    modelArchitecture.save(load_path + '.h5')
    
    return

if __name__ == '__main__':
    load_Model_with_trained_variables(best_dqn_path)
pass
