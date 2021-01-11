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
dqn_model = "model_DQN.h5"
ddqn_model = "model.npz"
pal_model = "PALmodel.npz"
best_dqn_model = "BestDQNPacman"
trying_h5 = "TryingH5.h5"
dqn_path = os.path.join(model_path,dqn_model)
ddqn_path = os.path.join(model_path,ddqn_model)
pal_path = os.path.join(model_path,pal_model)
best_dqn_path = os.path.join(model_path,best_dqn_model)
tryingH5_path = os.path.join(model_path,trying_h5)

def load_Model_with_trained_variables(load_path):

    # Keras Model
    hidden = 512
    initial_bias = tf.keras.initializers.Constant(0.1)
    
    inputs = Input(shape=(4, 84, 84))
    print("inputs are: ")
    print(inputs)
    
    x = Conv2D(filters=32, kernel_size=(8, 8), strides=4, padding='valid', data_format="channels_first", dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer=initial_bias, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, name='deepq/q_func/convnet/Conv')(inputs)
    x1 = Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu', padding='valid', data_format="channels_first", bias_initializer=initial_bias, name='deepq/q_func/convnet/Conv_1')(x)
    x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', padding='valid', data_format="channels_first", bias_initializer=initial_bias, name='deepq/q_func/convnet/Conv_2')(x1)
    conv_out = Flatten()(x2)
    print("conv_out is: ")
    print(tf.keras.backend.print_tensor(conv_out))
    print("x is: ")
    print(tf.keras.backend.print_tensor(x))
    
    action_out = Dense(hidden, activation='relu', name='deepq/q_func/action_value/fully_connected')(conv_out)
    print("action_out is: ")
    print(tf.keras.backend.print_tensor(action_out))
    action_scores = Dense(units = 9, name='deepq/q_func/action_value/fully_connected_1', activation='linear', use_bias=True, kernel_initializer="glorot_uniform", bias_initializer=initial_bias, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,)(action_out)  # num_actions in {4, .., 18}
    
    modelArchitecture = Model(inputs, action_scores)
    
    #try using this model with our h5 from training algorithms
    print("Before opening, model_path is: ")
    print(load_path)

    print("Weights are: ")
    weights = np.load(load_path)
    #32, 4, 8, 8
    #weigts['0/0/W'][range(1,8),1,1,1]
    print(weights['0/0/W'][1,1,1,1])
    print(weights['0/0/W'][1,1,2,1])
    print(weights['0/0/W'][1,1,1,2])
    print(weights['0/0/W'][2,1,1,1])
    #modelArchitecture.load_weights(load_path)
    #modelArchitecture.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = [metrics.categorical_accuracy])
#    for item in weights:
#        print(item)
#        print(weights[item])
#
#    for item in weights:
#        print("\n Item ")
#        print(item)
#        print(">>>>>>>>>>>>>>>>>>>>>>>>>")
#        print(item)
#        print(weights[item].ndim)
#        print(weights[item].size)
#
#    for item in weights:
#        print(item)
#
#    for layer in modelArchitecture.layers:
#        print(layer.name)
        
    np.set_printoptions(threshold=sys.maxsize)
#    modelArchitecture.summary()

#    print("Weights all at once are: ")
#    modelArchitecture.layers[0].get_weights()
#    modelArchitecture.layers[1].get_weights()
#    modelArchitecture.layers[2].get_weights()
#    modelArchitecture.layers[3].get_weights()
#    modelArchitecture.layers[4].get_weights()
#    modelArchitecture.layers[5].get_weights()
#    modelArchitecture.layers[6].get_weights()
#
    print("Weights layer-by-layer are: ")
    i = 0
    for layer in modelArchitecture.layers:
        layer.get_config()
        tempW = np.array(layer.get_weights())
        print("Size of weights array for layer " + str(i))
        print(tempW.size)
        print(tempW.ndim)
        print(tempW.shape)
        print("Original Weights for Layer " + str(i))
        print(tempW)
        i = i+1
    
    
    print("Now looking at imported model's weights~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    layer_1_weights_array = np.array(weights["0/0/W"])
    #try this way first
    item = layer_1_weights_array.transpose(2,3,1,0)
#    item = layer_1_weights_array.reshape((8,8,32,4),order = 'A')
    # sitem = layer_1_weights_array.reshape((8,8,4,32),order = 'A')
    print(item)
    print("............................................................................................................................................................................")
#    item2 = layer_1_weights_array.transpose(3,2,1,0)
#    print(item2)
    item3 = np.flipud(item2)
    print(item3)
    
#    break out and reshape arrays
    print("Layer 1:")
#    layer_1_weights_array = np.array(weights["0/0/W"])
#    layer_1_weights_array = np.transpose(np.array(weights["0/0/W"]))
#    print("Weights:")
#    print(layer_1_weights_array.size)
#    print(layer_1_weights_array.ndim)
#    print(layer_1_weights_array)
#    layer_1_bias_array = np.transpose(np.array(weights["0/0/b"]))
    layer_1_bias_array = np.array(weights["0/0/b"])
#    print("Bias:")
#    print(layer_1_bias_array.size)
#    print(layer_1_bias_array.ndim)
#    print(layer_1_bias_array)
#    layer_1_array = np.transpose(np.array([layer_1_weights_array, layer_1_bias_array]))
#    layer_1_array = np.array([layer_1_weights_array, layer_1_bias_array])
#    print("Layer 1 Array:")
#    print(layer_1_array.size)
#    print(layer_1_array.ndim)
#    print(layer_1_array)
    #set in corresponding layer
    print("Now layer 1 weights are: ")
    modelArchitecture.layers[1].set_weights([item, layer_1_bias_array])
#    print(modelArchitecture.layers[1].get_weights())
#
#    #break out and reshape arrays
    print("Layer 2:")
    layer_2_weights_array = np.array(weights["0/1/W"])
    item = layer_2_weights_array.transpose(2,3,1,0)
    item2 = layer_2_weights_array.transpose(3,2,1,0)
#    print("Weights:")
##    print(layer_2_weights_array)
    layer_2_bias_array = np.array(weights["0/1/b"])
#    layer_2_bias_array = np.array(weights["0/1/b"])
#    print("Bias:")
##    print(layer_2_bias_array)
#    layer_2_array = np.transpose(np.array([layer_2_weights_array, layer_2_bias_array]))
#    #set in corresponding layer
    modelArchitecture.layers[2].set_weights([item, layer_2_bias_array])
#
#    #break out and reshape arrays
    print("Layer 3:")
    layer_3_weights_array = np.array(weights["0/2/W"])
    item = layer_3_weights_array.transpose(2,3,1,0)
    item2 = layer_3_weights_array.transpose(3,2,1,0)
#    print("Weights:")
##    print(layer_3_weights_array)
##    layer_3_bias_array = np.transpose(np.array(weights["0/2/b"]))
    layer_3_bias_array = np.array(weights["0/2/b"])
#    print("Bias:")
##    print(layer_3_bias_array)
#    layer_3_array = np.transpose(np.array([layer_3_weights_array, layer_3_bias_array]))
#    #set in corresponding layer
    modelArchitecture.layers[3].set_weights([item, layer_3_bias_array])
#
#    #break out and reshape arrays
    print("Layer 5:")
    layer_5_weights_array = np.array(weights["0/3/W"])
    item2 = np.transpose(layer_5_weights_array)
    item = layer_5_weights_array.transpose(1,0)
    item3 = layer_5_weights_array.reshape(3136, 512)
    print("transpose 1,0: ")
    print(item)
    print("other layer 5 transpose")
    print(item2)
    print("other layer 5 transpose")
    print(item3)
##    print("Weights:")
##    print(layer_5_weights_array)
##    layer_5_bias_array = np.transpose(np.array(weights["0/3/b"]))
    layer_5_bias_array = np.array(weights["0/3/b"])
##    print("Bias:")
##    print(layer_5_bias_array)
#    layer_5_array = np.transpose(np.array([layer_5_weights_array, layer_5_bias_array]))
#    #set in corresponding layer
    modelArchitecture.layers[5].set_weights([item3, layer_5_bias_array])
#
#    #break out and reshape arrays
    print("Layer 6:")
    layer_6_weights_array = np.array(weights["1/W"])
    item2 = np.transpose(layer_6_weights_array)
    item3 = layer_6_weights_array.reshape(512,9)
    item = layer_6_weights_array.transpose(1,0)
##    print("Weights:")
##    print(layer_6_weights_array)
##    layer_6_bias_array = np.transpose(np.zeros(9))
    layer_6_bias_array = np.zeros(9)
##    print("Bias:")
##    print(layer_6_bias_array)
#    layer_6_array = np.transpose(np.array([layer_6_weights_array, layer_6_bias_array]))
#    #set in corresponding layer
    modelArchitecture.layers[6].set_weights([item3, layer_6_bias_array])
#
#
#    #save as H5
    modelArchitecture.save(load_path + '.h5')
    
    return

if __name__ == '__main__':
    load_Model_with_trained_variables(best_dqn_path)
pass
