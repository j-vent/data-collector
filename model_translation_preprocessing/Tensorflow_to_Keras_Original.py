"""
    Converting the tensorflow DQN model provided by baselines to a keras model.
"""

import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model
import coloredlogs, logging
import argparse

import joblib
import os
import sys
import numpy as np


def load_Model_with_trained_variables(load_path, args):
    # Load Checkpoint
    tf.reset_default_graph()
    dictOfWeights = {}; dictOfBiases = {}
    with tf.Session() as sess:
        col = joblib.load(os.path.expanduser(load_path))
        i = 0
        for var in col:
            i = i+1
            if type(var) is np.ndarray:
                if args.verbose is True:
                    logger.info(str(i) + " " + var + str(col[var].shape))
            else:
                if args.verbose is True:
                    logger.warning(str(i) + " " + var + " no ndarray")
            if "target" not in var:
                if "weights" in var:
                    dictOfWeights[var] = col[var]
                if "biases" in var:
                    dictOfBiases[var] = col[var]
            pass
    np.set_printoptions(threshold=sys.maxsize)
    # Keras Model
    hidden = 256
    ThirdLastBiases = dictOfBiases['deepq/q_func/action_value/fully_connected_1/biases:0']
    num_actions = ThirdLastBiases.size
    dueling = True
    inputs = Input(shape=(84, 84, 4))
    # ConvLayer
    x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='SAME', name='deepq/q_func/convnet/Conv')(inputs)
    x1 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='SAME', name='deepq/q_func/convnet/Conv_1')(x)
   
    x2 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='SAME', name='deepq/q_func/convnet/Conv_2')(x1)
    
    conv_out = Flatten()(x2)
    # Action values
    action_out = conv_out
    action_out = Dense(hidden, activation='relu', name='deepq/q_func/action_value/fully_connected')(action_out)
    action_scores = Dense(num_actions, name='deepq/q_func/action_value/fully_connected_1', activation='linear')(action_out)  # num_actions in {4, .., 18}
    # State values
    if dueling:
        state_out = conv_out
        state_out = Dense(hidden, activation='relu', name='deepq/q_func/state_value/fully_connected')(state_out)
        state_score = Dense(1, name='deepq/q_func/state_value/fully_connected_1')(state_out)
    # Finish model
    model = Model(inputs, [action_scores, state_score])
    
    #for layer in model.layers: print(layer.get_config(), layer.get_weights())
    modelActionPart = Model(inputs, action_scores)
    modelStatePart  = Model(inputs, state_score)

    # Load weights
    for layer in model.layers:
        if layer.name + "/weights:0" in dictOfWeights:
            newWeights = dictOfWeights[layer.name + "/weights:0"]
            newBiases = dictOfBiases[layer.name + "/biases:0"]
            # set_weights(list of ndarrays with 2 elements: 0->weights 1->biases)
            layer.set_weights([newWeights, newBiases])
            if args.verbose is True:
                logger.info("Found and applied values for layer " + layer.name)
        else:
            if args.verbose is True:
                logger.warning("No corresponding layer found for" + layer.name)
    
    return model, modelStatePart, modelActionPart




if __name__ == '__main__':

    #get rid of distracting TF errors
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    
    #Set up logger so we have pretty and informative logs
    logger = logging.getLogger()
    coloredlogs.install(level='DEBUG', fmt='%(asctime)s,%(msecs)03d %(filename)s[%(process)d] %(levelname)s %(message)s')

    logger.setLevel(logging.DEBUG)
    # Some examples.
    ''' logger.debug("this is a debugging message")
        logger.info("this is an informational message")
        logger.warning("this is a warning message")
        logger.error("this is an error message")
        logger.critical("this is a critical message")'''

    #This is essantially a stand-alone program, so it will have it's own args
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-input-location', type=str, default='../models', help='path to the folder, including name, where model to be converted is at')
    parser.add_argument('--model-input-name', type=str, default='freewayInit_copy', help='Name of model saved in model folder which you want to convert')
    parser.add_argument('--model-output-location', type=str, default='../models', help='path to the folder you wish to save in, including name, where model to be converted is at')
    parser.add_argument('--model-output-name', type=str, default='freewayInit_copy', help='Name of model you want it to have once converted')
    parser.add_argument('--verbose', action='store_true', help='Output information for debugging etc.')
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    # combine path to folder and model name
    load_path = os.path.join(args.model_input_location,args.model_input_name)
    # call to main function which creates a network and loads the weights
    (model, modelStatePart, modelActionPart) = load_Model_with_trained_variables(load_path, args)
    
    if args.verbose:
        # output info about the model
        logger.debug(model.summary())
        logger.debug(modelStatePart.summary())
        logger.debug(modelActionPart.summary())

    # make the output path
    output_path = os.path.join(args.model_output_location,args.model_output_name)
    # save model to the output path
    modelActionPart.save(output_path + '.h5')

pass
