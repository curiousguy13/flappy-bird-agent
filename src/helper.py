
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.signal
import cv2      

#change TRAINING to False and TESTING to True for testing
TRAINING=True
#Will Load model if TESTING = TRUE 
TESTING=False
QUIET=True
LOAD_MODEL = False
TRAINING_EPISODES=6000
TESTING_EPISODES=100
SCALED_HEIGHT=47
SCALED_WIDTH=47
ENTROPY_REGULARIZATION=0.01
MAX_GRADIENT_NORM=40.0
GAMMA = .99 # discount rate 
MEMORY_BUFFER=30
LEARNING_RATE = 1e-4
stateSize = 2209 # size of input frames after preprocessing (47*47)
actionSize = 2 
MODEL_PATH = './a3cmodels'
ENV_NAME = 'FlappyBird-v0'


###HELPER METHODS####

#Pre-process input frame
def preProcess(fgbg,frame):
    frame=fgbg.apply(frame)
    frame=cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    frame=cv2.resize(frame, (SCALED_HEIGHT, SCALED_WIDTH))
    imgplot = plt.imshow(frame)
    plt.show()
    frame = np.reshape(frame,[np.prod(frame.shape)]) / 255.0
    return frame

# Used to set local agent network parameters to those of global network.
def update_target_graph(source,target):
    source_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, source)
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, target)
    params = []
    for source_var,target_var in zip(source_vars,target_vars):
        params.append(target_var.assign(source_var))
    return params

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
