import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from helper import *

#Network for A3C
class Network():
    def __init__(self, stateSize, actionSize, scope, optimizer):
        with tf.variable_scope(scope):
            print("Building model.....")

            #Input layer
            self.inputs = tf.placeholder(shape=[None, stateSize], dtype = tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, SCALED_HEIGHT, SCALED_WIDTH, 1])
            #print('stateSize=',stateSize)
            #print('inputs=',self.inputs)
            #Convolutional Layers
            self.c1=slim.conv2d(activation_fn=tf.nn.relu, inputs = self.imageIn, num_outputs=32,kernel_size=[3,3], stride=[2,2], padding='VALID')
            self.c2=slim.conv2d(activation_fn=tf.nn.relu, inputs = self.c1, num_outputs=32,kernel_size=[3,3], stride=[2,2], padding='VALID')
            self.c3=slim.conv2d(activation_fn=tf.nn.relu, inputs = self.c2, num_outputs=32,kernel_size=[3,3], stride=[2,2], padding='VALID')
            self.c4=slim.conv2d(activation_fn=tf.nn.relu, inputs = self.c3, num_outputs=32,kernel_size=[3,3], stride=[2,2], padding='VALID')

            fc0 = slim.fully_connected(slim.flatten(self.c3), 256, activation_fn=tf.nn.relu)

            #RNN Layers
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            #initialize cells and hidden units with zeros
            cell_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            hidden_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [cell_init, hidden_init]
            #placeholders for cell and hidden units
            cell_input = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            hidden_input = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_input = (cell_input, hidden_input)

            #input to RNN
            rnn_in = tf.expand_dims(fc0, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_input = tf.contrib.rnn.LSTMStateTuple(cell_input, hidden_input)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_input, sequence_length=step_size,
                time_major=False)

            #lstm state representation
            lstm_cell, lstm_hidden = lstm_state
            self.state_out = (lstm_cell[:1, :], lstm_hidden[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            #Output layer for policy estimations
            #Using Xavier initializations for fully connected layer by default
            self.policy = slim.fully_connected(rnn_out,actionSize,
                activation_fn=tf.nn.softmax)

            #Using Xavier initializations for fully connected layer by default
            #Output layer for value estimations
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None)
            
            #Only the agent network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,actionSize,dtype=tf.float32)
                self.target_value = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_value - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                
                #Use entropy only when Training and not during testing
                if(TRAINING):
                    self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * ENTROPY_REGULARIZATION
                else:
                    self.loss = 0.5 * self.value_loss + self.policy_loss
                #Get gradients from local network
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,MAX_GRADIENT_NORM)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                #print('gv=', global_vars)
                self.apply_grads = optimizer.apply_gradients(zip(grads,global_vars))
