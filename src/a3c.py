import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.signal
import os
from random import choice
from time import sleep,time

from a3cAgent import Agent
from a3cNetwork import Network
from helper import *

def main():
    tf.reset_default_graph()

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    with tf.device("/cpu:0"):
        #to track global episodes
        global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)

        #try different optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        #Global Network
        master_network = Network(stateSize, actionSize, 'global', None)

        #if training, use all CPU threads. if testing use only 1 thread
        if(TRAINING==True and TESTING==False):
            numAgents = multiprocessing.cpu_count()
        elif(TRAINING==False and TESTING==True):
            numAgents=1
        else:
            print("Invalid values of TRAINING and TESTING")
            print("Exiting!!")
            exit()
        
        agents = []
        # Create agents
        for i in range(numAgents):
            agents.append(Agent(i,stateSize,actionSize,optimizer,MODEL_PATH,global_episodes))
        saver = tf.train.Saver(max_to_keep=2)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()

        #load existing model if LOAD_MODEL or TESTING are set to true
        if LOAD_MODEL == True or TESTING==True:
            print ('Loading Model...')
            checkpoint_state = tf.train.get_checkpoint_state(MODEL_PATH)
            saver.restore(sess,checkpoint_state.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            
        #start agent threads
        agent_threads = []
        for agent in agents:
            agent_work = lambda: agent.work(GAMMA,sess,coord,saver)
            agent_thread = threading.Thread(target=(agent_work))
            agent_thread.start()
            sleep(0.5)
            agent_threads.append(agent_thread)
        coord.join(agent_threads)


if __name__ == '__main__':
    main()
