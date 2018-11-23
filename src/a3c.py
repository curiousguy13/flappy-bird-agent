import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import os
from random import choice
from time import sleep,time
import gym
import gym_ple
import cv2      


#change TRAINING to False and TESTING to True for testing
TRAINING=True
#Will Load model if TESTING = TRUE 
TESTING=False
QUIET=True
LOAD_MODEL = False

SCALED_HEIGHT=47
SCALED_WIDTH=47
TRAINING_EPISODES=20000
TESTING_EPISODES=100
GAMMA = .99 # discount rate 
LEARNING_RATE = 1e-4
MAX_GRADIENT_NORM=40.0
ENTROPY_REGULARIZATION=0.01
stateSize = 2209 # size of input frames after preprocessing (47*47)
actionSize = 2 
MODEL_PATH = './a3cmodels'
ENvar_normAME = 'FlappyBird-v0'

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

        #if training, use all CPU threads. if testing use only 1
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
            
        agent_threads = []
        for agent in agents:
            agent_work = lambda: agent.work(GAMMA,sess,coord,saver)
            agent_thread = threading.Thread(target=(agent_work))
            agent_thread.start()
            sleep(0.5)
            agent_threads.append(agent_thread)
        coord.join(agent_threads)

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

            fc0 = slim.fully_connected(slim.flatten(self.c4), 256, activation_fn=tf.nn.relu)

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

            rnn_in = tf.expand_dims(fc0, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_input = tf.contrib.rnn.LSTMStateTuple(cell_input, hidden_input)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_input, sequence_length=step_size,
                time_major=False)
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
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * ENTROPY_REGULARIZATION

                #Get gradients from local network
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,MAX_GRADIENT_NORM)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                #print('gv=', global_vars)
                self.apply_grads = optimizer.apply_gradients(zip(grads,global_vars))

class Agent():
    def __init__(self,name,stateSize,actionSize,optimizer,MODEL_PATH,global_episodes):
        self.name = "agent_" + str(name)
        self.number = name        
        self.MODEL_PATH = MODEL_PATH
        self.optimizer = optimizer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter(MODEL_PATH+"/"+"train_"+str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = Network(stateSize,actionSize,self.name,optimizer)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        self.env = gym.make(ENvar_normAME)
        self.env.seed(123)
        self.actions = np.identity(self.env.action_space.n,dtype=bool).tolist()
        self.fgbg=cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=5, backgroundRatio=0.0000007)

        
    def train(self,experience_buffer,sess,GAMMA,bootstrap_value):
        experience_buffer = np.array(experience_buffer)

        #extract values from experience buffer
        observations = experience_buffer[:,0]
        actions = experience_buffer[:,1]
        rewards = experience_buffer[:,2]
        next_observations = experience_buffer[:,3]
        values = experience_buffer[:,5]
        
        # Here we take the rewards and values from the experience_buffer, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,GAMMA)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + GAMMA * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,GAMMA)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.state_input[0]:self.batch_rnn_state[0],
            self.local_AC.state_input[1]:self.batch_rnn_state[1]}
        value_loss,policy_loss,entropy,grad_norm,var_norm, self.batch_rnn_state,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_out,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return value_loss / len(experience_buffer),policy_loss / len(experience_buffer),entropy / len(experience_buffer), grad_norm,var_norm
        
    def work(self,GAMMA,sess,coord,saver):
        total_steps = 0
        episodes_to_run=0
        if(TRAINING):
            episodes_to_run=TRAINING_EPISODES
            episode_count = sess.run(self.global_episodes)
        else:
            episodes_to_run=TESTING_EPISODES
            episode_count=0
        print ("Starting agent " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop() and episode_count < episodes_to_run:
                sess.run(self.update_local_ops)
                experience_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
               
                state0 = self.env.reset()
                done=0
                state0 = preProcess(self.fgbg, state0)
                rnn_state = self.local_AC.state_init
                
                self.batch_rnn_state = rnn_state
                while (not done):
                    #Take an action using probabilities from policy network output.
                    a_dist,value,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                        feed_dict={self.local_AC.inputs:[state0],
                        self.local_AC.state_input[0]:rnn_state[0],
                        self.local_AC.state_input[1]:rnn_state[1]})
                    action = np.random.choice(a_dist[0],p=a_dist[0])
                    action = np.argmax(a_dist == action)
                    state1, reward, done, _ = self.env.step(action)
                    #clip reward for flappy bird
                    if(reward==-5):
                        reward=-1


                    if not done:
                        state1 = preProcess(self.fgbg, state1)
                        
                    else:
                        state1 = state0
                        
                    experience_buffer.append([state0,action,reward,state1,done,value[0,0]])
                    episode_values.append(value[0,0])

                    episode_reward += reward
                    state0 = state1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    if(TESTING and episode_count%5==0 and not QUIET):
                        self.env.render()
                    #Train using eperience buffer if the episode has not ended yet
                    if TRAINING and len(experience_buffer) == 30 and not done :
                        #Use current value estimation for training
                        value1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[state0],
                            self.local_AC.state_input[0]:rnn_state[0],
                            self.local_AC.state_input[1]:rnn_state[1]})[0,0]
                        value_loss,policy_loss,entropy,grad_norms,var_norms = self.train(experience_buffer,sess,GAMMA,value1)
                        experience_buffer = []
                        sess.run(self.update_local_ops)
                    if done:
                        break
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                
                # Update the network using the episode buffer at the end of the episode.
                if len(experience_buffer) != 0:
                    value_loss,policy_loss,entropy,grad_norms,var_norms = self.train(experience_buffer,sess,GAMMA,0.0)
                                
                if(TESTING):
                    print('episode reward=',self.episode_rewards[-1:])
                    print('score=',np.mean(self.episode_rewards[:]))


                if TRAINING and episode_count % 5 == 0 and episode_count != 0:

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                   
                    summary.value.add(tag='Performance/Mean Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Performance/Reward', simple_value=float(self.episode_rewards[-1:][0]))
                    summary.value.add(tag='Performance/Mean Episode Length', simple_value=float(mean_length))
                    summary.value.add(tag='Performance/Episode Length', simple_value=float(self.episode_lengths[-1:][0]))
                    summary.value.add(tag='Performance/Value', simple_value=float(mean_value))
                    
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(value_loss))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(policy_loss))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(entropy))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(grad_norms))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
                
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
            
                if self.name == 'agent_0':
                    sess.run(self.increment)
                episode_count += 1

                #Print Episode count periodically
                if(episode_count % 200==0):
                    print(sess.run(self.global_episodes))

###HELPER METHODS####

def preProcess(fgbg,frame):
    #frame = scipy.misc.imresize(frame,[SCALED_HEIGHT,SCALED_WIDTH])
    #frame=frame[:,:,0]
    frame=fgbg.apply(frame)
    frame=cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    frame=cv2.resize(frame, (SCALED_HEIGHT, SCALED_WIDTH))
    frame = np.reshape(frame,[np.prod(frame.shape)]) / 255.0
    return frame

# Copies one set of variables to another.
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

if __name__ == '__main__':
    main()
