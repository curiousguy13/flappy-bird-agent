import tensorflow as tf
from a3cNetwork import Network
import numpy as np
from helper import *
import gym
import gym_ple
import cv2      


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
        self.local_network = Network(stateSize,actionSize,self.name,optimizer)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        self.env = gym.make(ENV_NAME)
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
        feed_dict = {self.local_network.target_value:discounted_rewards,
            self.local_network.inputs:np.vstack(observations),
            self.local_network.actions:actions,
            self.local_network.advantages:advantages,
            self.local_network.state_input[0]:self.batch_rnn_state[0],
            self.local_network.state_input[1]:self.batch_rnn_state[1]}
        value_loss,policy_loss,entropy,grad_norm,var_norm, self.batch_rnn_state,_ = sess.run([self.local_network.value_loss,
            self.local_network.policy_loss,
            self.local_network.entropy,
            self.local_network.grad_norms,
            self.local_network.var_norms,
            self.local_network.state_out,
            self.local_network.apply_grads],
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
               
                #initial state
                state0 = self.env.reset()
                
                #Flag to check if episode is done or not
                done=0
                state0 = preProcess(self.fgbg, state0)

                #initialize rnn state
                rnn_state = self.local_network.state_init
                self.batch_rnn_state = rnn_state
                while (not done):
                    #Take an action using probabilities from policy network output.
                    a_dist,value,rnn_state = sess.run([self.local_network.policy,self.local_network.value,self.local_network.state_out], 
                        feed_dict={self.local_network.inputs:[state0],
                        self.local_network.state_input[0]:rnn_state[0],
                        self.local_network.state_input[1]:rnn_state[1]})
                    action = np.random.choice(a_dist[0],p=a_dist[0])
                    action = np.argmax(a_dist == action)
                    state1, reward, done, _ = self.env.step(action)
                    
                    #clip reward for flappy bird
                    if(reward==-5):
                        reward=-1

                    if done:
                        state1 = state0
                    else:
                        state1 = preProcess(self.fgbg, state1)
                        
                    experience_buffer.append([state0,action,reward,state1,done,value[0,0]])
                    episode_values.append(value[0,0])

                    episode_reward += reward
                    state0 = state1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    if(TESTING and episode_count%5==0 and not QUIET):
                        self.env.render()
                    #Train using eperience buffer if the episode has not ended yet
                    if TRAINING and len(experience_buffer) == MEMORY_BUFFER and not done :
                        #Use current value estimation for training
                        value1 = sess.run(self.local_network.value, 
                            feed_dict={self.local_network.inputs:[state0],
                            self.local_network.state_input[0]:rnn_state[0],
                            self.local_network.state_input[1]:rnn_state[1]})[0,0]
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
                    
                    if episode_count % 50 == 0 and self.name == 'agent_0':
                        saver.save(sess,MODEL_PATH+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                   
                    # Write summary values to tensorboard
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
                if(episode_count % 100==0):
                    print(sess.run(self.global_episodes))
