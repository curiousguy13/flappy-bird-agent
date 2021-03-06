import numpy as np
import gym
import matplotlib.pyplot as plt

from gym.wrappers import Monitor
import gym_ple
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.layers import Activation,Conv2D,MaxPooling2D,Flatten 
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy, Policy
from rl.memory import SequentialMemory
from gymEnvironment import GymEnvironment
import cv2      
from PIL import Image       

SCALED_IMAGE_HEIGHT=80  
SCALED_IMAGE_WIDTH=80
WINDOW_LENGTH=4
TRAINING_STEPS=2000000
TEST_EPISODES=5
LOAD=False
LOAD_FILE='dqn_FlappyBird-v0_weights_ep_500000.h5f'
ENV_NAME = 'FlappyBird-v0'
numberOfUniquePipesCrossedByAgent=0
currPipe=0

def main():


    env = gym.make(ENV_NAME)
    env.seed(123)
    nb_actions = env.action_space.n
    input_shape= (WINDOW_LENGTH,)+(SCALED_IMAGE_HEIGHT, SCALED_IMAGE_WIDTH)

    '''
    model = Sequential()
    model.add(Conv2D(16,(3,3),padding='same',input_shape=input_shape, data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(5,5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('relu'))
    #model.add(Dense(16))
    #model.add(Activation('relu'))
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))
    '''

    #network architecture
    model = Sequential()
    model.add(Conv2D(32,(8,8),strides=4, padding='same',input_shape=input_shape, data_format="channels_first"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(4,4), strides=2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(2,2), strides=1))
    model.add(MaxPooling2D(pool_size=(1,1)))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    #model.add(Dense(16))
    #model.add(Activation('relu'))
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))

    print(model.summary())
    
    #policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', 
    #                            value_max=.4, value_min=.2, value_test=0.0, nb_steps=TRAINING_STEPS/5)

    policy=MyPolicy()
    #load our processor
    processor=GameProcessor()

    #memory for experience replay
    memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)
    
    #initialize agent
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=TRAINING_STEPS/50,
                target_model_update=5e-2, policy=policy, processor=processor, gamma=0.95)
    dqn.compile(Adam(lr=1e-3), metrics=['mse'])

    if(LOAD):
        dqn.load_weights(LOAD_FILE)
    else:
        dqn.fit(env, nb_steps=TRAINING_STEPS, visualize=False, verbose=2)

        # After training is done, we save the final weights.
        dqn.save_weights('dqn_{}_weights_ep_{}.h5f'.format(ENV_NAME, TRAINING_STEPS), overwrite=False)
    

    #reset processor variables before testing
    processor.score=0
    processor.pipes_crossed=0
    processor.end=0

    #test our agent
    dqn.test(env, nb_episodes=TEST_EPISODES, visualize=True)
    print('Test Score over ' + str(TEST_EPISODES) + ' episodes is : ' + str(processor.score))

class MyPolicy(Policy):
    n=0
    def select_action(self, q_values):
        if(currPipe<numberOfUniquePipesCrossedByAgent):
            eps=0.15
        else:
            eps=0.8 - (0.75*n/10000)
            self.n+=1
            if(eps<0.2):
                n=0
        return EpsGreedyQPolicy(eps).select_action(q_values)
class GameProcessor(Processor):
    
    #background subtractor
    #tried all other background subtractors from opencv as well
    #manually tuned hyperparams
    fgbg=cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=5, backgroundRatio=0.0000007)
    
    #number of total pipes crossed over all episodes. used to calculate score
    pipes_crossed=0.0

    #how many times did the episode end. used for score calculation
    end=0

    score=0.0
    
    pipes_crossed_in_this_episode=0
    def process_observation(self, observation):
        
        #print('og')
        #imgplot = plt.imshow(observation)
        #plt.show()
        processed_observation=self.fgbg.apply(observation)
        #print('bgsub')
        #imgplot = plt.imshow(processed_observation)
        #plt.show()
        #convert to grayscale
        #processed_observation = cv2.cvtColor(processed_observation, cv2.COLOR_BGR2GRAY)
        
        #downscale image for more manageable training 
        
        #processed_observation=self.fgbg.apply(processed_observation)
        #print('gray')
        #imgplot = plt.imshow(processed_observation)
        #lt.show()
        #processed_observation=cv2.morphologyEx(processed_observation, cv2.MORPH_CLOSE, (5,5))
        processed_observation=cv2.adaptiveThreshold(processed_observation,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

        #print('thresholding')
        #imgplot = plt.imshow(processed_observation)
        #plt.show()
        
        #print('resize')
        processed_observation=cv2.resize(processed_observation, (SCALED_IMAGE_HEIGHT, SCALED_IMAGE_WIDTH))
        #kernel=(3,3)
        #processed_observation = cv2.morphologyEx(processed_observation, cv2.MORPH_OPEN, kernel)

        #uncomment to display each frame
        #imgplot = plt.imshow(processed_observation)
        #plt.show()
        
        #normalize pixel values [0,1]
        return np.divide(processed_observation, 255.0)
    
    #reward function
    def process_reward(self, reward):
        #print(reward)
        if(reward==0):
            #we want to reward the agent by a little bit for just flying
            reward = 1.0
        elif(reward==1):
            #reward by a lot if pipe crossed
            print('pipe_crossed')
            reward = 100
            self.pipes_crossed+=1
            self.pipes_crossed_in_this_episode+=1
            currPipe=self.pipes_crossed_in_this_episode
        elif(reward==-5):
            #negative reward if terminal state reached
            reward = -20
            self.end+=1

            #score calculation
            self.score=(self.pipes_crossed-self.end)/self.end
            print('pipes crossed=',self.pipes_crossed)
            print('score=', self.score)


            numberOfUniquePipesCrossedByAgent=self.pipes_crossed_in_this_episode
            self.pipes_crossed_in_this_episode=0
            currPipe=0
            #return normalized reward [-1, 1]
        return reward/100

if __name__ == "__main__":
    main()