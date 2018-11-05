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
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from gymEnvironment import GymEnvironment
import cv2      
from PIL import Image       

SCALED_IMAGE_HEIGHT=80  
SCALED_IMAGE_WIDTH=80
WINDOW_LENGTH=4
TRAINING_STEPS=200
class GameProcessor(Processor):
    #tried all other background subtractors as well
    #tuned hyperparams
    fgbg=cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=5, backgroundRatio=0.0000007)
    pipes_crossed=0.0
    score=0.0
    end=0
    def process_observation(self, observation):
        #print(observation.shape)
        #observation=observation[:,:]
        #img=Image.fromarray(observation)
        #img=img.convert('L') #grayscale
        #processed_observation = np.array(img)
        
        #imgplot = plt.imshow(observation)
        #plt.show()
        #cv2.imshow('ob',observation)
        gray_image = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        processed_observation=cv2.resize(gray_image, (SCALED_IMAGE_HEIGHT, SCALED_IMAGE_WIDTH))
        processed_observation=self.fgbg.apply(processed_observation)
        
        imgplot = plt.imshow(processed_observation)
        plt.show()
        
        #print("observation shape=", processed_observation.shape)
        #cv2.imshow('bgremoved',fgmask)
        #return observation
        return np.divide(processed_observation, 255.0)
    def process_reward(self, reward):
        #print(reward)
        if(reward==0):
            reward = 1.0
        elif(reward==1):
            print('pipe_crossed')
            reward = 25
            self.pipes_crossed+=1
        elif(reward==-1):
            reward = -5
        elif(reward==-5):
            reward = -10
            self.end+=1
            self.score=(self.pipes_crossed-self.end)/self.end
            print('pipes crossed=',self.pipes_crossed)
            print('score=', self.score)
        #print(reward/25)
        return reward/25


ENV_NAME = 'FlappyBird-v0'
#flappyEnv=GymEnvironment('FlappyBird-v0')
#flappyEnv.setSeed(0)
#nb_actions=flappyEnv.getEnv().action_space
#nb_observation_space=flappyEnv.getEnv().observation_space
# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
#input_shape=(1,)+env.observation_space.shape[:2]
input_shape= (WINDOW_LENGTH,)+(SCALED_IMAGE_HEIGHT, SCALED_IMAGE_WIDTH)
print(input_shape)
# Next, we build a very simple model.
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
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Dense(16))
#model.add(Activation('relu'))
model.add(Dense(env.action_space.n))
model.add(Activation('linear'))

print(model.summary())
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', 
                              value_max=1., value_min=.1, value_test=0.0, nb_steps=TRAINING_STEPS)
processor=GameProcessor()
memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)
#policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10000,
               target_model_update=1e-2, policy=policy, processor=processor)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
#dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
dqn.fit(env, nb_steps=TRAINING_STEPS, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights_ep_{}.h5f'.format(ENV_NAME, TRAINING_STEPS), overwrite=False)
#dqn.load_weights('dqn_FlappyBird-v0_weights_ep_1000000.h5f')
# Finally, evaluate our algorithm for 5 episodes.
processor.score=0
processor.pipes_crossed=0
processor.end=0
dqn.test(env, nb_episodes=5, visualize=True)
print(processor.score)
