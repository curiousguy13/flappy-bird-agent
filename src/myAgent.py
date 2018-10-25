# Not The world's simplest agent!
# Todo:Save Model
# Todo:Add Logging

import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

fgbg=cv2.bgsegm.createBackgroundSubtractorMOG()
def preProcessInput(observation):
    #print(observation.shape)
   
    #imgplot = plt.imshow(observation)
    #plt.show()
    #cv2.imshow('ob',observation)
    observation=fgbg.apply(observation)
    observation=observation[:,:]
    #imgplot = plt.imshow(observation)
    #plt.show()
    #print(observation.shape)
    #cv2.imshow('bgremoved',fgmask)
class MyAgent(object):
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space= observation_space

    def act(self, observation, reward, done):
        observation=preProcessInput(observation)
        print(self.action_space.n)
        # Next, we build a very simple model.
        
        model = Sequential()
        model.add(Conv2D(16,(3,3),padding='same',input_shape=self.observation_space.shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(32,(5,5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(self.action_space.n))
        model.add(Activation('linear'))
        print(model.summary())
        


        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                    target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        
        return self.action_space.sample()
    def actTest(self, observation, reward, done):
        return self.action_space.sample()
