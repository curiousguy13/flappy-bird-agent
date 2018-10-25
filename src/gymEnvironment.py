import logging
import os, sys

import gym
from gym.wrappers import Monitor
import gym_ple
import matplotlib.pyplot as plt

#TODO: Add logging
class GymEnvironment(object):
    def __init__(self, name):
        self.name=name
        self.env=gym.make(name)
    
    def getEnv(self):
        return self.env

    def setMonitor(self, outDir):
        self.env=Monitor(self.env, outDir, force=True)
    
    def setSeed(self, seed):
        self.env.seed(seed)
    
    def train(self, agent, episodeCount, quiet):
        
        done=False
        rewards=list()
        for episode in range(episodeCount):
            reward=0
            timestep=0
            ob=self.env.reset()
            print(ob.shape)
            action = agent.act(ob, reward, done)
            timestep+=1
            ob, reward, done, _ = self.env.step(action)
            #imgplot = plt.imshow(ob)
            #plt.show()
            while True:
                action = agent.act(ob, reward, done)
                timestep+=1
                ob, reward, done, _ = self.env.step(action)
                
                #flappy specific reward function
                if(reward!=-5):
                    reward+=1
                    print(reward)
                if done:
                    break

                if(not quiet):
                    self.env.render()
                
            rewards.append(reward)
            print(rewards)
            
        self.env.close()   
    def test(self, agent, episodeCount, quiet):
        reward=0
        done=False
        rewards=list()
        for episode in range(episodeCount):
            ob=self.env.reset()
            print(ob.shape)
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = self.env.step(action)
            #imgplot = plt.imshow(ob)
            #plt.show()
            while True:
                action = agent.actTest(ob, reward, done)
                ob, reward, done, _ = self.env.step(action)
                if done:
                    break
                if(not quiet):
                    self.env.render()
            rewards.append(reward)
        self.env.close()          

