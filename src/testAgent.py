from gymEnvironment import GymEnvironment
from randomAgent import RandomAgent
from myAgent import MyAgent


if __name__ == '__main__':
    flappyEnv=GymEnvironment('FlappyBird-v0')
    agent=MyAgent(flappyEnv.getEnv().action_space, flappyEnv.getEnv().observation_space)
    
    flappyEnv.setMonitor('/tmp/my-agent-results')
    flappyEnv.setSeed(0)
    flappyEnv.train(agent, episodeCount=5, quiet=True)
    #flappyEnv.test(agent, episodeCount=5, quiet=False)