from gymEnvironment import GymEnvironment
from randomAgent import RandomAgent
from tensorforce.agents import PPOAgent


if __name__ == '__main__':
    flappyEnv=GymEnvironment('FlappyBird-v0')
    agent=RandomAgent(flappyEnv.getEnv().action_space)
    
    flappyEnv.setMonitor('/tmp/random-agent-results')
    flappyEnv.setSeed(0)
    flappyEnv.train(agent, episodeCount=5, quiet=True)
    flappyEnv.test(agent, episodeCount=5, quiet=False)