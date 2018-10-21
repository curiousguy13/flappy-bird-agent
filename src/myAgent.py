# Not The world's simplest agent!
class MyAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
    def actTest(self, observation, reward, done):
        return self.action_space.sample()
