from src.agents.agent import Agent

class Random(Agent):
    def __init__(self):
        super().__init__()

    # the training action is any random action from within the environment action space
    def training_action(self, env, _observation):
        return env.action_space.sample()

    # the evaluation action is... also any random action from within the environment action space
    def evaluation_action(self, env, _observation):
        return env.action_space.sample()

    # the random agent never learns 🤦
    def update(self, _observation, _action, _reward):
        pass
