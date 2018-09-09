import random
import numpy as np
from src.agents.agent import Agent

class Qlearner(Agent):
    def __init__(self, parameters):
        self.alpha = parameters['alpha']
        self.gamma = parameters['gamma']
        self.epsilon = parameters['epsilon']
        self.q_table = np.zeros([parameters['num_states'], parameters['num_actions']])
        super().__init__()

    # the training action is to either explore (try new paths), or exploit (use our current knowledge)
    def training_action(self, env, observation):
        self.previous_observation = observation
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.q_table[observation])

    # the evaluation action is the best move we can make using our current knowledge
    def evaluation_action(self, env, observation):
        return np.argmax(self.q_table[observation])

    # update our q_table record for the previous state
    def update(self, observation, action, reward):
        old_value = self.q_table[self.previous_observation, action]
        next_max = np.max(self.q_table[observation])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[self.previous_observation, action] = new_value

