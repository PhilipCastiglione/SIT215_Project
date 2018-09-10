import gym
from src.driver import Driver
from src.agents.random import Random
from src.agents.qlearner import Qlearner

def taxi_random():
    env = gym.make('Taxi-v2')
    agent = Random()
    driver = Driver({
        'debug': False,
        'training_episodes': 100000,
        'evaluation_episodes': 100,
        'env': env,
        'agent': agent,
    })
    driver.run_taxi_random()

def taxi_qlearner():
    env = gym.make('Taxi-v2')
    agent = Qlearner({
        'alpha': 0.1,
        'gamma': 0.6,
        'epsilon': 0.1,
    })
    driver = Driver({
        'debug': False,
        'training_episodes': 100000,
        'evaluation_episodes': 100,
        'env': env,
        'agent': agent,
    })
    driver.run_taxi_qlearner()

def cartpole_random():
    env = gym.make('CartPole-v1')
    agent = Random()
    driver = Driver({
        'debug': False,
        'training_episodes': 100000,
        'evaluation_episodes': 100,
        'env': env,
        'agent': agent,
    })
    driver.run_cartpole_random()

def cartpole_qlearner():
    env = gym.make('CartPole-v1')
    agent = Qlearner({
        'alpha': 0.1,
        'gamma': 0.6,
        'epsilon': 0.8,
    })
    driver = Driver({
        'debug': False,
        'training_episodes': 100000,
        'evaluation_episodes': 100,
        'env': env,
        'agent': agent,
    })
    driver.run_cartpole_qlearner()

if __name__ == '__main__':
    #taxi_random()
    #taxi_qlearner()

    #cartpole_random()
    cartpole_qlearner()

