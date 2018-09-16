import gym
from src.driver import Driver
from src.agents.random import Random
from src.agents.qlearner import Qlearner

def taxi_random():
    agent = Random()
    driver = Driver({
        'training_episodes': 1000,
        'env': gym.make('Taxi-v2'),
        'agent': agent,
    })
    driver.run_taxi_random()

def taxi_qlearner():
    agent = Qlearner({
        'alpha': 0.1,
        'gamma': 0.6,
        'epsilon': 0.1,
    })
    driver = Driver({
        'training_episodes': 100000,
        'env': gym.make('Taxi-v2'),
        'agent': agent,
    })
    driver.run_taxi_qlearner()

def cartpole_random():
    agent = Random()
    driver = Driver({
        'training_episodes': 1000,
        'env': gym.make('CartPole-v1'),
        'agent': agent,
    })
    driver.run_cartpole_random()

def cartpole_qlearner():
    agent = Qlearner({
        'alpha': 0.1,
        'gamma': 0.1,
        'epsilon': 0.3,
    })
    driver = Driver({
        'training_episodes': 1000000,
        'env': gym.make('CartPole-v1'),
        'agent': agent,
    })
    driver.run_cartpole_qlearner()

if __name__ == '__main__':
    #taxi_random()
    #taxi_qlearner()
    #cartpole_random()
    cartpole_qlearner()

