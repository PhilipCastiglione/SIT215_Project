import gym
from src.driver import Driver
from src.agents.random import Random
from src.agents.qlearner import Qlearner
from src.agents.tdlearner import TDlearner

'''
USAGE INSTRUCTIONS

At the bottom of this file, these functions defining a combination of an agent
and an environment are invoked. All but one are commented out. Choose which
agent and environment you want to run, and uncomment that line. For example,
to see the Qlearner agent operating the Taxi environment, uncomment:

    #taxi_qlearner()

It is recommended you leave all other function invocations commented out when
you run this file, as it will be faster and you will only see the output you
are interested in.
'''

def taxi_random():
    agent = Random()
    driver = Driver({
        'epochs': 1000,
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
        'epochs': 10000,
        'env': gym.make('Taxi-v2'),
        'agent': agent,
    })
    driver.run_taxi_qlearner()

def cartpole_random():
    agent = Random()
    driver = Driver({
        'epochs': 1000,
        'env': gym.make('CartPole-v1'),
        'agent': agent,
    })
    driver.run_cartpole_random()

def cartpole_qlearner():
    agent = Qlearner({
        'alpha': 0.2,
        'gamma': 0.5,
        'epsilon': 0.1,
    })
    driver = Driver({
        'epochs': 50000,
        'env': gym.make('CartPole-v1'),
        'agent': agent,
    })
    driver.run_cartpole_qlearner()

def cartpole_tdlearner():
    agent = TDlearner({
        'alpha': 0.2,
        'gamma': 0.5,
        'epsilon': 0.1,
    })
    driver = Driver({
        'epochs': 50000,
        'env': gym.make('CartPole-v1'),
        'agent': agent,
    })
    driver.run_cartpole_tdlearner()

def frozen_lake_random():
    agent = Random()
    driver = Driver({
        'epochs': 1000,
        'env': gym.make('FrozenLake-v0'),
        'agent': agent,
    })
    driver.run_frozen_lake_random()

def frozen_lake_qlearner():
    agent = Qlearner({
        'alpha': 0.1,
        'gamma': 0.6,
        'epsilon': 0.3,
    })
    driver = Driver({
        'epochs': 10000,
        'env': gym.make('FrozenLake-v0'),
        'agent': agent,
    })
    driver.run_frozen_lake_qlearner()

def frozen_lake_tdlearner():
    agent = TDlearner({
        'alpha': 0.1,
        'gamma': 0.6,
        'epsilon': 0.3,
    })
    driver = Driver({
        'epochs': 10000,
        'env': gym.make('FrozenLake-v0'),
        'agent': agent,
    })
    driver.run_frozen_lake_tdlearner()

if __name__ == '__main__':
    taxi_random()
    #taxi_qlearner()
    #cartpole_random()
    #cartpole_qlearner()
    #cartpole_tdlearner()
    #frozen_lake_random()
    #frozen_lake_qlearner()
    #frozen_lake_tdlearner()

