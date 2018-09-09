import gym
import numpy
import scipy
import IPython

"""
This file will test that your environment is configured correctly to use the tools
in this repository. If successful, you will see a window with a random agent
operating the cartpole environment for a short while.

Refer to README.md for instructions.
"""
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

