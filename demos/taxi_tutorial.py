import gym
from gym.spaces import prng
import numpy as np
from IPython.display import clear_output
from time import sleep
import random

"""
This file contains workings for the taxi problem tutorial at:

    https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

All credit to the authors.
"""


# the .env on the end removes the default 200 timestep limit - this goes against the philosophy though :/
env = gym.make("Taxi-v2").env

"""
env.render()

env.reset() # reset environment to a new, random state
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state
env.render()

"""

"""
# this is required to generate different starting positions
prng.seed(1337)

env.s = 328  # set environment to illustration's state

epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    if reward == -10:
        penalties += 1

    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(mode='ansi'),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1


print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

print_frames(frames)
"""

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# %%time # this is a magic ipython thing
"""Training the agent"""

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

all_rewards = []
for _ in range(episodes):
    state = env.reset()
    episode_reward = 0
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        episode_reward += reward

        if reward == -10:
            penalties += 1

        epochs += 1
    all_rewards.append(episode_reward)
    
    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

import matplotlib
# using an alternative backend to macos gui driver, because there is an issue with
# matplotlib, virtualenv and macos: https://github.com/pypa/virtualenv/issues/54
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.plot(all_rewards, linewidth=1)
plt.ylabel('reward')
plt.xlabel('iterations')
plt.show()
