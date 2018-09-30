import random
import math
import numpy as np

# cartpole bucket hyperparameters
CARTPOLE_POSITION_BUCKETS = 2
CARTPOLE_POSITION_RANGE = (-2.0, 2.0)
CARTPOLE_VELOCITY_BUCKETS = 8
CARTPOLE_VELOCITY_RANGE = (-1.2, 1.2)
CARTPOLE_THETA_BUCKETS = 16
CARTPOLE_THETA_RANGE = (-0.08, 0.08)
CARTPOLE_THETA_VELOCITY_BUCKETS = 6
CARTPOLE_THETA_VELOCITY_RANGE = (-1.2, 1.2)

# implementation of SARSA
class TDlearner():
    def __init__(self, parameters):
        self.alpha = parameters['alpha']
        self.gamma = parameters['gamma']
        self.epsilon = parameters['epsilon']
        super().__init__()

    # TAXI

    def initialize_taxi_q_policy(self, env):
        self.q_policy = np.ones([env.observation_space.n, env.action_space.n])
        self.obs_range = env.action_space.n

    def taxi_training_action(self, env, observation):
        self.previous_observation = observation
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample() # explore
        else:
            # exploit using a probability weighted selection fom future states
            # with the existing policy
            next_actions = self.q_policy[observation]
            next_actions_sum = sum(next_actions)
            weighted_actions = [action / next_actions_sum for action in next_actions]
            return np.random.choice(np.arange(self.obs_range), p=weighted_actions)

    def taxi_evaluation_action(self, observation):
        next_actions = self.q_policy[observation]
        next_actions_sum = sum(next_actions)
        weighted_actions = [action / next_actions_sum for action in next_actions]
        return np.random.choice(np.arange(self.obs_range), p=weighted_actions)

    def taxi_update(self, observation, action, reward):
        # updates the policy with the reward gained, using a probability weighted
        # selection fom future states with the existing policy
        old_value = self.q_policy[self.previous_observation, action]
        next_actions = self.q_policy[observation]
        next_actions_sum = sum(next_actions)
        weighted_actions = [action / next_actions_sum for action in next_actions]
        next_action_score = np.random.choice(next_actions, p=weighted_actions)
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_action_score)
        self.q_policy[self.previous_observation, action] = new_value

    # CARTPOLE

    def initialize_cartpole_q_policy(self, env):
        obs_space = CARTPOLE_POSITION_BUCKETS * CARTPOLE_VELOCITY_BUCKETS * CARTPOLE_THETA_BUCKETS * CARTPOLE_THETA_VELOCITY_BUCKETS
        self.q_policy = np.ones([obs_space, env.action_space.n])
        self.obs_range = env.action_space.n

        # establish weak priors to optimise training - if theta < 0, move left, if theta > 0 move right
        for i in range(obs_space):
            if (i % (CARTPOLE_THETA_BUCKETS * CARTPOLE_THETA_VELOCITY_BUCKETS) < (CARTPOLE_THETA_BUCKETS / 2)):
                self.q_policy[i][0] = 0.1
            elif (i % (CARTPOLE_THETA_BUCKETS * CARTPOLE_THETA_VELOCITY_BUCKETS) >= (CARTPOLE_THETA_BUCKETS / 2)):
                self.q_policy[i][1] = 0.1

    def cartpole_training_action(self, env, observation):
        self.previous_observation = observation
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample() # explore
        else:
            # exploit using a probability weighted selection fom future states
            # with the existing policy
            next_actions = self.q_policy[self._cartpole_obs_index(observation)]
            next_actions_sum = sum(next_actions)
            weighted_actions = [action / next_actions_sum for action in next_actions]
            return np.random.choice(np.arange(self.obs_range), p=weighted_actions)

    def cartpole_evaluation_action(self, observation):
        next_actions = self.q_policy[self._cartpole_obs_index(observation)]
        next_actions_sum = sum(next_actions)
        weighted_actions = [action / next_actions_sum for action in next_actions]
        return np.random.choice(np.arange(self.obs_range), p=weighted_actions)

    def cartpole_update(self, observation, action, reward):
        # updates the policy with the reward gained, using a probability weighted
        # selection fom future states with the existing policy
        old_value = self.q_policy[self._cartpole_obs_index(self.previous_observation), action]
        next_actions = self.q_policy[self._cartpole_obs_index(observation)]
        next_actions_sum = sum(next_actions)
        weighted_actions = [action / next_actions_sum for action in next_actions]
        next_action_score = np.random.choice(next_actions, p=weighted_actions)
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_action_score)
        self.q_policy[self._cartpole_obs_index(self.previous_observation), action] = new_value

    def _cartpole_obs_index(self, observation):
        # because cartpole observations are continuous, we have to bucket them and
        # calculate an index for the qtable
        position, velocity, theta, theta_velocity = observation

        bucketed_position = self._bucket(position, CARTPOLE_POSITION_BUCKETS, CARTPOLE_POSITION_RANGE)
        bucketed_velocity = self._bucket(velocity, CARTPOLE_VELOCITY_BUCKETS, CARTPOLE_VELOCITY_RANGE)
        bucketed_theta = self._bucket(theta, CARTPOLE_THETA_BUCKETS, CARTPOLE_THETA_RANGE)
        bucketed_theta_velocity = self._bucket(theta_velocity, CARTPOLE_THETA_VELOCITY_BUCKETS, CARTPOLE_THETA_VELOCITY_RANGE)

        position_index = (bucketed_position - 1) * CARTPOLE_VELOCITY_BUCKETS * CARTPOLE_THETA_BUCKETS * CARTPOLE_THETA_VELOCITY_BUCKETS
        velocity_index = (bucketed_velocity - 1) * CARTPOLE_THETA_BUCKETS * CARTPOLE_THETA_VELOCITY_BUCKETS
        theta_index = (bucketed_theta - 1) * CARTPOLE_THETA_VELOCITY_BUCKETS
        theta_velocity_index = (bucketed_theta_velocity - 1)

        index = position_index + velocity_index + theta_index + theta_velocity_index
        return index

    def _bucket(self, observation, num_buckets, obs_range):
        # calculate bucket number
        r_min = obs_range[0]
        r_max = obs_range[1]
        r_range = r_max - r_min
        bucket_size = r_range / num_buckets
        bucket = math.ceil((observation + r_range / 2) / bucket_size)

        # bound
        bucket = min(bucket, num_buckets)
        bucket = max(bucket, 1)
        return bucket

    # FROZEN LAKE

    def initialize_frozen_lake_q_policy(self, env):
        self.q_policy = np.ones([env.observation_space.n, env.action_space.n])
        self.obs_range = env.action_space.n

    def frozen_lake_training_action(self, env, observation):
        self.previous_observation = observation
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample() # explore
        else:
            # exploit using a probability weighted selection fom future states
            # with the existing policy
            next_actions = self.q_policy[observation]
            next_actions_sum = sum(next_actions)
            weighted_actions = [action / next_actions_sum for action in next_actions]
            return np.random.choice(np.arange(self.obs_range), p=weighted_actions)

    def frozen_lake_evaluation_action(self, observation):
        next_actions = self.q_policy[observation]
        next_actions_sum = sum(next_actions)
        weighted_actions = [action / next_actions_sum for action in next_actions]
        return np.random.choice(np.arange(self.obs_range), p=weighted_actions)

    def frozen_lake_update(self, observation, action, reward):
        # updates the policy with the reward gained, using a probability weighted
        # selection fom future states with the existing policy
        old_value = self.q_policy[self.previous_observation, action]
        next_actions = self.q_policy[observation]
        next_actions_sum = sum(next_actions)
        weighted_actions = [action / next_actions_sum for action in next_actions]
        next_action_score = np.random.choice(next_actions, p=weighted_actions)
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_action_score)
        self.q_policy[self.previous_observation, action] = new_value

