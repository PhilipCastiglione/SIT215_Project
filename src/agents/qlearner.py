import random
import numpy as np

CARTPOLE_THETA_BUCKETS = 6
CARTPOLE_THETA_VELOCITY_BUCKETS = 4

class Qlearner():
    def __init__(self, parameters):
        self.alpha = parameters['alpha']
        self.gamma = parameters['gamma']
        self.epsilon = parameters['epsilon']
        super().__init__()

    def initialize_taxi_q_table(self, env):
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

    def taxi_training_action(self, env, observation):
        self.previous_observation = observation
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.q_table[observation])

    def taxi_evaluation_action(self, observation):
        return np.argmax(self.q_table[observation])

    def taxi_update(self, observation, action, reward):
        old_value = self.q_table[self.previous_observation, action]
        next_max = np.max(self.q_table[observation])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[self.previous_observation, action] = new_value

    def initialize_cartpole_q_table(self, env):
        self.q_table = np.zeros([CARTPOLE_THETA_BUCKETS * CARTPOLE_THETA_VELOCITY_BUCKETS, env.action_space.n])

    def cartpole_training_action(self, env, observation):
        self.previous_observation = observation
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.q_table[self._cartpole_observation(observation)])

    def cartpole_evaluation_action(self, observation):
        return np.argmax(self.q_table[self._cartpole_observation(observation)])

    def cartpole_update(self, observation, action, reward):
        old_value = self.q_table[self._cartpole_observation(self.previous_observation), action]
        next_max = np.max(self.q_table[self._cartpole_observation(observation)])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[self._cartpole_observation(self.previous_observation), action] = new_value

    def _cartpole_observation(self, observation):
        _, _, theta, theta_velocity = observation
        bucketed_theta = self._bucket(theta, CARTPOLE_THETA_BUCKETS, (-0.18, 0.18))
        bucketed_theta_velocity = self._bucket(theta_velocity, CARTPOLE_THETA_VELOCITY_BUCKETS, (-2.0, 2.0))
        index = (bucketed_theta - 1) * CARTPOLE_THETA_VELOCITY_BUCKETS + (bucketed_theta_velocity - 1)
        return index

    def _bucket(self, observation, num_buckets, obs_range):
        # calculate bucket number
        r_min = obs_range[0]
        r_max = obs_range[1]
        r_range = r_max - r_min
        bucket_size = r_range / num_buckets
        bucket = int(observation / bucket_size + num_buckets / 2)

        # bound
        bucket = min(bucket, num_buckets)
        bucket = max(bucket, 1)
        return bucket


"""
    # the training action is to either explore (try new paths), or exploit (use our current knowledge)
    def training_action(self, env, observation):
        self.previous_observation = observation
        if random.uniform(0, 1) < self.epsilon:
            return env.action_space.sample()
        else:
            state = self.bucket(observation) if self.bucketized_states else observation
            return np.argmax(self.q_table[state])

    # the evaluation action is the best move we can make using our current knowledge
    def evaluation_action(self, env, observation):
        state = self.bucket(observation) if self.bucketized_states else observation
        return np.argmax(self.q_table[state])

    # update our q_table record for the previous state
    def update(self, observation, action, reward):
        previous_state = self.bucket(self.previous_observation) if self.bucketized_states else self.previous_observation
        state = self.bucket(observation) if self.bucketized_states else observation

        old_value = self.q_table[previous_state, action]
        next_max = np.max(self.q_table[state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[previous_state, action] = new_value

    # calculate a number of buckets for the number of actions, so we can convert an array
    # of values into an integer number of buckets (with an arbitrary degree of accuracy)
    def num_buckets(self):
        return self.buckets_per_state ** len(self.states_range[0])

    # determine the bucket of an observation
    def bucket(self, observation):
        index = 0
        for i, o in enumerate(observation):
            # calculate bucket number
            r_min = self.states_range[0][i]
            r_max = self.states_range[1][i]
            r_range = r_max - r_min
            bucket_size = r_range / self.buckets_per_state
            bucket = int(o / bucket_size + self.buckets_per_state / 2)

            # bound
            bucket = min(bucket, self.buckets_per_state)
            bucket = max(bucket, 1)

            index += (bucket - 1) * (self.buckets_per_state ** i)
        return index
        """
