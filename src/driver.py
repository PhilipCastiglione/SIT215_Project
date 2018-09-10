import random
import sys
from gym.spaces import prng

# TODO: think about display, reporting etc
# think about when to stop convergence, make the param just a max
class Driver:
    def __init__(self, params):
        self.debug = params['debug']
        self.training_episodes = params['training_episodes']
        self.evaluation_episodes = params['evaluation_episodes']
        self.env = params['env']
        self.agent = params['agent']
        self.rewards = []

    def run_taxi_random(self):
        training_action = lambda _observation: self.agent.action(self.env)
        update = lambda _observation, _action, _reward: None
        evaluation_action = training_action

        self.run(training_action, update, evaluation_action)

    def run_taxi_qlearner(self):
        self.agent.initialize_taxi_q_table(self.env)

        training_action = lambda observation: self.agent.taxi_training_action(self.env, observation)
        update = lambda observation, action, reward: self.agent.taxi_update(observation, action, reward)
        evaluation_action = lambda observation: self.agent.taxi_evaluation_action(observation)

        self.run(training_action, update, evaluation_action)

    def run_cartpole_random(self):
        training_action = lambda _observation: self.agent.action(self.env)
        update = lambda _observation, _action, _reward: None
        evaluation_action = training_action

        self.run(training_action, update, evaluation_action)

    def run_cartpole_qlearner(self):
        self.agent.initialize_cartpole_q_table(self.env)

        training_action = lambda observation: self.agent.cartpole_training_action(self.env, observation)
        update = lambda observation, action, reward: self.agent.cartpole_update(observation, action, reward)
        evaluation_action = lambda observation: self.agent.cartpole_evaluation_action(observation)

        self.run(training_action, update, evaluation_action)

    def run(self, training_action, update, evaluation_action):
        self.train(training_action, update)
        #self.evaluate(evaluation_action)
        self.report()

    def train(self, training_action, update):
        for _ in range(self.training_episodes):
            observation = self.env.reset()
            done = False
            step = 0
            episode_reward = 0
            while not done:
                if (self.debug):
                    self.env.render()
                    print(f"observation: {observation}")

                action = training_action(observation)
                observation, reward, done, info = self.env.step(action)
                episode_reward += reward
                update(observation, action, reward)
                step += 1

                if done: # OpenAI gym enforces a maximum of 200 steps if not solved
                    if (self.debug):
                        print(f"Episode finished after {step} timesteps")
                    break
            self.rewards.append(episode_reward)

    def evaluate(self, evaluation_action):
        rewards = []
        for _ in range(self.evaluation_episodes):
            observation = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = evaluation_action(observation)
                observation, reward, done, info = self.env.step(action)
                episode_reward += reward

                if done: # OpenAI gym enforces a maximum of 200 steps if not solved
                    break
            rewards.append(episode_reward)
        average_reward = sum(rewards) / self.evaluation_episodes
        print(f"average reward level over {self.evaluation_episodes} episodes: {average_reward}")

    def report(self):
        print(self.rewards)

    """
    def demonstrate(self):
        for _ in range(10):
            observation = self.env.reset()
            done = False
            while not done:
                self.env.render()
                action = self.agent.evaluation_action(self.env, observation)
                observation, reward, done, info = self.env.step(action)

                if done: # OpenAI gym enforces a maximum of 200 steps if not solved
                    break
    """

