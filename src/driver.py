import matplotlib
# using an alternative backend to macos gui driver, because there is an issue with
# matplotlib, virtualenv and macos: https://github.com/pypa/virtualenv/issues/54
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"""
The Driver class marshals the behaviour of the provided gym environment and
agent to train, update and evaluate it's ability to perform.

Driver contains the main training loop and can plot charts to show training
performance, as well as use the environments render method to demonstrate
trained performance.

Driver contains specific methods to arrange running for combinations of
agent and environment.
"""
class Driver:
    def __init__(self, params):
        self.epochs = params['epochs']
        self.env = params['env']
        self.agent = params['agent']
        self.training_rewards = []
        self.evaluation_rewards = []

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

    def run_cartpole_tdlearner(self):
        self.agent.initialize_cartpole_q_policy(self.env)

        training_action = lambda observation: self.agent.cartpole_training_action(self.env, observation)
        update = lambda observation, action, reward: self.agent.cartpole_update(observation, action, reward)
        evaluation_action = lambda observation: self.agent.cartpole_evaluation_action(observation)

        self.run(training_action, update, evaluation_action)

    def run_frozen_lake_random(self):
        training_action = lambda _observation: self.agent.action(self.env)
        update = lambda _observation, _action, _reward: None
        evaluation_action = training_action

        self.run(training_action, update, evaluation_action)

    def run_frozen_lake_qlearner(self):
        self.agent.initialize_frozen_lake_q_table(self.env)

        training_action = lambda observation: self.agent.frozen_lake_training_action(self.env, observation)
        update = lambda observation, action, reward: self.agent.frozen_lake_update(observation, action, reward)
        evaluation_action = lambda observation: self.agent.frozen_lake_evaluation_action(observation)

        self.run(training_action, update, evaluation_action)

    def run_frozen_lake_tdlearner(self):
        self.agent.initialize_frozen_lake_q_policy(self.env)

        training_action = lambda observation: self.agent.frozen_lake_training_action(self.env, observation)
        update = lambda observation, action, reward: self.agent.frozen_lake_update(observation, action, reward)
        evaluation_action = lambda observation: self.agent.frozen_lake_evaluation_action(observation)

        self.run(training_action, update, evaluation_action)

    # main engine: training and evaluation loop, plot then demonstrate
    def run(self, training_action, update, evaluation_action):
        for i in range(self.epochs):
            if ((i + 1) % 1000 == 0):
                print("progress: {}%".format(100 * (i + 1) // self.epochs))
            self.train_once(training_action, update)
            self.evaluate_once(evaluation_action)

        self.plot()
        
        try:
            self.demonstrate(evaluation_action)
        except NotImplementedError:
            print("Cannot demonstrate: render method on env not implemented.")

    # a single instance of training of the agent in the environment
    def train_once(self, training_action, update):
        observation = self.env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = training_action(observation)
            observation, reward, done, info = self.env.step(action)
            episode_reward += reward
            update(observation, action, reward)
        self.training_rewards.append(episode_reward)

    # a single instance of evaluation of the agent at it's current level of training
    def evaluate_once(self, evaluation_action):
        observation = self.env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = evaluation_action(observation)
            observation, reward, done, info = self.env.step(action)
            episode_reward += reward
        self.evaluation_rewards.append(episode_reward)

    # plot training and evaluation reward levels at each epoch
    def plot(self):
        plt.subplot('211')
        plt.plot(self.training_rewards, linewidth=1)
        plt.title('Training reward over time')
        plt.ylabel('reward')
        plt.xlabel('iterations')

        plt.subplot('212')
        plt.plot(self.evaluation_rewards, linewidth=1)
        plt.title('Evaluation reward over time')
        plt.ylabel('reward')
        plt.xlabel('iterations')

        plt.show()

    # use the environments render method and print some additional info
    # to the console. permit user input for repeated demonstrations in a loop
    def demonstrate(self, evaluation_action):
        user_input = 'Y'
        while (user_input == 'Y'):
            observation = self.env.reset()
            done = False
            episode_reward = 0
            reward = 0
            step = 0
            while not done:
                print(f"Step: {step} | Cumulative Reward: {episode_reward}")
                step += 1
                print("RENDERING...")
                self.env.render()
                action = evaluation_action(observation)
                print('observation: ', observation)
                print('action: ', action)
                print('reward: ', reward)
                observation, reward, done, info = self.env.step(action)
                episode_reward += reward

            user_input = input('Enter Y for another demo: ')

