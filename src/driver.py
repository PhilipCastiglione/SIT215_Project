import matplotlib
# using an alternative backend to macos gui driver, because there is an issue with
# matplotlib, virtualenv and macos: https://github.com/pypa/virtualenv/issues/54
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Driver:
    def __init__(self, params):
        self.training_episodes = params['training_episodes']
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

    def run(self, training_action, update, evaluation_action):
        for i in range(self.training_episodes):
            if ((i + 1) % 1000 == 0):
                print("progress: {}%".format(100 * (i + 1) // self.training_episodes))
            self.train(training_action, update)
            self.evaluate(evaluation_action)
            # TODO: remove
            #if ((i + 1) % 5000 == 0):
                #print("current q_table")
                #[print(line) for line in self.agent.q_table]

        self.plot()
        #self.demonstrate(evaluation_action)

    def train(self, training_action, update):
        observation = self.env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = training_action(observation)
            observation, reward, done, info = self.env.step(action)
            episode_reward += reward
            update(observation, action, reward)
        self.training_rewards.append(episode_reward)

    def evaluate(self, evaluation_action):
        observation = self.env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = evaluation_action(observation)
            observation, reward, done, info = self.env.step(action)
            episode_reward += reward
        self.evaluation_rewards.append(episode_reward)

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

    def demonstrate(self, evaluation_action):
        user_input = 'Y'
        while (user_input == 'Y'):
            observation = self.env.reset()
            done = False
            step = 0
            while not done:
                print(f"Step: {str(step)}")
                step += 1
                self.env.render()
                action = evaluation_action(observation)
                observation, reward, done, info = self.env.step(action)

            user_input = input('See demo? Y/N')

