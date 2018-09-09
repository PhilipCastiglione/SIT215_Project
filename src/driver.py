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

    def run(self):
        self.train()
        self.evaluate()

    def train(self):
        '''
        # TODO
        # this is required to generate different starting positions
        self.seed = random.randint(0, 2**32 - 1)
        prng.seed(seed)
        print(sys._getframe(1).f_code.co_name)
        print(f'running {funcName} with random seed {seed}')
        '''

        for _ in range(self.training_episodes):
            observation = self.env.reset()
            done = False
            step = 0
            while not done:
                if (self.debug):
                    self.env.render()
                    print(f"observation: {observation}")

                action = self.agent.training_action(self.env, observation)
                observation, reward, done, info = self.env.step(action)
                self.agent.update(observation, action, reward)
                step += 1

                if done: # OpenAI gym enforces a maximum of 200 steps if not solved
                    if (self.debug):
                        print(f"Episode finished after {step} timesteps")
                    break

    def evaluate(self):
        rewards = []
        for _ in range(self.evaluation_episodes):
            observation = self.env.reset()
            done = False
            step = 0
            episode_reward = 0
            while not done:
                action = self.agent.evaluation_action(self.env, observation)
                observation, reward, done, info = self.env.step(action)
                episode_reward += reward
                step += 1

                if done: # OpenAI gym enforces a maximum of 200 steps if not solved
                    break
            rewards.append(episode_reward)
        average_reward = sum(rewards) / self.evaluation_episodes
        print(f"average reward level over {self.evaluation_episodes} episodes: {average_reward}")
