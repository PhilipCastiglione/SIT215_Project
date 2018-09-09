import gym
from src.driver import Driver
from src.agents.random import Random
from src.agents.qlearner import Qlearner

# entry point; you can change the hyperparameters here
if __name__ == '__main__':
    env_names = [
        'Taxi-v2',
        'CartPole-v1'
    ]
    envs = [gym.make(name) for name in env_names]

    qlearner_hyperparameters = {
        'alpha': 0.1,
        'gamma': 0.6,
        'epsilon': 0.1,
    }

    for env in envs:
        parameters = qlearner_hyperparameters
        num_states, num_actions = extract_shape(env)
        parameters['num_states'] = num_states
        parameters['num_actions'] = num_actions

        agents = [
            Random(),
            Qlearner(parameters),
        ]

        for agent in agents:
            driver_params = {
                'debug': True,
                'training_episodes': 20,
                'evaluation_episodes': 10,
                'env': env,
                'agent': agent,
            }
            driver = Driver(driver_params)
            driver.run()

# this is because it appears that different Spaces do not consistently
# define shape on the base Space class, some use the space object and
# some don't.
#
# the shape we need is:
#   - num_states
#   - num_actions
def extract_shape(env):
    if hasattr(env.observation_space, 'n'):
        num_states = env.observation_space.n
    else:
        num_states = env.observation_space.shape[0]

    if hasattr(env.action_space, 'n'):
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    return num_states, num_actions

