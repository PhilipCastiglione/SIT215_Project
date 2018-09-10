class Random():
    def __init__(self):
        super().__init__()

    # the training action is any random action from within the environment action space
    def action(self, env):
        return env.action_space.sample()

