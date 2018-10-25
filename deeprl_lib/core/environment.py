

class Environment:
    
    def __init__(self):
        self.record_video = False
        self.verbose = False
        self.log_freq = 100
        self.folder = None
        self.max_steps = 500

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
    
    @staticmethod
    def env_from_gym(gym_env):
        return GymEnvironment(gym_env)

# TODO: implement functions with every details required
class GymEnvironment(Environment):
    
    def __init__(self, env):
        super(GymEnvironment, self).__init__()
        self.env = env

    def step(self, action):
        obs, rw, done, _ = self.env.step(action)
        return obs, rw, done

    def reset(self):
        return self.env.reset()


        