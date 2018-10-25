

class Agent:
    
    def __init__(self, name):
        self.name = name

    def train(self, model, env, hypers):
        raise NotImplementedError

    def eval(self, model, env, hypers):
        raise NotImplementedError

    def serialize(self, folder):
        raise NotImplementedError
    
    @staticmethod
    def deserialize(folder):
        raise NotImplementedError
