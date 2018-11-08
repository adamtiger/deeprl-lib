

class Algorithm:
    
    def __init__(self, name, model, env):
        self.name = name
        self.model = model
        self.env = env

    def train(self, hypers):
        raise NotImplementedError

    def serialize(self, folder):
        raise NotImplementedError
    
    @staticmethod
    def deserialize(folder):
        raise NotImplementedError
