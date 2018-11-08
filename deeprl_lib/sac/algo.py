from deeprl_lib.core import Algorithm


class SACalgo(Algorithm):

    def __init__(self, model, env):
        super(SACalgo, self).__init__('sac', model, env)
        
    def train(self, hypers):
        pass
    
    def serialize(self, folder):
        pass

    @staticmethod
    def deserialize(folder):
        pass