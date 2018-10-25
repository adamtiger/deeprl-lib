

class Model:

    def fit(self, x, y):
        raise NotImplementedError
    
    def predict(self, x):
        raise NotImplementedError

    def serialize(self, folder):
        raise NotImplementedError

    @staticmethod
    def deserialize(folder):
        raise NotImplementedError
        