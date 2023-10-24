
class Mutator:
    def __init__(self, model):
        self.model = model

    def mutate(self, genome):
        raise NotImplementedError
