from mutators.mutator import Mutator
import random


class ZeroOrderDirect(Mutator):
    def __init__(self, model, seed: int = 42):
        super().__init__(model)
        self.random = random.Random(seed)

    def mutate(self, *args, **kwargs):
        pass