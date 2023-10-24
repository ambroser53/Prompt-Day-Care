
class FitnessScorer:
    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def worst_fitness(self):
        raise NotImplementedError('worst_fitness not implemented for this fitness test, implement in child class. Should return a single numerical fitness value.')

    def calculate(self, fitness_evals):
        raise NotImplementedError('calculate not implemented for this fitness test, implement in child class. Should return a single numerical fitness value.')

    def compare(self, fitness1, fitness2):
        raise NotImplementedError('compare not implemented for this fitness test, implement in child class. Should return True if fitness1 is better than fitness2, False otherwise.')

    def sort(self, iterable, descending=False):
        raise NotImplementedError('sort not implemented for this fitness test, implement in child class. Should return a sorted list of dictionaries based on the fitness key')
