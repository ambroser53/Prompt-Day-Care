from typing import Union, List
from prompt_day_care.genomes.genome import Genome

class FitnessScorer:
    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def worst_fitness(self) -> Union[int, float]:
        raise NotImplementedError('worst_fitness not implemented for this fitness test, implement in child class. Should return a single numerical fitness value.')

    def calculate(self, fitness_evals):
        raise NotImplementedError('calculate not implemented for this fitness test, implement in child class. Should return a single numerical fitness value.')

    def compare(self, genomes: List[Genome]) -> Genome:
        '''Compares a list of genomes and returns the best one
        
        Args:
            genomes: A list of genomes to compare
            
        Returns:
            The best genome
        '''
        return max(genomes, key=lambda genome: genome.fitness)

    def sort(self, genomes: List[Genome], descending=False) -> List[Genome]:
        '''Sorts a list of genomes by fitness

        Args:
            genomes: A list of genomes to sort
            descending: Whether to sort in descending order (default: False)

        Returns:
            The sorted list of genomes
        '''
        return sorted(genomes, key=lambda x: x.fitness, reverse=descending)
