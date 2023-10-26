from typing import List
from prompt_day_care.genomes import Genome
from prompt_day_care.fitness_scorers.fitness_scorer import FitnessScorer
from prompt_day_care.mutators.mutator import Mutator
import random

class BinaryTournamentEvolve:
    def __init__(self, mutators: List[Mutator], fitness_scorer: FitnessScorer, seed: int = 42):
        self.mutators = mutators
        self.fitness_scorer = fitness_scorer
        self.random = random.Random(seed)

    def evolve(self, population: List[Genome]) -> List[Genome]:
        ''' Evolves a population using binary tournament selection.

        Args:
            population (List[Genome]): The population to evolve
            mutators (List[Mutator]): The mutators to use
            fitness_scorer (FitnessScorer): The fitness scorer to use

        Returns:
            The evolved population
        '''
        self.random.shuffle(population)
        pairs = list(zip(population[::2], population[1::2]))
        superior_population = [self.fitness_scorer.compare(pair) for pair in pairs]

        next_population = []
        for genome in superior_population:
            # TODO: See if batching this works better with async
            mutator = self.random.choice(self.mutators)
            next_population.extend(mutator.mutate(genome))

        return next_population


                

            
        
