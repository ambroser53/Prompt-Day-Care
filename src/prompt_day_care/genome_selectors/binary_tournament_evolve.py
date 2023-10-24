from typing import List, Tuple
from prompt_day_care.genomes.genome import Genome
import random

class BinaryTournamentEvolve:
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def evolve(self, population: List[Genome], fitness_scorer) -> List[Genome]:
        ''' Evolves a population using binary tournament selection.

        Args:
        '''
        self.random.shuffle(population)
        pairs = list(zip(population[::2], population[1::2]))
        superior_population = [fitness_scorer.compare(pair) for pair in pairs]

                

            
        
