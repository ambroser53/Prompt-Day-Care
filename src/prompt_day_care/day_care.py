import os
from glob import glob
from prompt_day_care.genomes.genome import Genome
from prompt_day_care.utils.config import Config
from prompt_day_care.fitness_scorers.fitness_scorer import FitnessScorer
from prompt_day_care.mutators.mutator import Mutator
import re
from typing import List
import random
import numpy as np
from torch import manual_seed
from torch.cuda import manual_seed_all
import json


def get_next_run_number(path: str) -> int:
    runs = glob(os.path.join(path, 'RUN_[0-9]*'))
    dirs = [int(re.search(r'RUN_([0-9]+)$', run).group(1)) for run in runs if os.path.isdir(run)]
    return max(dirs) + 1 if dirs else 0

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    manual_seed_all(seed)


class DayCare:
    def __init__(self, mutators: List[Mutator], fitness_scorer: FitnessScorer, output_dir: str, **kwargs) -> None:
        if len(kwargs.get('population', [])) < 2 or kwargs.get('population_size', 0) < 2:
            raise ValueError('Must provide "population_size" or an "population" of at least size 2')
        
        if population := kwargs.pop('population', []):
            self.population = population
            kwargs.pop('population_size', None)
        else:
            self._initialise_population(kwargs.pop('population_size'), kwargs.pop('prompts_per_unit', 1))

        self.config = Config(**kwargs)

        self.mutators = mutators
        self.fitness_scorer = fitness_scorer

        # TODO: continue run if loaded from generation instead of making a new run
        run_number = get_next_run_number(output_dir)
        self.output_dir = os.path.join(output_dir, f'RUN_{run_number}')
        os.makedirs(self.output_dir)

        # TODO: Finish init function

    def _initialise_population(self, population_size: int, prompts_per_unit: int) -> None:
        pass

    @classmethod
    def from_generation(cls, path: str) -> 'DayCare':
        '''Loads a day care from a generation directory
        
        Args:
            path (str): The path to the generation directory
            
        Returns:
            DayCare: The day care loaded from the generation directory    
        '''
        if not os.path.exists(path) or not os.path.isdir(path):
            raise FileNotFoundError(f'No generation found at {path}')
        
        with open(os.path.join(path, 'config.json')) as f:
            config = json.load(f)
        
        genome_files = glob(os.path.join(path, 'genomes', '*.genome'))
        population = [Genome.from_disk(genome_file) for genome_file in genome_files]

        # TODO: load mutators, fitness scorer, output_dir, etc. and config
        return cls(population, **config)
    
    @classmethod
    def from_run(cls, path: str) -> 'DayCare':
        '''Loads the last generation of a day care from a run directory
        
        Args:
            path (str): The path to the run directory

        Returns:
            DayCare: The day care loaded from the run directory
        '''
        if not os.path.exists(path) or not os.path.isdir(path):
            raise FileNotFoundError(f'No run found at {path}')

        generations = glob(os.path.join(path, 'generation_*'))
        if len(generations) == 0:
            raise FileNotFoundError(f'No generations found at {path}')
        
        latest_generation = max(int(re.search(r'generation_([0-9]+)$', gen).group(1)) for gen in generations)
        return cls.from_generation(latest_generation)
    
    def save_generation(self) -> None:
        '''Saves the current generation to disk'''
        generation_dir = os.path.join(self.output_dir, f'generation_{self.config.generation_number}')

        os.makedirs(os.path.join(generation_dir, 'genomes'))
        for genome in self.population:
            genome.to_disk(generation_dir)

        self.config.to_disk(generation_dir)

        # TODO: save mutators, fitness scorer, etc.