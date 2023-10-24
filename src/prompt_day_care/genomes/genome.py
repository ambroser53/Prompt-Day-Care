import os
import json

class Genome:
    def __init__(self, fitness_scorer, system_prompt, task_prompt, reasoning_prompt):
        self.fitness_scorer = fitness_scorer
        self.id = hash(self)
        self.system_prompt = system_prompt
        self.task_prompt = task_prompt
        self.reasoning_prompt = reasoning_prompt
    
    def determine_fitness(self, fitness_evals):
        # TODO: Should this be here?
        pass

    def to_disk(self, path: str) -> None:
        save_file = os.path.join(path, f'{self.id}.genome')

        with open(save_file, 'w') as f:
            json.dump(self.__dict__, f)
    
    @classmethod
    def from_disk(cls, path: str) -> 'Genome':
        if not os.path.exists(path):
            raise FileNotFoundError(f'Genome file {path} does not exist.')
        
        with open(path, 'r') as f:
            genome = json.load(f)

        return Genome(**genome)
    
    def __repr__(self):
        return f'Genome({self.fitness_scorer}, SYSTEM PROMPT: {self.system_prompt}, \
            TASK PROMPT: {self.task_prompt}, REASONING PROMPT: {self.reasoning_prompt})'