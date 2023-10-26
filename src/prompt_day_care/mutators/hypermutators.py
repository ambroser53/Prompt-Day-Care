from prompt_day_care.mutators.mutator import Mutator
from prompt_day_care.genomes import Genome
from typing import Tuple


class HyperMutator(Mutator):
    def __init__(self, *args):
        super().__init__(args)

    def mutate(self, genome: Genome) -> Genome:
        raise NotImplementedError
    
    def test(self, genome, new_mutation_prompt):
        unit = {
            'mutation_prompt': self.random.choice(self.mutation_prompts),
            'thinking_style': genome.thinking_style,
            'task_prompt': genome.task_prompt,
        }
        raise NotImplementedError
    

class HypermutatorZeroOrderDirect(HyperMutator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def mutate(self, **kwargs) -> Tuple[Genome, Genome]:
        unit = {
            'thinking_style': self.random.choice(self.thinking_styles),
            'task_description': self.task_description
        }
        # TODO: Finish mutate function
        raise NotImplementedError
 

class HypermutatorZeroOrderDirect(HyperMutator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def mutate(self, **kwargs) -> Tuple[Genome, Genome]:
        unit = {
            'thinking_style': self.random.choice(self.thinking_styles),
            'task_description': self.task_description
        }
        # TODO: Finish mutate function
        raise NotImplementedError