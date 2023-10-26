from prompt_day_care.mutators.mutator import Mutator
from prompt_day_care.genomes import Genome
from typing import Tuple, Dict


class ZeroOrderDirect(Mutator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def mutate(self, **kwargs) -> Tuple[Genome, Genome]:
        unit = {
            'mutation_prompt': self.random.choice(self.mutation_prompts),
            'thinking_style': self.random.choice(self.thinking_styles),
            'task_description': self.task_description
        }
        # TODO: Finish mutate function
        raise NotImplementedError


class FirstOrderDirect(Mutator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def mutate(self, genome: Genome, **kwargs) -> Tuple[Genome, Genome]:
        unit = {
            'mutation_prompt': self.random.choice(self.mutation_prompts),
            'task_prompt': str(genome)
        }
        # TODO: Finish mutate function
        raise NotImplementedError
    

class Lamarckian(Mutator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def mutate(self, genome: Genome, correct_reasonings: Dict[str, str], **kwargs) -> Tuple[Genome, Genome]:
        unit = {
            'mutation_prompt': self.random.choice(self.mutation_prompts),
            'task_prompt': str(genome)
        }
        raise NotImplementedError