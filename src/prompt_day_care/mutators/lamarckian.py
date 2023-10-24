from mutators.mutator import Mutator
from genomes.genome import Genome
from typing import Tuple, Dict

class FirstOrderDirect(Mutator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def mutate(self, genome: Genome, correct_reasonings: Dict[str, str], **kwargs) -> Tuple[Genome, Genome]:
        
        unit = {
            'mutation_prompt': self.random.choice(self.mutation_prompts),
            'task_prompt': str(genome)
        }
        raise NotImplementedError