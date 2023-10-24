from mutators.mutator import Mutator
from genomes.genome import Genome
from typing import Tuple

class EDALineage(Mutator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def mutate(self, genome, *args) -> Tuple[Genome, Genome]:
        # TODO: Work out how to implement this
        raise NotImplementedError