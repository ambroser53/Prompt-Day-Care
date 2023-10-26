from prompt_day_care.mutators.mutator import Mutator
from prompt_day_care.genomes import Genome
from typing import Tuple


class EDAMutator(Mutator):
    def __init__(self, *args) -> None:
        super().__init__(*args)


class EDARandom(EDAMutator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def mutate(self, **kwargs) -> Tuple[Genome, Genome]:
        # TODO: Work out how to implement this
        raise NotImplementedError
    

class EDARanked(EDAMutator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def mutate(self, **kwargs) -> Tuple[Genome, Genome]:
        # TODO: Work out how to implement this
        raise NotImplementedError
    

class EDALineage(EDAMutator):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def mutate(self, genome, *args) -> Tuple[Genome, Genome]:
        # TODO: Work out how to implement this
        raise NotImplementedError