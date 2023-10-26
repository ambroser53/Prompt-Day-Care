from prompt_day_care.genomes import Genome
from typing import Tuple, List
import random

class Mutator:
    def __init__(self, model, mutation_prompts: List[str], thinking_styles: List[str], task_prompts: List[str], task_description: str, seed: int = 42):
        self.model = model
        self.mutation_prompts = mutation_prompts
        self.thinking_styles = thinking_styles
        self.task_prompts = task_prompts
        self.task_descriptions = task_description
        self.random = random.Random(seed)

    def mutate(self, **kwargs) -> Tuple[Genome, Genome]:
        raise NotImplementedError