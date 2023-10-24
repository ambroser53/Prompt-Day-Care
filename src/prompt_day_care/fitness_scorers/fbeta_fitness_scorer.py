import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score
import pandas as pd
from fitness_scorers.fitness_scorer import FitnessScorer
from typing import Union

class BinaryfBetaFitnessScorer(FitnessScorer):
    def __init__(self, task: str, beta: float):
        if not beta >= 0:
            raise ValueError('Beta must be greater than 0')
        
        self.beta = beta
        self.task = task

    def worst_fitness(self) -> Union[int, float]:
        return 0

    def calculate(self, fitness_evals):
        df = self.prepare_evals(fitness_evals)
        return fbeta_score(df[f'gold_{self.task}_f'], df[f'{self.task}_f'], beta=self.beta)      

    def prepare_evals(self, fitness_evals):
        df = pd.DataFrame.from_dict(fitness_evals)
        df[f'{self.task}_f'] = np.where(df[self.task] == ' Yes', 1, 0)
        df[f'gold_{self.task}_f'] = np.where(df[f'gold_{self.task}'] == True, 1, 0)
        return df