import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score
from .fitness_scorer import FitnessScorer

class BinaryfBetaFitnessScorer(FitnessScorer):
    def __init__(self, task, beta):
        self.beta = beta
        self.task = task

    def worst_fitness(self):
        return 0

    def calculate(self, fitness_evals):
        df = self.prepare_evals(fitness_evals)
        return fbeta_score(df['gold_' + self.task + '_f'], df[self.task + '_f'], beta=self.beta)

    def compare(self, fitness1, fitness2):
        return fitness1 > fitness2

    def sort(self, iterable, descending=False):
        return sorted(iterable, key=lambda x: x['fitness']['best'], reverse=descending)

    def prepare_evals(self, fitness_evals):
        '''
        Prepares fitness evaluations for analysis by adding columns for fitness and gold fitness.
        :param fitness_evals: a dict with fitness evaluations, gold labels must be under 'gold_{task}_f' and model generations under '{task}_f'
        :return: dataframe with correct label formats
        '''
        df = pd.DataFrame.from_dict(fitness_evals)
        df[self.task + '_f'] = np.where(df[self.task] == ' Yes', 1, 0)
        df['gold_' + self.task + '_f'] = np.where(df['gold_' +self.task] == True, 1, 0)
        return df