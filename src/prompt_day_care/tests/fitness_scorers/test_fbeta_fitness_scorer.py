import unittest
import prompt_day_care.fitness_scorers.fbeta_fitness_scorer as FBetaFitnessScorer

class TestFBetaFitnessScorer(unittest.TestCase):

    def test_beta_less_than_zero(self):
        self.assertRaises(ValueError, FBetaFitnessScorer, 'test', -1)

    def test_beta_equal_to_zero(self):
        try:
            scorer = FBetaFitnessScorer('test', 0)
        except ValueError:
            self.fail('Raised ValueError unexpectedly')

    def test_worst_fitness(self):
        scorer = FBetaFitnessScorer('test', 1)
        self.assertEqual(scorer.worst_fitness(), 0)

    def test_calculate(self):
        scorer = FBetaFitnessScorer('test', 1)
        fitness_evals = {
            'test': [' Yes', ' No', ' Yes', ' No'],
            'gold_test': [True, False, False, True]
        }
        self.assertEqual(scorer.calculate(fitness_evals), 0.5)

    def test_compare_1_greater_than_2(self):
        scorer = FBetaFitnessScorer('test', 1)
        self.assertTrue(scorer.compare(1, 0))

    def test_compare_1_less_than_2(self):
        scorer = FBetaFitnessScorer('test', 1)
        self.assertFalse(scorer.compare(0, 1))

    def test_sort_already_in_order(self):
        scorer = FBetaFitnessScorer('test', 1)
        iterable = [{'fitness': {'best': 0}}, {'fitness': {'best': 1}}]
        self.assertEqual(scorer.sort(iterable), iterable)

    def test_sort_not_in_order(self):
        scorer = FBetaFitnessScorer('test', 1)
        iterable = [{'fitness': {'best': 1}}, {'fitness': {'best': 0}}]
        self.assertEqual(scorer.sort(iterable), [{'fitness': {'best': 0}}, {'fitness': {'best': 1}}])

    def test_sort_not_in_order_descending(self):
        scorer = FBetaFitnessScorer('test', 1)
        iterable = [{'fitness': {'best': 1}}, {'fitness': {'best': 0}}]
        self.assertEqual(scorer.sort(iterable, descending=True), [{'fitness': {'best': 1}}, {'fitness': {'best': 0}}])

    def test_prepare_evals(self):
        scorer = FBetaFitnessScorer('test', 1)
        fitness_evals = {
            'test': [' Yes', ' No', ' Yes', ' No'],
            'gold_test': [True, False, False, True]
        }
        df = scorer.prepare_evals(fitness_evals)
        self.assertEqual(df['test_f'].tolist(), [1, 0, 1, 0])
        self.assertEqual(df['gold_test_f'].tolist(), [1, 0, 0, 1])


if __name__ == '__main__':
    unittest.main()