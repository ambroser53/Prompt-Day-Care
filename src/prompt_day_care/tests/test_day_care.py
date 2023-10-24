import unittest
from unittest.mock import patch
import os
import shutil
from prompt_day_care.day_care import (
    get_next_run_number,
    DayCare
)

class TestGetNextRunNumber(unittest.TestCase):
    def setUp(self):
        os.makedirs('test_data')

        for i in range(3):
            os.makedirs(f'test_data/RUN_{i}')

    def tearDown(self):
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')
    
    def test_no_dirs(self):
        shutil.rmtree('test_data')
        self.assertEqual(get_next_run_number('test_data'), 0)

    def test_single_digit_runs(self):
        self.assertEqual(get_next_run_number('test_data'), 3)

    def test_double_digit_runs(self):
        os.mkdir(f'test_data/RUN_{10}')
        self.assertEqual(get_next_run_number('test_data'), 11)

    def test_max_run_is_dir(self):
        open(f'test_data/RUN_{2}/RUN_{5}', 'w+').close()
        self.assertEqual(get_next_run_number('test_data'), 3)

class TestDayCare(unittest.TestCase):
    
    def setUp(self):
        os.makedirs('test_data/RUN_0/generation_0')

        # TODO: create test genome files
        # 

    def tearDown(self):
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')

    def test_no_population_and_no_population_size(self):
        with self.assertRaises(ValueError):
            day_care = DayCare(mutators=[], fitness_scorer=None, output_dir='test_data')

    def test_population_less_than_2(self):
        with self.assertRaises(ValueError):
            day_care = DayCare(mutators=[], fitness_scorer=None, output_dir='test_data', population=[1])

    def test_population_size_less_than_2(self):
        with self.assertRaises(ValueError):
            day_care = DayCare(mutators=[], fitness_scorer=None, output_dir='test_data', population_size=1)

    @patch('prompt_day_care.day_care.DayCare._initialise_population')
    def test_population_size_and_population(self, mock_initialise_population):
        day_care = DayCare(mutators=[], fitness_scorer=None, output_dir='test_data', population_size=2, population=[1, 2])
        assert mock_initialise_population.call_count == 0
