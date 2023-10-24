import unittest
from prompt_day_care.utils.config import Config
import os
import shutil
import json

class TestConfig(unittest.TestCase):

    def tearDown(self):
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')
    
    def test_empty_instantiation(self):
        Config()

    def test_instantiation_with_kwargs(self):
        config = Config(
            key1='value1',
            key2='value2',
        )

        self.assertEqual(config.key1, 'value1')
        self.assertEqual(config.key2, 'value2')

    def test_load_valid_config(self):
        os.mkdir('test_data')
        with open('test_data/config.json', 'w+') as f:
            json.dump({'test_key': 'test_value'}, f, indent=4)

        config = Config.load_config('test_data/config.json')
        self.assertEqual(config.test_key, 'test_value')

    def test_load_invalid_config(self):
        with self.assertRaises(FileNotFoundError):
            Config.load_config('test_data/config.json')

    def test_save_config(self):
        os.mkdir('test_data')
        config = Config(
            key1='value1',
            key2='value2',
        )

        config.save_config('test_data/config.json')
        with open('test_data/config.json', 'r') as f:
            self.assertEqual(json.load(f), {'key1': 'value1', 'key2': 'value2'})

    def test_pop(self):
        config = Config(
            key1='value1',
            key2='value2',
        )

        self.assertEqual(config.pop('key1'), 'value1')
        self.assertEqual(config.key2, 'value2')

    def test_add(self):
        config = Config(
            key1='value1',
            key2='value2',
        )

        config.add('key3', 'value3')
        self.assertEqual(config.key3, 'value3')

    
if __name__ == '__main__':
    unittest.main()