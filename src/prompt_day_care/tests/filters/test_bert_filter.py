import unittest
from unittest.mock import patch
from prompt_day_care.filters.bert_filter import BertFilter
import transformers
from torch import tensor

class MockBertClass:
    def __init__(self, hidden_states, device='cpu'):
        self.hidden_states = hidden_states
        self.device = device

    def __call__(self, *args, **kwargs):
        return self
    
    def eval(self):
        pass

    def to(self, device):
        pass
    

class TestBertFilter(unittest.TestCase):

    @patch('transformers.models.bert.modeling_bert.BertModel.from_pretrained')
    def test_valid_instantiation(self, mock_from_pretrained):
        filterer = BertFilter('bert-base-uncased')

    @patch('transformers.models.bert.modeling_bert.BertModel.from_pretrained')
    def test_threshold_over_1(self, mock_from_pretrained):
        with self.assertRaises(ValueError):
            filterer = BertFilter('bert-base-uncased', threshold=1.1)

    @patch('transformers.models.bert.modeling_bert.BertModel.from_pretrained')
    def test_threshold_under_0(self, mock_from_pretrained):
        with self.assertRaises(ValueError):
            filterer = BertFilter('bert-base-uncased', threshold=-0.1)

    @patch.object(transformers.models.bert.modeling_bert.BertModel, 'from_pretrained')
    def test_filter_identical_samples(self, mock_bert_model):
        mock_bert_model.return_value = MockBertClass([tensor([[[1.0, 1.0, 1.0]],
                                                              [[1.0, 1.0, 1.0]],
                                                              [[1.0, 1.0, 1.0]]])])

        filterer = BertFilter('bert-base-uncased')

        prompts = ['Identical prompt',
                   'Identical prompt',
                   'Identical prompt']

        filtered_prompts = filterer.filter_prompts(prompts)

        self.assertEqual(filtered_prompts, [False, True, True])

    @patch.object(transformers.models.bert.modeling_bert.BertModel, 'from_pretrained')
    def test_filter_different_samples(self, mock_bert_model):
        mock_bert_model.return_value = MockBertClass([tensor([[[1.0, 1.0, 1.0]],
                                                              [[-1.0, -1.0, -1.0]]])])

        filterer = BertFilter('bert-base-uncased')

        prompts = ['Prompt A',
                   'Prompt B']

        filtered_prompts = filterer.filter_prompts(prompts)

        self.assertEqual(filtered_prompts, [False, False])

    @patch.object(transformers.models.bert.modeling_bert.BertModel, 'from_pretrained')
    def test_filter_similar_samples(self, mock_bert_model):
        mock_bert_model.return_value = MockBertClass([tensor([[[1.0, 1.0, 1.0]],
                                                              [[0.97, 0.97, 0.97]],
                                                              [[-1.0, -1.0, -1.0]],
                                                              [[-0.97, -0.97, -0.97]]])])

        filterer = BertFilter('bert-base-uncased')

        prompts = ['Prompt A',
                   'Similar prompt A',
                   'Prompt B',
                   'Similar prompt B']

        filtered_prompts = filterer.filter_prompts(prompts)

        self.assertEqual(filtered_prompts, [False, True, False, True])


if __name__ == '__main__':
    unittest.main()