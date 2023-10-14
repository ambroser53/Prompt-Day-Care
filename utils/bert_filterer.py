import torch
from transformers import BertTokenizer, BertModel
from typing import List

class BertFilterer:
    '''Class to filter prompts based on their similarity to one another, using the [CLS] token of a BERT model'''
    def __init__(self, model_name_or_path: str, threshold: float = 0.95):
        self.model = BertModel.from_pretrained(model_name_or_path, output_hidden_states=True)
        self.model.eval()

        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

        self.threshold = threshold

    def filter_prompts(self, prompts: List[str]) -> List[str]:
        '''Randomly reorders prompts, calculats similarity matrix and filters out prompts that are too similar to
        those at earlier indices in the matrix.

        Args:
            prompts: List of prompts to filter

        Returns:
            List of prompts that are sufficiently distinct from one another
        '''

        # Tokenise
        batch = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding=True)

        # Extract hidden states
        with torch.no_grad():
            outputs = self.model(batch['input_ids'].to(self.model.device), batch['attention_mask'].to(self.model.device))
            hidden_states = outputs.hidden_states

        # Extract the hidden states corresponding to the [CLS] tokens
        cls_states = hidden_states[-1][:, 0, :]

        # Create matrix of cosine similarities
        cls_norm = cls_states.norm(dim=1)[:, None]
        cls_clamp = cls_states / torch.clamp(cls_norm, min=1e-10)
        similarity_matrix = torch.mm(cls_clamp, cls_clamp.transpose(0, 1)).triu(diagonal=1)

        similar_idx = (similarity_matrix >= self.threshold).any(dim=0)

        # Decode the distinct prompts
        return list(similar_idx)

    def __call__(self, *args, **kwargs):
        return self.filter_prompts(*args, **kwargs)
    

if __name__ == '__main__':
    unfiltered_prompts = ['What is the capital of France?',
                           'What is the capital of Germany?',
                           'There are 1 million plants in the house',
                           'What is the capital of Germany?']
    
    print(unfiltered_prompts)

    filter = BertFilterer('bert-base-uncased')
    filtered_prompts = filter.filter_prompts(unfiltered_prompts)
    
    print(filtered_prompts)