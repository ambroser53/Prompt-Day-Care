from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_templates import Template
import lmql
from typing import List, Union, Dict
from torch.utils.data import DataLoader
from datasets import Dataset


class Model:
    def generate(self, **kwargs):
        raise NotImplementedError
    

class HFModel:
    def __init__(self, model_name_or_path: str, template: Template = None, batch_size: int = 1):
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.batch_size = batch_size
        self.template = template

        self.collator = DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt")

    def get_batches(self, samples):
        formatted_samples = list(map(lambda sample: AlpacaTemplate.generate_prompt(**sample), samples))
        tokenized_samples = tokenizer(formatted_samples, truncation=True, padding=True)

        dataset = Dataset.from_dict(tokenized_samples)
        return DataLoader(tokenized_samples, batch_size=self.batch_size, shuffle=False, collate_fn=self.collator)


    def generate(self, samples: Union[Dict[str, str], List[Dict[str, str]]]) -> List[str]:
        if isinstance(samples, dict):
            samples = [samples]

        outputs = []
        for batch in self.get_batches(samples):
            outputs.extend(self.model.generate(**batch))

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return list(map(lambda sample: AlpacaTemplate.get_response(sample), decoded))
        



class LMQLModel:
    def __init__(self, model_name_or_path):
        pass


