from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_templates import Template
import lmql

class Model:
    def generate(self, **kwargs):
        raise NotImplementedError
    

class HFModel:
    def __init__(self, model_name_or_path: str, template: Template = None):
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.template = template

    def generate(self, stuff):
        prompt = self.template.generate_prompt(stuff)

        self.model.generate(prompt)


class LMQLModel:
    def __init__(self, model_name_or_path):
        pass
