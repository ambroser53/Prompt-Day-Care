from typing import Optional

class Template:
    @classmethod
    def generate_prompt(self, **kwargs):
        raise NotImplementedError

class AlpacaTemplate:
    template_with_input = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    template_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    
    @classmethod
    def generate_prompt(self, instruction: str, input: Optional[str] = None, label: Optional[str] = None):
        if input:
            res = self.template_with_input.format(instruction=instruction, input=input)
        else:
            res = self.template_no_input.format(instruction=instruction)

        if label:
            res += label

        return res