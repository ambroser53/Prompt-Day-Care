from prompt_day_care.day_care import DayCare
from prompt_day_care.models import HFModel
from prompt_day_care.prompt_templates import AlpacaTemplate


def main():
    model = HFModel('EleutherAI/pythia-70m')
