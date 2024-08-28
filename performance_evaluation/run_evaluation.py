from openai import OpenAI
from src.translators.LLM import LLM
from src.translators.MTA import MTA
from src.translators.assistant import Assistant

from datasets import load_dataset

DATASETS = ["hgissbkh/WMT23-Test"] # List of datasets to evaluate on, taken from HuggingFace Datasets
BASEMODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"] # List of models to evaluate on, taken from OpenAI API

client = OpenAI() # Initialize the OpenAI client

for dataset in DATASETS:
    for model in BASEMODELS:
        # Load the dataset
        dataset = load_dataset(dataset)

        

        # Initialize the models
        llm = LLM(client, model_name=model, system_prompt="Translate the following text to English:", temp=0.15)
        mta = MTA()
        # Evaluate the models



