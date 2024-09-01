import os
import logging
from openai import OpenAI
from src.translators.LLM import LLM
from src.translators.MTA import MTA
from evaluators.comet import CometEvaluator
import pandas as pd

from datasets import load_dataset

logger = logging.getLogger(__name__)

DATASETS = ["hgissbkh/WMT23-Test"] # List of datasets to evaluate on, taken from HuggingFace Datasets
BASEMODELS = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"] # List of models to evaluate on, taken from OpenAI API

DEFAULT_SYSTEM_PROMPT = """
You are one expert translator and your task is to translate the following text from {source_lang} to {target_lang}:

You should only output the translated text. without any format or markdown text.
Your translated text:
"""

OUTPUT_PATH = "results"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

client = OpenAI() # Initialize the OpenAI client

comet_evaluator = CometEvaluator("Unbabel/XCOMET-XL", logger) # Initialize the Comet evaluator

test_df = load_dataset(DATASETS[0])["test"].to_pandas() # Load the dataset

for dataset in DATASETS:
    for model in BASEMODELS:
        # Load the dataset
        test_df = load_dataset(dataset)["test"].to_pandas()
        source_target_pairs = test_df["lp"].unique()
        test_df = test_df.sample(10)

        results = {}
        for source_target_pair in source_target_pairs:
            source_lang, target_lang = source_target_pair.split("-")
            print(f"Evaluating in {model} as basemodel on {source_lang} to {target_lang} translation")

            system_prompt = DEFAULT_SYSTEM_PROMPT.format(source_lang=source_lang, target_lang=target_lang)
            print("System prompt:", system_prompt)

            # Load the translator
            llm_translator = LLM(client, model, system_prompt)
            mta_translator = MTA(client, model, "General", source_lang, target_lang, "US", logger)
            # assistant_translator = Assistant(client, system_prompt, domain="SC2") No need to test this agent since its general testing

            results[source_target_pair] = {}

            sub_set = test_df[test_df["lp"] == source_target_pair]
            for index, row in sub_set.iterrows():
                input_text = row["src"]
                reference = row["ref"]
                print("Input text:", input_text)

                # Send the request to the translator
                llm_response = llm_translator.send_request(input_text)
                mta_response = mta_translator.send_request(input_text)

                results[source_target_pair][index] = {
                    "src": input_text,
                    "reference": reference,
                    "llm_response": llm_response,
                    "mta_response": mta_response
                }      
        # Save the results
        with open(f"results_{model}_{dataset}.json", "w") as f:
            f.write(results)


