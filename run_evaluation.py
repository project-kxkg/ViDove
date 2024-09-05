# %%
import os
import logging
from pathlib import Path
from openai import OpenAI
from src.translators.LLM import LLM
from src.translators.MTA import MTA
from evaluators.comet import CometEvaluator
import pandas as pd
import json

from datasets import load_dataset

# Setup the init parameters
logger = logging.getLogger(__name__)

DATASETS = ["hgissbkh/WMT23-Test"] # List of datasets to evaluate on, taken from HuggingFace Datasets
BASEMODELS = ["gpt-3.5-turbo"] # List of models to evaluate on, taken from OpenAI API "gpt-4", "gpt-4o"

DEFAULT_SYSTEM_PROMPT = """
You are one expert translator and your task is to translate the following text from {source_lang} to {target_lang}:

You should only output the translated text. without any format or markdown text.
Your translated text:
"""

OUTPUT_PATH = Path ("results") # Path to save the results

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

client = OpenAI() # Initialize the OpenAI client

comet_evaluator = CometEvaluator("Unbabel/wmt22-comet-da", logger, gpus=0) # Initialize the Comet evaluator 
#Using the default model for development, update to the latest model for the real evaluation Unbabel/XCOMET-XXL (needs to grant access to the model)

# %%
# Get the responses from two translators
for dataset in DATASETS:
    for model in BASEMODELS:
        # Load the dataset
        test_df = load_dataset(dataset)["test"].to_pandas()
        source_target_pairs = test_df["lp"].unique()
        test_df = test_df.sample(5)

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
            output_path = OUTPUT_PATH / dataset.split("/")[1] / model / source_target_pair
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(output_path / "results.json", "w") as f:
                json.dump(results[source_target_pair], f)


# %%
# Run evaluation on the responses by batch size of 8
# Load the results
RESULT_PATH = Path("results")
datasets = os.listdir(RESULT_PATH)
BASEMODELS = ["gpt-3.5-turbo"]

for dataset in datasets:
    for model in BASEMODELS:
        results_list = (RESULT_PATH / dataset / model).rglob("*.json")
        results = {}
        for results_file in results_list:
            with open(results_file, "r") as f:
                results.update(json.load(f))

        llm_responses = []
        mta_responses = []
        for index, response in results.items():
            llm_responses.append({"src": response["src"], "mt": response["llm_response"], "ref": response["reference"]})
            mta_responses.append({"src": response["src"], "mt": response["mta_response"], "ref": response["reference"]})

        # Evaluate the responses
        llm_comet_scores = comet_evaluator.evaluate(llm_responses)
        mta_comet_scores = comet_evaluator.evaluate(mta_responses)

        # Save the scores
        scores_df = pd.DataFrame({
            "llm_comet_scores": llm_comet_scores,
            "mta_comet_scores": mta_comet_scores
        })

        if not os.path.exists(OUTPUT_PATH / "scores"):
            os.makedirs(OUTPUT_PATH / "scores")

        scores_df.to_csv(OUTPUT_PATH / "scores" / f"{dataset}_{model}.csv")



# %%
