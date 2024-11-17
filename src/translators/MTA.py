from openai import AzureOpenAI
from logging import Logger
from .abs_api_model import AbsApiModel

class MTA(AbsApiModel):
    def __init__(self, client:AzureOpenAI, model_name:str, domain:str, source_language:str, target_language:str, target_country:str, logger:Logger, max_iterations:int=5) -> None:
        super().__init__()
        self.client = client
        if model_name in ["gpt-4o-mini", "gpt-4o"]:
            self.model_name = model_name
        else:
            raise NotImplementedError
        self.max_iterations = max_iterations
        self.domain = domain
        self.source_language = source_language
        self.target_language = target_language
        self.target_country = target_country
        self.logger=logger

    def send_request(self, input):
        current_iteration = 0
        history = None

        # Translator Agent
        translation_prompt = f"""This is an {self.source_language} to {self.target_language} translation in the field of {self.domain}, please provide the {self.target_language} translation for this text.\
        Do not provide any explanations or text apart from the translation. {self.source_language}: {input} {self.target_language}:"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": translation_prompt
                    }
                ]
            )
        history = response.choices[0].message.content

        while current_iteration <= self.max_iterations:
            # Suggestions Agent

            reflection_prompt = f"""Your task is to carefully read a content in the {history} and a translation from {self.source_language} to {self.target_language}, and then give constructive criticism and helpful suggestions to improve the translation. \
            The final style and tone of the translation should match the style of {self.target_language} colloquially spoken in {self.target_country}. When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
            (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
            (ii) fluency (by applying {self.target_language} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),
            (iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),
            (iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {self.target_language}).
            Write a list of specific, helpful and constructive suggestions for improving the translation.
            Each suggestion should address one specific part of the translation.
            Output only the suggestions and nothing else."""

            response =self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": reflection_prompt
                        }
                    ]
                )
            suggestion = response.choices[0].message.content

            self.logger.info(suggestion)

            # Editor Agent
            prompt = f"""Your task is to carefully read, then edit, a translation of the content in the {history} from {self.source_language} to {self.target_language}, taking into\
            account a list of expert suggestions and constructive criticisms.

            // Expert Suggestions:
                {suggestion}

            Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:
            (i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
            (ii) fluency (by applying {self.target_language} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions),
            (iii) style (by ensuring the translations reflect the style of the source text)
            (iv) terminology (inappropriate for context, inconsistent use), or
            (v) other errors.

            Output only the new translation and nothing else."""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                        }
                    ]
                )
            reply = response.choices[0].message.content
            self.logger.info(reply)
            if history == reply:
                return reply
            else:
                history = reply
                current_iteration += 1
        return reply
