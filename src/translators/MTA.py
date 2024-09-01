from openai import OpenAI
from .abs_api_model import AbsApiModel
from prompts import translation_prompt, reflection_prompt, editor_prompt

class MTA(AbsApiModel):
    def __init__(self, client:OpenAI, model_name:str, domain:str, source_language:str, target_language:str, target_country:str, logger:str, max_iterations:int=5) -> None:
        super().__init__()
        self.client = client
        if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
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
