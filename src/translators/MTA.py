from openai import OpenAI
from .abs_api_model import AbsApiModel
from .prompts import orignal_translationprompt,orignal_reflectionprompt,orignal_editorprompt
from .assistant import Assistant



class MTA(AbsApiModel):
    def __init__(self, client:OpenAI, model_name:str, domain:str, source_language:str, target_language:str, target_country:str, logger:str, system_prompt:str, max_iterations:int=5) -> None:
        super().__init__()
        self.client = client
        if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4o","assistant"]:
            self.model_name = model_name
        else:
            raise NotImplementedError
        self.max_iterations = max_iterations
        self.domain = domain
        self.source_language = source_language
        self.target_language = target_language
        self.target_country = target_country
        self.logger=logger
        self.system_prompt=system_prompt

    def send_request(self, input):
        current_iteration = 0
        history = None
        
        translation_prompt=fixed_translationprompt(self.source_language,self.target_language,self.domain,input)
        # Translator Agent
        
        if self.model_name == "assistant":
            translator = Assistant(self.client, system_prompt = self.system_prompt, domain = "SC2")
            history = translator.send_request(input)
        else:
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
            reflection_prompt=fixed_reflectionprompt(self.source_language,self.target_language,history,self.target_country)
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
            editor_prompt=fixed_editorprompt(self.source_language,self.target_language,history,suggestion)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": editor_prompt
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
