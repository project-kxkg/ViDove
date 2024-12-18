import logging
import traceback
from time import sleep
import warnings
import openai

from tqdm import tqdm

from src.srt_util.srt import split_script

from .assistant import Assistant
from .basic_rag import BasicRAG
from .LLM import LLM
from .MTA import MTA

SUPPORT_LANG_MAP = {
    "EN": "English",
    "ZH": "Chinese",
    "ES": "Spanish",
    "FR": "France",
    "DE": "Germany",
    "RU": "Russian",
    "JA": "Japanese",
    "AR": "Arabic",
    "KR": "Korean",
}


class Translator:
    def __init__(
        self, model_name, src_lang, tgt_lang, domain, task_id, client, chunk_size=1000
    ):
        self.task_logger = logging.getLogger(f"task_{task_id}")
        self.task_logger.info("initializing translator")
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.domain = domain
        self.task_id = task_id
        self.system_prompt = self.prompt_selector()
        self.client = client
        self.srt = None

        if self.model_name == "Assistant":
            self.translator = Assistant(
                self.client, system_prompt=self.system_prompt, domain=domain
            )
        elif self.model_name in ["gpt-4o-mini", "gpt-4o"]:
            self.translator = LLM(
                self.client, self.model_name, system_prompt=self.system_prompt
            )
        elif self.model_name == "Multiagent":
            self.translator = MTA(
                self.client,
                "gpt-4o",
                self.domain,
                self.src_lang,
                self.tgt_lang,
                SUPPORT_LANG_MAP[self.tgt_lang],
                self.task_logger,
            )
        elif self.model_name == "RAG":
            self.translator = BasicRAG(
                self.src_lang,
                self.tgt_lang,
                self.task_logger,
                self.domain,
                self.model_name,
            )
        else:
            print(f"Unsupported model name: {self.model_name}")
            raise NotImplementedError

        self.task_logger.info(f"Using {self.model_name} as translation model")

    def set_srt(self, srt):
        self.srt = srt
        self.script_arr, self.range_arr = split_script(
            srt.get_source_only(), self.chunk_size
        )
        self.task_logger.info("SRT file set")

    def prompt_selector(self):
        try:
            src_lang = SUPPORT_LANG_MAP[self.src_lang]
            tgt_lang = SUPPORT_LANG_MAP[self.tgt_lang]
            assert src_lang != tgt_lang
        except:
            print("Unsupported language, is your abbreviation correct?")
            print(f"supported language map: {SUPPORT_LANG_MAP}")
            self.task_logger.info(
                f"Unsupported language detected: {src_lang} to {tgt_lang}"
            )

        prompt = f"""
            you are a translation assistant, your job is to translate a video in domain of {self.domain} from {src_lang} to {tgt_lang},
            you will be provided with a segement in {src_lang} parsed by line, where your translation text should keep the original
            meaning and the number of lines. DO NOT INCLUDE THE INDEX NUMBER IN YOUR TRANSLATION.  /n/n
            """
        self.task_logger.info(f"System Prompt: {prompt}")
        return prompt

    def translate(self, max_retries = 1):
        """
        Translates the given script array into another language using the chatgpt and writes to the SRT file.
, 
        This function takes a script array, a range array, a model name, a video name, and a video link as input. It iterates
        through sentences and range in the script and range arrays. If the translation check fails for five times, the function
        will attempt to resolve merge sentence issues and split the sentence into smaller tokens for a better translation.
        """

        if self.srt is None:
            raise ValueError("SRT file not set")

        if self.system_prompt is None:
            self.system_prompt = "你是一个翻译助理，你的任务是翻译视频，你会被提供一个按行分割的英文段落，你需要在保证句意和行数的情况下输出翻译后的文本。"
            self.task_logger.info(f"translation prompt: {self.system_prompt}")
        previous_length = 0
        for sentence, range_ in tqdm(zip(self.script_arr, self.range_arr)):
            # update the range based on previous length
            range_ = (range_[0] + previous_length, range_[1] + previous_length)

            print(f"now translating sentences {range_}")
            self.task_logger.info(f"now translating sentences {range_}")
            retry_count = 0
            while retry_count < max_retries:
                try:
                    translation = self.translator.send_request(sentence)
                    break  # Success - exit the loop
                except openai.BadRequestError as e:
                    retry_count += 1
                    # Access the content filter results
                    error_response = e.response.json()
                    filter_results = error_response['error']['innererror']['content_filter_result']
                    
                    # Extract categories where filtered is True
                    filtered_categories = [
                        category for category, details in filter_results.items()
                        if details.get('filtered') is True
                    ]
                    
                    # Optionally, you can also get categories with their severity
                    filtered_with_severity = {
                        category: details['severity']
                        for category, details in filter_results.items()
                        if details.get('filtered') is True
                    }
                    print(f"An error has occurred during translation (attempt {retry_count}/{max_retries}):", e)

                    print(traceback.format_exc())
                    self.task_logger.debug("An error has occurred during translation:", e)
                    
                    if retry_count < max_retries:
                        self.task_logger.info(
                            "Retrying... the script will continue after 30 seconds."
                        )
                        sleep(30)
                    else:
                        self.task_logger.warning(f"Max retries ({max_retries}) reached, skipping translation for: {sentence}")
                        self.task_logger.warning(f"Filtered categories: {' '.join(filtered_categories)} with severity: {' '.join(filtered_with_severity)}")
                        warnings.warn(f"Max retries ({max_retries}) reached, skipping translation for: {sentence}, please check if the video contains any {filtered_categories}")
                        translation = ""
                        
            self.task_logger.info(f"source text: {sentence}")
            self.task_logger.info(f"translate text: {translation}")
            self.srt.set_translation(translation, range_, self.model_name, self.task_id)