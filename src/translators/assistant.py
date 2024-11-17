import time

from openai import AzureOpenAI

from .abs_api_model import AbsApiModel

SUPPORT_DOMAIN = ["SC2"]
ID_MAP = {"SC2": "asst_48Ha6WDx8Kybf2pa5nxDvKe1"} # should move to secrete place in the future

class Assistant(AbsApiModel):   
    def __init__(self, client:AzureOpenAI, system_prompt, temp = 0.15, domain = "SC2"):
        super().__init__()
        self.client = client
        self.thread_id = self.client.beta.threads.create(tool_resources={
            "file_search": {
                "vector_store_ids": ["vs_MMig3veTRYvKAvGkuhjzRfuv"]
            }
        }).id
        if domain not in SUPPORT_DOMAIN:
            raise NotImplementedError
        self.assistant_id = ID_MAP[domain]
        self.system_prompt = system_prompt
        self.temp = temp
        
    def send_request(self, input):
        """
        Translates input sentence with desired LLM.

        :param input: Sentence for translation.
        """

        thread_message = self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content= self.system_prompt  + "/n" + input,
            # file_ids=["file-ZWoegkS6ha4nrfie0iEchnVi", "file-bT6x3aqi4MsG9eKmIizFmzZE"]
        )

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
        )

        run = self.wait_on_run(run)

        # retrieve all messages added after our last user message

        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread_id, order="asc", after=thread_message.id
        ).data

        return messages[0].content[0].text.value.strip()

    def wait_on_run(self, run):
        while run.status == "queued" or run.status == "in_progress":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id,
            )
            # print(run.status)
            time.sleep(0.2)
        return run
    