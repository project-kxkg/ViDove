import os
from logging import Logger

from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    StreamingAgentChatResponse,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from .abs_api_RAG import AbsApiRAG

PERSIST_DIR = "./storage/basic_rag"
DATA_DIR = "domain_dict/SC2"

# text_qa_template

SYSTEM_PROMPT = PromptTemplate(
    "You are a professional translator. your job is to translate texts in domain of {domain} from {source_language} to {target_language} \n"
    "you will be provided with a segment in source language parsed by line, where your translation text should keep the original meaning and the number of lines. \n"
    "You should only output the translated text line by line without any other notation. \n"
    "--------------- \n"
    "if you detect the the word is in the following context, please use it as a reference.\n"
    "{context_str}"
    "--------------- \n"
    "Please translate the following text from {source_language} to {target_language} \n"
    "{query_str}"
    "Your translation:"
)


class BasicRAG(AbsApiRAG):
    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        logger: Logger,
        domain="starcraft2",
        llm_name: str = "gpt-4o-mini",
        embedding_name: str = "text-embedding-3-small",
    ) -> None:
        super().__init__()
        self.embeddings = OpenAIEmbedding(model=embedding_name)
        self.llm = OpenAI(model=llm_name)
        self.system_prompt = SYSTEM_PROMPT.partial_format(
            source_lang=source_lang, target_lang=target_lang, domain=domain
        )
        self.logger = logger
        self.loaded = False

    def load_model(self):
        Settings.embed_model = self.embeddings
        Settings.llm = self.llm
        self.logger.info(
            f"Loading the model, set {Settings.embed_model} as embedding model and {Settings.llm} as the generator"
        )
        if not os.path.exists(PERSIST_DIR):
            self.logger.info("Loading the RAG from the scratch")
            documents = SimpleDirectoryReader(DATA_DIR).load_data()
            index = VectorStoreIndex.from_documents(
                documents,
                transformations=[SentenceSplitter(chunk_size=40, chunk_overlap=5)],
            )
            index.storage_context.persist(PERSIST_DIR)
        else:
            self.logger.info("Loading the RAG from the storage")
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)

        self.query_engine = index.as_query_engine()
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": self.system_prompt}
        )
        self.loaded = True
        self.logger.info("Model loaded")

    def send_request(self, input) -> AgentChatResponse | StreamingAgentChatResponse:
        if not self.loaded:
            self.load_model()
        return self.query_engine.query(input)
