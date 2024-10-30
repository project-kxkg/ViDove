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
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
import openai

# Setup the tavily API before using online search
# Set up Tavily tool
from tavily import TavilyClient

from .abs_api_RAG import AbsApiRAG

PERSIST_DIR = "./storage/basic_rag"
DATA_DIR = "domain_dict/SC2"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# text_qa_template

SYSTEM_PROMPT = (
    "You are a professional translator. your job is to translate texts in domain of {domain} from {source_language} to {target_language} \n"
    "you will be provided with a segment in source language parsed by line, where your translation text should keep the original meaning and the number of lines. \n"
    "You should only output the translated text line by line without any other notation. \n"
    "--------------- \n"
    "if you detect the the word is in the following context, please use it as a reference.\n"
    "{context_str}"
    "--------------- \n"
    "Here are some supporting documents that might help you translate the text, refer to them if necessary: "
    "{supporting_documents}"
    "Please translate the following text from {source_language} to {target_language} \n"
    "{query_str}"
    "Your translation:"
)


class TavilyRAG(AbsApiRAG):
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
        self.llm_client = openai.OpenAI()
        self.llm_name = llm_name
        self.src_lang = source_lang
        self.tgt_lang = target_lang
        self.domain = domain
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.logger = logger
        self.loaded = False

    def load_model(self):
        # Settings.embed_model = self.embeddings
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

        self.retriever = VectorIndexRetriever(index, similarity_top_k=10, embed_model=self.embeddings)

        self.loaded = True
        self.logger.info("Model loaded")

    def seach_tavily(self, query: str, max_results: int = 3):
        responses = self.tavily_client.search(
            query, max_results=max_results, search_depth="advanced", include_domains=[self.domain]
        )
        return [Document(text=result["content"], extra_info={"url": result["url"]}) for result in responses["results"]]

    def send_request(self, input) -> AgentChatResponse | StreamingAgentChatResponse:
        if not self.loaded:
            self.load_model()

        context_documents = self.retriever.retrieve(input)
        context_str = "\n".join([doc.text for doc in context_documents])

        web_supporting_documents = self.seach_tavily(input)

        prompt = SYSTEM_PROMPT.format(
            source_language=self.src_lang,
            target_language=self.tgt_lang,
            domain=self.domain,
            context_str=context_str,
            supporting_documents=web_supporting_documents,
            query_str=input,
        )

        response = self.llm_client.chat.completions.create(
            model=self.llm_name,
            messages = [{
                "role": "user",
                "content": prompt
            }]
        ).choices[0].message.content

        return response, web_supporting_documents



        
        
