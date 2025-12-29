from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_docling import DoclingLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from typing import Optional, Union
from pathlib import Path

DEFAULT_LLM_MODEL = "gemma3:1b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFUALT_PERSIST_DIRECTORY = "./chroma_langchain_db"
DEFAULT_TOP_K = 3
DEFUALT_MODEL_TEMPERATURE = 0.0
DEFUALT_COLLECTION_NAME = "RAG_collection"
DEFAULT_CHUNK_SIZE = 1000
DEFUALT_CHUNK_OVERLAP = 100


class RagPipeline:

    def __init__(
        self,
        llm_model: Optional[str] = DEFAULT_LLM_MODEL,
        embedding_model: Optional[str] = DEFAULT_EMBEDDING_MODEL,
        model_temperature: Optional[float] = DEFUALT_MODEL_TEMPERATURE,
        collection_name: Optional[str] = DEFUALT_COLLECTION_NAME,
        persist_directory: Optional[str] = DEFUALT_PERSIST_DIRECTORY,
        Search_top_K: Optional[int] = DEFAULT_TOP_K,
        chunk_size: Optional[int] = DEFAULT_CHUNK_SIZE,
        chunk_overlap: Optional[int] = DEFUALT_CHUNK_OVERLAP,
    ):
        self.llm = ChatOllama(model=llm_model, temperature=model_temperature)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": Search_top_K}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.prompt = ChatPromptTemplate.from_template(
            """
        Answer the question based on the following context and chat_history:
        Chat_history:
        {chat_history}
        Context:
        {context}
        Question: {question}
        Answer:"""
        )

    def store_docs(self, file_path: Union[str, Path]):
        loader = DoclingLoader(file_path=file_path)
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        chunks = filter_complex_metadata(docs)
        self.vector_store.add_documents(chunks)
        return self.vector_store

    def get_retriever(self):
        return self.retriever

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def create_rag_chain(self):
        chain = self.prompt | self.llm
        return chain
