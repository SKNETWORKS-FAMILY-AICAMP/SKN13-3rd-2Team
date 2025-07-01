from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb.config
import os

persist_dir = "chroma_card_db"
os.makedirs(persist_dir, exist_ok=True)
client_settings = chromadb.config.Settings(
    anonymized_telemetry=False,
    persist_directory=persist_dir
)
embeddings = HuggingFaceEmbeddings(model_name="BM-K/KoSimCSE-roberta-multitask")
from langchain_core.documents import Document
docs = [Document(page_content="테스트 문서", metadata={"a": 1})]
db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory=persist_dir,
    client_settings=client_settings
)
db.persist()