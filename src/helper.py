from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain_core.documents import Document


# -------------------------------
# Load PDF files
# -------------------------------
def load_pdf_files(data_path: str) -> List[Document]:
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# -------------------------------
# Clean metadata
# -------------------------------
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        minimal_docs.append(
            Document(page_content=doc.page_content, metadata={"source": src})
        )

    return minimal_docs


# -------------------------------
# Split text into chunks
# -------------------------------
def text_split(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(docs)


# -------------------------------
# Load embeddings
# -------------------------------
"""def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )
"""


def download_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")
