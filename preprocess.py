from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from config import *

EMBED_CONTEXT_MODEL = embedding_context_model

def load_document(file):
    loader = PyPDFLoader(file)
    documents = loader.load()
    return documents

def split_chunk(document, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.split_documents(document)
    return chunked_docs

def get_chunk_id(chunked_docs):
    last_page_id = f"{chunked_docs[0].metadata.get('source')}:{chunked_docs[0].metadata.get('page')}"
    current_chunk_index = 0

    for chunk in chunked_docs:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
            last_page_id = current_page_id
        else:
            current_chunk_index = 0
            last_page_id = current_page_id
        chunk_id = f"{source}:{page}:{current_chunk_index}"
        chunk.metadata["id"] = chunk_id
    return chunked_docs

def embed_to_vectordb(chunked_docs):
    model = HuggingFaceEmbeddings(model_name=EMBED_CONTEXT_MODEL)
    vectorstore = Chroma(
        embedding_function=model,
        persist_directory="FAQdb"  
    )
    vectorstore.add_documents(documents=chunked_docs)
    vectorstore.persist()

    return vectorstore