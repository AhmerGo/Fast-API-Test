# SCRIPT INFO: This script allows you to create an embeddings and vector store from a single file.
import multiprocessing
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores.faiss import FAISS


# Setup
llama_path = './Models/mistral-7b-openorca.Q4_0.gguf'
callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])
loader = TextLoader('./Dataset/HSU_website_data.txt')
embeddings = LlamaCppEmbeddings(
    model_path=llama_path,
    n_threads=max(multiprocessing.cpu_count() - 1, 1),
    n_ctx=int(32768)  # Need to use the same context size as the model
    # n_ctx=int(os.environ['MODEL_CONTEXT_SIZE'])  # Need to use the same context size as the model
)


# Split text
def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks


# Indexing
def create_index(chunks):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return search_index


def main():
    # Create Index
    docs = loader.load()
    chunks = split_chunks(docs)
    index = create_index(chunks)

    # Save Index (use this to save the index for later use)
    index.save_local("HSU_index")


if __name__ == "__main__":
    main()
