import asyncio
import logging
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from ctransformers import AutoModelForCausalLM, AutoTokenizer
import torch


class HSU:
    def __init__(self, model_path, index_path, device):
        logging.info(f"Initializing HSU with model_path: {
                     model_path}, index_path: {index_path}, device: {device}")
        try:
            self.model_path = model_path
            self.index_path = index_path
            self.device = device

            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2")
            self.index = FAISS.load_local(
                index_path, self.embeddings, allow_dangerous_deserialization=True)
            logging.info(f"FAISS index loaded. Index dimensions: {
                         self.index.index.d}")

            self.llm = CTransformers(model=self.model_path, max_length=2048)

            prompt_template = """Assistant is a large language model trained by Anthropic.

            Context: {context}

            H: {question}

            A:"""

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=prompt_template
            )

            document_prompt = PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            )

            qa_chain = load_qa_chain(
                self.llm, chain_type="stuff", prompt=prompt)
            summarize_chain = load_summarize_chain(
                self.llm, chain_type="stuff", document_prompt=document_prompt)

            self.qa = RetrievalQA(
                combine_documents_chain=summarize_chain,
                retriever=self.index.as_retriever(),
                return_source_documents=True,
                verbose=True
            )

            logging.info("HSU initialization completed successfully")
        except Exception as e:
            logging.exception(f"Error occurred during HSU initialization: {e}")
            raise

    async def rag(self, question, index_path):
        logging.info(f"Received question: {question}")
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # Create a mock Document object or adjust as necessary to match the expected structure
            mock_document = Document(page_content=question)

            # Adjust the call to use a list of mock Document objects
            index = FAISS.from_documents([mock_document], embeddings)
            index.save_local(index_path)

            query_vector = self.embeddings.embed_documents(question)
            logging.info(f"Query vector length: {len(query_vector)}")

            result = await asyncio.to_thread(self.qa.invoke, input={"query": question})
            return result
        except Exception as e:
            logging.exception(f"Error occurred during RAG processing: {e}")
            raise
