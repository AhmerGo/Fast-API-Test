import logging
import os
import torch
from ts.torch_handler.base_handler import BaseHandler
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import GPT4All
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import pymongo
from bson import json_util

from database_helper import Connection

class HSUHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.embeddings = None
        self.llm = None
        self.index = None
    
    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = "nvidia" if torch.cuda.is_available() else "cpu"
        self.model_path = os.path.join(model_dir, self.manifest["model"]["serializedFile"])
        self.index_dir = model_dir
        self.embeddings = LlamaCppEmbeddings(model_path=self.model_path)
        logging.info(f"!!!!! DEVICE INFO : {torch.cuda.get_device_name(0)}")
        logging.info(f"!!!!! IS CUDA AVAILABLE ? : {torch.cuda.is_available()}")
        logging.info(f"!!!!!!CUDA VERSION : {torch.version.cuda}")
        self.llm = GPT4All(model=self.model_path, device=self.device)
        self.index = FAISS.load_local(self.index_dir, self.embeddings, allow_dangerous_deserialization=True)
    
    def preprocess(self, data):
        input_data = data[0].get("data")
        if input_data is None:
            input_data = data[0].get("body")
        
        return input_data

    def inference(self, input_data):
        question = input_data["question"]
        user_id = input_data["user_id"]
        
        db_connection = Connection()
        db_connection.connect("admin", "Stevencantremember", "admin")
        chat_history = db_connection.get_chat_history(user_id)
        
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
        for user_input, bot_response in chat_history:
            memory.save_context({"question": user_input}, {"answer": bot_response})
        
        condense_question_prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template="Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.index.as_retriever(),
            chain_type="stuff",
            condense_question_prompt=condense_question_prompt,
            verbose=True,
            memory=memory
        )
        
        result = conversation_chain({"question": question})["answer"]
        
        memory.save_context({"question": question}, {"answer": result})
        updated_chat_history = [(msg.content, memory.chat_memory.messages[i+1].content) for i, msg in enumerate(memory.chat_memory.messages) if i % 2 == 0]
        db_connection.update_chat_history(user_id, updated_chat_history)
        
        db_connection.close()

        output = {"answer": result}

        
        return [output]

    
    def postprocess(self, inference_output):
        return inference_output
