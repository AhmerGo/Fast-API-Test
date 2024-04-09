import logging
import torch
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import GPT4All
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from database_helper import Connection

class HSU:
    model_path = "../LLM/Models/wizardlm-13b-v1.2.Q4_0.gguf"
    index_path = "../LLM/HSU_index"
    embeddings = None
    llm = None

    @staticmethod
    def initialize():
        HSU.embeddings = LlamaCppEmbeddings(model_path=HSU.model_path)
        logging.info(f"!!!!! DEVICE INFO : {torch.cuda.get_device_name(0)}")
        logging.info(f"!!!!! IS CUDA AVAILABLE ? : {torch.cuda.is_available()}")
        logging.info(f"!!!!!!CUDA VERSION : {torch.version.cuda}")
        device = "nvidia" if torch.cuda.is_available() else "cpu"
        HSU.llm = GPT4All(model=HSU.model_path, device=device)

    @staticmethod
    def rag(question, user_id):
        try:
            if HSU.embeddings is None or HSU.llm is None:
                HSU.initialize()

            db_connection = Connection()
            db_connection.connect("admin", "Stevencantremember", "admin")

            # Retrieve the chat history for the user from the database
            chat_history = db_connection.get_chat_history(user_id)

            # Create a ConversationBufferMemory instance with the retrieved chat history
            memory = ConversationBufferMemory(memory_key="history", input_key="input")
            for user_input, bot_response in chat_history:
                memory.save_context({"input": user_input}, {"output": bot_response})

            index = FAISS.load_local(
                HSU.index_path,
                HSU.embeddings,
                allow_dangerous_deserialization=True
            )

            prompt_template = """You are a chatbot having a conversation with a human.

            {history}
            Human: {input}
            Chatbot:"""

            prompt = PromptTemplate(
                input_variables=["history", "input"], 
                template=prompt_template
            )

            conversation_chain = ConversationChain(
                llm=HSU.llm,
                prompt=prompt,
                verbose=True,
                memory=memory
            )

            result = conversation_chain.predict(input=question)

            # Update the chat history in the database
            memory.save_context({"input": question}, {"output": result})
            updated_chat_history = [(msg.content, memory.chat_memory.messages[i+1].content)
                                    for i, msg in enumerate(memory.chat_memory.messages) if i % 2 == 0]
            db_connection.update_chat_history(user_id, updated_chat_history)

            db_connection.close()

            return {"answer": result}

        except Exception as e:
            logging.error("An error occurred during the question-answering process.", exc_info=True)