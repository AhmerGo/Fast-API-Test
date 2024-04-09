from threading import Thread
from queue import Queue, Empty
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import GPT4All
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import time

model_path = "./Models/wizardlm-13b-v1.2.Q4_0.gguf"
index_path = "./HSU_index"

# Initialize a global queue for communication between threads
q = Queue()
job_done = object()


def initialize_embeddings() -> LlamaCppEmbeddings:
    return LlamaCppEmbeddings(model_path=model_path)


class StreamingCallback:
    """Callback handler for streaming responses to a queue."""

    def __init__(self, q):
        self.q = q

    def put(self, response):
        self.q.put(response)

    def end(self):
        self.q.put(job_done)


def main():
    device = "gpu"
    llm = GPT4All(model=model_path, device=device)
    embeddings = initialize_embeddings()
    index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    memory = ConversationBufferMemory(input_key="question")  # Specify the input key
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=index.as_retriever(),
        chain_type="stuff",
        verbose=True,
        max_tokens_limit=10000,
        memory=memory
    )
    chat_history = []
    print("Welcome to the chatbot! Type 'exit' to stop.")

    callback = StreamingCallback(q)

    def get_response(question, chat_history):
        result = qa({"question": question, "chat_history": chat_history})
        callback.put(result['answer'])
        callback.end()

    while True:
        question = input("Please enter your question: ")
        if question.lower() == 'exit':
            break

        # Start the task in a new thread
        Thread(target=get_response, args=(question, chat_history)).start()

        # Wait for and collect the response
        while True:
            try:
                next_response = q.get(True, timeout=1)
                if next_response is job_done:
                    break
                print("Answer:", next_response)
                chat_history.append((question, next_response))
            except Empty:
                continue


if __name__ == "__main__":
    main()
