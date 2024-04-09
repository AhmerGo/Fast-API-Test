from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import GPT4All
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

model_path = "./Models/wizardlm-13b-v1.2.Q4_0.gguf"
index_path = "./HSU_index"


def initialize_embeddings() -> LlamaCppEmbeddings:
    return LlamaCppEmbeddings(model_path=model_path)


def main():
    device = "gpu"
    llm = GPT4All(model=model_path, device=device)
    embeddings = initialize_embeddings()
    index = FAISS.load_local(index_path, embeddings,
                             allow_dangerous_deserialization=True)
    memory = ConversationBufferMemory(
        input_key="question")  # Specify the input key
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
    while True:
        question = input("Please enter your question: ")
        if question.lower() == 'exit':
            break
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print("Answer:", result['answer'])


if __name__ == "__main__":
    main()
