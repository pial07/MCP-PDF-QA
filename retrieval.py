from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI  # ✅ Chat model import

def build_rag_qa_chain(vectordb, api_key):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=api_key)  # ✅ Correct class and arg

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    return chain
