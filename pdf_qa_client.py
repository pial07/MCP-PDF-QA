import streamlit as st
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI 
from decouple import config

# Load API key
OPENAI_API_KEY = config("OPENAI_API_KEY")

# Load embeddings and vector store
persist_directory = "H:/py4e/Doc Analysis/output/vectorstore"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY) # type: ignore
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Setup QA chain
llm = ChatOpenAI(temperature=0, model="gpt-4", api_key=OPENAI_API_KEY) # type: ignore
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), return_source_documents=True)

# Streamlit UI
st.set_page_config(page_title="PDF QA", page_icon="📄")
st.title("📄 Ask Questions from PDF")

query = st.text_input("Ask a question:")

if st.button("Submit") and query:
    result = qa_chain(query)
    st.markdown("### ✅ Answer")
    st.write(result["result"])

    st.markdown("### 📂 Sources")
    for doc in result["source_documents"]:
        st.markdown(f"- `{doc.metadata.get('source', 'Unknown')}`")
