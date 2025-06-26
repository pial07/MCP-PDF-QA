# Import required libs
import os
import pickle
import concurrent.futures
import traceback
from decouple import config
from mcp.server.fastmcp import FastMCP
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from retrieval import build_rag_qa_chain


# Initialize MCP server
mcp = FastMCP("pdfqaserver")

# Load OpenAI key from environment
OPENAI_API_KEY = config("OPENAI_API_KEY", default=None, cast=str)
print("Using OpenAI API Key:", OPENAI_API_KEY)
# Define path to saved Chroma vectorstore
persist_directory = r"H:\py4e\Doc Analysis\output\vectorstore"

# Load vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)  # type: ignore
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)




# Create tool
@mcp.tool()
def pdf_qa_tool(query: str) -> str:
    """
    Ask any question from the PDF documents stored in the vector database.
    Now supports conversational context with memory.
    """

    def rag_query():
        try:
            rag_chain = build_rag_qa_chain(vectordb=vectordb, api_key=OPENAI_API_KEY)
            result = rag_chain({
                "question": query,
                "chat_history": []
            })
            sources = "\n".join([
                f"- {doc.metadata.get('source_file', 'Unknown source')}"
                for doc in result.get("source_documents", [])
            ])
            return f"📘 Answer:\n{result['answer']}\n\n📂 Sources:\n{sources}"
        except Exception as e:
            return f"❌ Error:\n{traceback.format_exc()}"

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(rag_query)
            return future.result(timeout=60)  # ⏱ Set max timeout (can be increased if needed)
    except concurrent.futures.TimeoutError:
        return "⏱️ GPT-4 took too long. Try again or simplify your query."

# Start server
if __name__ == "__main__":
    mcp.run(transport="stdio")
