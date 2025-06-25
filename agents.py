from langchain_openai import ChatOpenAI
from decouple import config
from smolagents import ToolCallingAgent, ToolCollection
from mcp import StdioServerParameters

# Load your OpenAI key
OPENAI_API_KEY = config("OPENAI_API_KEY", default=None)

print("Using OpenAI API Key:", OPENAI_API_KEY[:10], "...")

# ✅ Use ChatOpenAI directly
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, api_key=OPENAI_API_KEY)

# ✅ Start PDF QA server and pass key as env variable
server_parameters = StdioServerParameters(
    command="uv",
    args=["run", "pdfqaserver.py"],
    env={"OPENAI_API_KEY": str(OPENAI_API_KEY)}
)

# Use MCP to load tools and run the agent
with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = ToolCallingAgent(tools=[*tool_collection.tools], model=model)
    agent.run("Summarize the latest engineering safety protocol from the documents.")
