import getpass
import os
import operator
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("AZURE_OPENAI_ENDPOINT")
_set_env("AZURE_OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")

db = None

# embedding model
embed_model = AzureOpenAIEmbeddings(model="text-embedding-3-small")

llm = AzureChatOpenAI(
    api_version="2023-07-01-preview",
    azure_deployment="gpt-4o",
)

prompt = ChatPromptTemplate([
    ("system", "You are an intelligent, friendly, and helpful AI assistant."), 
    MessagesPlaceholder("history"), 
    ("user", "{msg}")
])


class State(TypedDict):
    messages: Annotated[list, operator.add]

def get_input(state: State):
    msg = input("Ask me anything: ")
    return {"messages": [HumanMessage(msg)]}

def generate(state: State):
    print("--Generating Response--")
    message = prompt.invoke({'msg': state['messages'][-1].content, 'history': state['messages'][:-1]})
    response = llm.invoke(message).content
    return {'messages': [AIMessage(content=response)]}

        
builder = StateGraph(State)

builder.add_node("get_input", get_input)
builder.add_node("generate", generate)

builder.add_edge(START, "get_input")
builder.add_edge("get_input", "generate")
builder.add_edge("generate", "get_input")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

for event in graph.stream({"messages": []}, config, stream_mode='values'):
    if len(event["messages"]) != 0:
        event["messages"][-1].pretty_print()