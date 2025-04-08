import getpass
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_openai import AzureChatOpenAI


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
    MessagesPlaceholder("context"), 
    ("user", "{question}")
])


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def get_input(state: State):
    question = input("Ask me anything: ")
    return {'question': question, 'context': [], 'answer': ""}

def generate(state: State):
    print("--Generating Response--")
    messages = prompt.invoke({'question': state['question'], 'context': state['context']})
    response = llm.invoke(messages)
    return {'answer': response}

        
builder = StateGraph(State)

builder.add_node("get_input", get_input)
builder.add_node("generate", generate)

builder.add_edge(START, "get_input")
builder.add_edge("get_input", "generate")
builder.add_edge("generate", "get_input")

graph = builder.compile()

for event in graph.stream({'question': "", 'context': [], 'answer': ""}, stream_mode='values'):
    if event.get('answer',''):
        event['answer'].pretty_print()