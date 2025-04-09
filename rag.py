import getpass
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from typing import TypedDict, List
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_openai import AzureChatOpenAI
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

if len(os.listdir('knowledge-base')) == 0:
    loader = PyPDFLoader(file_path="data/新手包_瑞光總部攻略_20240311.pdf")
    docs = []
    docs = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=30,
        add_start_index=True)

    all_splits = text_splitter.split_documents(docs)

    print(f"Split the pdf into {len(all_splits)} sub-documents.")

    # vector storage
    db = Chroma.from_documents(
        documents=all_splits, 
        embedding=embed_model,
        persist_directory="./knowledge-base"
    )
else:
    db = Chroma(persist_directory="./knowledge-base", embedding_function=embed_model)


llm = AzureChatOpenAI(
    api_version="2023-07-01-preview",
    azure_deployment="gpt-4o",
)

prompt = hub.pull('rlm/rag-prompt')

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def get_input(state: State):
    question = input("Ask me anything: ")
    return {'question': question, 'context': [], 'answer': ""}

def retrieve(state: State):
    print("--Retrieving Information--")
    question = state['question']
    retrieved_docs = db.as_retriever().invoke(question)[0].page_content
    return {'context': retrieved_docs}

def check_relevance(state: State):
    print("--Check Question Relevance--")
    response = llm.invoke(f"Check if the question: {state['question']} is relevant to the context: {state['context']}, if relevant, answer YES, else, answer NO")
    if "YES" in response.content:
        return "generate"
    else:
        return "search"

def generate(state: State):
    print("--Generating Response--")
    messages = prompt.invoke({'question': state['question'], 'context': state['context']})
    response = llm.invoke(messages)
    return {'answer': response}

from langchain_tavily import TavilySearch

def search(state: State):
    print("--Search For Answer--")
    tool = TavilySearch(
        max_results=5,
        topic="general",
    )
    return {'context': tool.invoke({"query": state['question']})}
        
builder = StateGraph(State)

builder.add_node("get_input", get_input)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("search", search)

builder.add_edge(START, "get_input")
builder.add_edge("get_input", "retrieve")
builder.add_conditional_edges(source="retrieve", path=check_relevance)
builder.add_edge("generate", "get_input")
builder.add_edge("search", "generate")

config = {"configurable": {"thread_id": "1"}}
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

for event in graph.stream({'question': '咖啡吧在哪裡?'}, config, stream_mode='values'):
    if event.get('answer',''):
        print("history:")
        print(graph.get_state_history(config))
        print("answer: ")
        event['answer'].pretty_print()