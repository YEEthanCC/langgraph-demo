import getpass
import os
import operator
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, List, Annotated
# from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime
import json
import re
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("AZURE_OPENAI_ENDPOINT")
_set_env("AZURE_OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")


# embedding model
embed_model = AzureOpenAIEmbeddings(model="text-embedding-3-small")


llm = AzureChatOpenAI(
    api_version="2023-07-01-preview",
    azure_deployment="gpt-4o",
)

prompt = ChatPromptTemplate([
    ("system", "You are an intelligent, friendly, and helpful AI chatbot."), 
    MessagesPlaceholder("history"), 
    ("user", "{msg}")
])

save_prompt = ChatPromptTemplate([
        ("system", 
            "You are an intelligent assistant that extracts structured knowledge from a conversation between a user and an LLM. "
            "Your task is to extract and organize:\n\n"
            "- All facts as key-value pairs (where the key is a concise label, and the value is the specific factual detail).\n"
            "- All concepts as key-value pairs (where the key is the concept name or category, and the value is a brief explanation or definition from the conversation).\n\n"
            "Do not invent or infer information. Only extract what is explicitly stated in the conversation."
            "The content should be exclusive to knowledge that you do not know previously and learned from the past conversation"
        ),
        ("user",  
            "Below is a list of messages between a user and an LLM. "
            "Extract all explicitly mentioned facts and concepts. "
            "Return only the structured JSON.\n\n"
            "### Conversation:\n{conversation_history}\n\n"
        )
])

class State(TypedDict):
    messages: Annotated[list, operator.add]
    recall_memory: str

def get_input(state: State):
    print("--Get User Input--")
    msg = input("Input: ")
    return {"messages": [HumanMessage(msg)]}

def check_input(state: State):
    print("--Check Input--")
    if state['messages'][-1].content == "\q":
        return "save_memory"
    else:
        return "recall"

def recall(state: State):
    print("--Recall Memory--")
    if len(os.listdir("memory/faiss/")) != 0:
        vector_store = FAISS.load_local("memory/faiss/", embeddings=embed_model, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={'k': 1})
        retrieved_documents = retriever.invoke(state['messages'][-1].content)
        return {"recall_memory": str(retrieved_documents[0].page_content)}


def generate(state: State):
    print("--Generating Response--")
    history = state['messages'][:-1]
    history.append(state['recall_memory'])
    message = prompt.invoke({'msg': state['messages'][-1].content, 'history': history})
    response = llm.invoke(message).content
    return {'messages': [AIMessage(content=response)]}

def save_memory(state: State):
    print("Saving conversation...")
    message = save_prompt.invoke({'conversation_history': state['messages']})
    response = llm.invoke(message).content
    response = re.sub(r"^```json|```$", "", response.strip()).strip()
    data = json.loads(response)
    with open(f"memory/json/{datetime.now()}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    documents = [Document(page_content=response)]
    vectorstore = FAISS.from_documents(documents, embed_model)
    vectorstore.save_local("memory/faiss/")

        
builder = StateGraph(State)

builder.add_node("get_input", get_input)
builder.add_node("recall", recall)
builder.add_node("generate", generate)
builder.add_node("save_memory", save_memory)

builder.add_edge(START, "get_input")
builder.add_conditional_edges(source="get_input", path=check_input)
builder.add_edge("recall", "generate")
builder.add_edge("generate", "get_input")
builder.add_edge("save_memory", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}

for event in graph.stream({"messages": [], "recall_memory": ""}, config, stream_mode='values'):
    if len(event["messages"]) != 0:
        event["messages"][-1].pretty_print()