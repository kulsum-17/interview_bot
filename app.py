# app.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]

    # Fixed role (system message added only once)
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_message = SystemMessage(
            content="you are a intelligent hiring assistant chatbot for TalentScout a company that help in technical placements gather the candidate information such as name, age ,college name,degree,tech stack,experience and then user 5 technical relevent question on the basis of these information be a little concise ask a question on the basis of the experience don't ask difficult question from a fresher and don't ask a easy question from a experienced person ,ask the question one by one not all together and on the basis ."
        )
        messages = [system_message] + messages

    response = llm.invoke(messages)

    return {"messages": [response]}

# Memory
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
