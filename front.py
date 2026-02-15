# streamlit_app.py

import streamlit as st
from app import chatbot
from langchain_core.messages import HumanMessage

st.title("TalentScout hiring assistant chatbot")

CONFIG = {"configurable": {"thread_id": "thread-1"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Show previous messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Type here")

if user_input:

    # Show user message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user",avatar="🙎‍♀️"):
        st.markdown(user_input)

    # Get response from LangGraph
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG
    )

    ai_message = response["messages"][-1].content

    # Store assistant response
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    with st.chat_message("assistant",avatar="🧑‍🏫"):
        st.markdown(ai_message)
