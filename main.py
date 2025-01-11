import os
import json
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vectorize_documents import embeddings

# Load configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(working_dir, "config.json")
with open(config_path, 'r') as config_file:
    config_data = json.load(config_file)
GROQ_API_KEY = config_data.get("GROQ_API_KEY")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def display_message(role, content):
    with st.chat_message(role):
        st.write(content)

def get_assistant_response(user_input):
    try:
        response = st.session_state.conversational_chain({"question": user_input})
        return response["answer"]
    except Exception as e:
        st.error(f"Error: {e}")
        return "Sorry, I couldn't process your request."

# Display chat history
for message in st.session_state.chat_history:
    display_message(message["role"], message["content"])

# Get user input
user_input = st.chat_input("Ask AI...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    display_message("user", user_input)

    assistant_response = get_assistant_response(user_input)
    display_message("assistant", assistant_response)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})