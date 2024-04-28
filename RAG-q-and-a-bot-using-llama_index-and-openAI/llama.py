import nest_asyncio
import streamlit as st
nest_asyncio.apply()

from llama_index.evaluation import generate_question_context_pairs
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.evaluation import generate_question_context_pairs
from llama_index.evaluation import RetrieverEvaluator
from llama_index.llms import OpenAI

import os
import pandas as pd
os.environ["OPENAI_API_KEY"]="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"

documents = SimpleDirectoryReader("./docs").load_data()
st.header("Chat with the your docs 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about documents u uploaded!"}
    ]
# Define an LLM
llm = OpenAI(model="gpt-4")

# Build index with a chunk_size of 512
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)
vector_index = VectorStoreIndex(nodes)
#query_engine = vector_index.as_query_engine()
#response_vector = query_engine.query("for the ae gpt paper show me the statistics along with the numbers of ae-gpt prediction errors on different entity type ?")
#print(response_vector.response)
chat_engine = vector_index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])
# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history        
