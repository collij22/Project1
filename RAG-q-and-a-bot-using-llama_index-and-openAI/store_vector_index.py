import nest_asyncio
import streamlit as st
nest_asyncio.apply()

from llama_index.evaluation import generate_question_context_pairs
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.evaluation import generate_question_context_pairs
from llama_index.evaluation import RetrieverEvaluator
from llama_index.llms import OpenAI
import pickle

import os
import pandas as pd
os.environ["OPENAI_API_KEY"]="api key here"

documents = SimpleDirectoryReader("./docs").load_data()

llm = OpenAI(model="gpt-3.5-turbo")
documents = SimpleDirectoryReader("./docs").load_data()
# Build index with a chunk_size of 512
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)
vector_index = VectorStoreIndex(nodes)
file_path = 'index.pkl'
with open(file_path, 'wb') as file:
    pickle.dump(vector_index, file)
