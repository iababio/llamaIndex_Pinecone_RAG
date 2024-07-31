import os
from getpass import getpass
import gradio as gr
import random
import time

pinecone_api_key = os.getenv("PINECONE_API_KEY") or getpass("Enter your Pinecone API Key: ")
openai_api_key = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API Key: ")

from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings import OpenAIEmbedding
from llama_index.ingestion import IngestionPipeline

# This will be the model we use both for Node parsing and for vectorization
embed_model = OpenAIEmbedding(api_key=openai_api_key)

# Define the initial pipeline
pipeline = IngestionPipeline(
    transformations=[
        SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        ),
        embed_model,
    ],
)

from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec

from llama_index.vector_stores import PineconeVectorStore

# Initialize connection to Pinecone
pc = PineconeGRPC(api_key=pinecone_api_key)
index_name = "anualreport"

# Initialize your index
pinecone_index = pc.Index(index_name)

# Initialize VectorStore
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

pinecone_index.describe_index_stats()

from llama_index import VectorStoreIndex
from llama_index.retrievers import VectorIndexRetriever

# Set the OpenAI API key if not already set
if not os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = openai_api_key

# Instantiate VectorStoreIndex object from our vector_store object
vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Grab 5 search results
retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

from llama_index.query_engine import RetrieverQueryEngine

# Pass in your retriever from above, which is configured to return the top 5 results
query_engine = RetrieverQueryEngine(retriever=retriever)

def query_anual_report(query):
    response = query_engine.query(query)
    return response.response

# Define the chat functions
def user(user_message, history):
    return "", history + [[user_message, None]]

def bot(history):
    bot_message = query_anual_report(history[-1][0])
    history[-1][1] = ""
    for character in bot_message:
        history[-1][1] += character
        time.sleep(0.01)  # Reduced sleep time to make response appear faster
        yield history

# Define Gradio Blocks interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
