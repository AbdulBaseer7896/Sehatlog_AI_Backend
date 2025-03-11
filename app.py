from flask import Flask, render_template, jsonify, request
from utilits.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from utilits.prompt import *
from utilits.Groq import GroqLLM
import os
from flask_cors import CORS  # <-- Add this missing import

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

groq_client_value = os.environ.get('GROQ_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["groq_client"] = groq_client_value
embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
print(retriever)

llm = GroqLLM(groq_client_value, "llama-3.3-70b-versatile")

# Create a prompt template.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt_template )
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return "its wokring"


from controller import *


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

