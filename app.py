from flask import Flask, render_template, jsonify, request
from utilits.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from utilits.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


# llm = OpenAI(temperature=0.4, max_tokens=500)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return "its wokring"


@app.route("/get", methods=["GET"])
def chat():
    msg = request.args.get("msg")  # Use 'args.get()' for GET request
    if msg is None:
        return jsonify({"error": "No message parameter provided"}), 400  # Return a 400 error if msg is not provided

    print("Received message:", msg)

    # Replace this with actual logic for response generation
    # For example, you could process 'msg' here with a retriever or model
    print("Response from retriever:", retriever)

    return jsonify({"response": str(retriever)})  # Just for testing; replace with actual response




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
