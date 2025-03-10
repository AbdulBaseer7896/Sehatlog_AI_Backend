# from flask import Flask , render_template , request
# from sqlalchemy.exc import OperationalError
# from utilits.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate
# from utilits.prompt import *
# from utilits.Groq import *
# from groq import Groq


# import os
# app = Flask(__name__)

# app.debug = True
# app.secret_key = "your_secret_key"


# load_dotenv()

# PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# groq_client = Groq(api_key=GROQ_API_KEY)

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# embeddings = download_hugging_face_embeddings()


# index_name = "medicalbot"

# # Embed each chunk and upsert the embeddings into your Pinecone index.
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# llm = GroqLLM(groq_client, "llama-3-70b-8192")

# # In app.py replace:
# document_chain = create_stuff_documents_chain(llm, system_prompt)

# # With proper prompt template setup:


# prompt_template = ChatPromptTemplate.from_template(
#     system_prompt + "\n\nQuestion: {input}"
# )
# document_chain = create_stuff_documents_chain(llm, prompt_template)


# # question_answer_chain = create_stuff_documents_chain(llm, prompt)
# # rag_chain = create_retrieval_chain(retriever, question_answer_chain)




# @app.route("/")
# def hello_world():
#     return 'its its workinng fines'

# from controller import *

# # @app.errorhandler(OperationalError)
# # def handle_operational_error(error):
# #     app.logger.error(f"OperationalError: {str(error)}")
# #     return render_template('error.html', message="An error occurred while processing your request."), 500

# if __name__ == '__main__':
#     app.run(debug=True) 













# from flask import Flask, request
# from utilits.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from utilits.prompt import system_prompt
# from utilits.Groq import GroqLLM
# from groq import Groq
# import os

# app = Flask(__name__)
# load_dotenv()

# # Initialize clients
# groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
# embeddings = download_hugging_face_embeddings()

# # Pinecone setup
# docsearch = PineconeVectorStore.from_existing_index(
#     index_name="medicalbot",
#     embedding=embeddings
# )
# retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# # LLM setup
# llm = GroqLLM(client=groq_client, model_name="llama3-70b-8192")

# # Prompt template
# prompt_template = ChatPromptTemplate.from_template(
#     system_prompt + "\n\nQuestion: {input}"
# )

# # Chain setup
# document_chain = StuffDocumentsChain(llm=llm, prompt=prompt_template)

# rag_chain = create_retrieval_chain(retriever, document_chain)

# @app.route("/")
# def hello_world():
#     return 'API is working'
# from controller import *


# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
from sqlalchemy.exc import OperationalError
from utilits.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
# Remove OpenAI if you're using Groq now
# from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone
from utilits.prompt import system_prompt
from utilits.Groq import GroqLLM

import os
app = Flask(__name__)
app.debug = True
app.secret_key = "your_secret_key"

load_dotenv()
# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
groq_client_value = os.environ.get('GROQ_API_KEY')

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["groq_client"] = groq_client_value

embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"
index = pc.Index(index_name)
docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"  # Match your index's text field name
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# Instantiate your Groq LLM.
llm = GroqLLM(groq_client_value, "llama-3.3-70b-versatile")

# Create a prompt template.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])


# Use the helper function to create the chain.
document_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, document_chain)

@app.route("/")
def hello_world():
    return 'its its workinng fines'

from controller import *

if __name__ == '__main__':
    app.run(debug=True)
