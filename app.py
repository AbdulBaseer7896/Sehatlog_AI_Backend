from flask import Flask, request, jsonify
from utilits.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from utilits.prompt import system_prompt
from groq import Groq
from pinecone import Pinecone
import os



from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain
# Initialize Flask app
app = Flask(__name__)
app.debug = True
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "default-secret-key")

# Load environment variables
load_dotenv()

# Initialize clients
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone index configuration
index_name = "medicalbot"
index = pc.Index(index_name)

# Create vector store
docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Create retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Custom Groq LLM wrapper
class GroqLLM:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name
        
    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user", 
                "content": prompt
            }],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

# Initialize LLM
llm = GroqLLM(groq_client, "llama3-70b-8192")

# Create prompt template
prompt_template = ChatPromptTemplate.from_template(
    system_prompt + "\n\nQuestion: {input}"
)

# Create chains
document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt_template
)

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

@app.route("/")
def home():
    return "Medical Chatbot API is running"

@app.route("/chat", methods=["POST"])
def chat_handler():
    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "Invalid request format"}), 400
            
        result = rag_chain.invoke({"input": data["message"]})
        return jsonify({"response": result["answer"]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))