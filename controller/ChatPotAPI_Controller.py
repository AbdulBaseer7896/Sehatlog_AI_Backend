from app import app
from functools import wraps
from flask import session
from flask import redirect , url_for , render_template , request , flash ,jsonify
# from model.admin_modle import Admin_Modle
import json
import ast
from datetime import datetime, date ,  timedelta
from collections import defaultdict
from app import app, rag_chain
from flask import request, jsonify

@app.route('/chatbot/api', methods=["GET", "POST"])
def chatPot():
    if request.method == 'GET':
        return "ChatBot API is working"

    elif request.method == 'POST':
        try:
            data = request.get_json()
            user_input = data.get("question", "")

            if not user_input:
                return jsonify({"error": "No question provided"}), 400
            
            # Process the input through the RAG chain
            response = rag_chain.invoke({"input": user_input})

            # Extract the correct output from the response
            if "answer" in response:
                chatbot_response = response["answer"]
            elif "output_text" in response:
                chatbot_response = response["output_text"]
            else:
                chatbot_response = "I'm sorry, I couldn't generate a response."

            return jsonify({"response": chatbot_response})

        except Exception as e:
            print("Error:", str(e))
            return jsonify({"error": str(e)}), 500
