
from app import app
from flask import request, jsonify
import joblib
import numpy as np
import os
import tensorflow as tf
from utilits.Groq import GroqLLM
from services.OCR_Service import extrict_Data_From_image_using_OCR 
from services.PromptServices import system_prompt_for_general_report , system_prompt_for_General_Reports

# Load the saved model dictionary instead of a direct model object
saved_model_dict = joblib.load('./TrainModelFiles/Liver_Disease_model.pkl')

# Reconstruct the TensorFlow model from JSON and set its weights
model_json = saved_model_dict["model_json"]
weights = saved_model_dict["weights"]
model_instance = tf.keras.models.model_from_json(model_json)
model_instance.set_weights(weights)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
UPLOAD_FOLDER = 'uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/analyze-image/General_Reports', methods=['POST'])
def General_Reports():
    if 'medical_image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['medical_image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        # OCR Processing
        ocr_response = extrict_Data_From_image_using_OCR(file)
        

        groq_llm = GroqLLM(api_key=os.environ["GROQ_API_KEY"])
        report_analysis = groq_llm.analyze_medical_report(
            system_prompt_for_General_Reports, 
            ocr_response.pages[0].markdown,
            report_key="Is_Medical_Report"  # Use a different key for liver reports
        )
        print("Liver report analysis:", report_analysis)

        # Data Extraction
        extracted_data = groq_llm.extract_medical_data_for_General_Report(system_prompt_for_general_report, ocr_response.pages[0].markdown)
        print("Extracted data:", extracted_data)
        if 'error' in extracted_data:
            return jsonify({
                "error": "Could not extract data from image. Please enter manually.",
                "show_form": True,
                # "form_data": feature_averages
            }), 200

        final_data = {**extracted_data}
        print("Report flag:", report_analysis.get('Is_Medical_Report'))
        
        return jsonify({
            "success": True,
            "Is_Medical_Report": report_analysis.get('Is_Medical_Report', False),
            "extracted_data": extracted_data,
            # "default_values": feature_averages,
            "form_data": final_data
        })
        
    except Exception as e:
        return jsonify({
            "error": "Image processing failed. Please enter data manually.",
            "show_form": True,
            # "form_data": feature_averages
        }), 200

