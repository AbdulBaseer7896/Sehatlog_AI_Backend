
from app import app
from flask import request, jsonify
import joblib
import numpy as np
import os
import tensorflow as tf
from utilits.Groq import GroqLLM
from services.OCR_Service import extrict_Data_From_image_using_OCR
from services.PromptServices import system_prompt_Liver_disease, system_prompt_for_Liver_Report

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
feature_averages = {
    "Age": 65,
    "Gender": "Female",
    "Total_Bilirubin": 0.7,
    "Direct_Bilirubin": 0.1,
    "Alkaline_Phosphotase": 187,
    "Alamine_Aminotransferase": 16,
    "Aspartate_Aminotransferase": 18,
    "Total_Protiens": 6.8,
    "Albumin": 3.3,
    "Albumin_and_Globulin_Ratio": 0.9,
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/analyze-image/liver', methods=['POST'])
def liver_analyze_image():
    if 'medical_image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['medical_image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        # OCR Processing
        ocr_response = extrict_Data_From_image_using_OCR(file)
        
        # groq_llm = GroqLLM(api_key=os.environ["GROQ_API_KEY"])
        # report_analysis = groq_llm.analyze_medical_report(system_prompt_for_Liver_Report, ocr_response.pages[0].markdown)
        # print("Report analysis:", report_analysis)
        
        groq_llm = GroqLLM(api_key=os.environ["GROQ_API_KEY"])
        report_analysis = groq_llm.analyze_medical_report(
            system_prompt_for_Liver_Report, 
            ocr_response.pages[0].markdown,
            report_key="is_Liver_report"  # Use a different key for liver reports
        )
        print("Liver report analysis:", report_analysis)

        # Data Extraction
        extracted_data = groq_llm.extract_medical_data(system_prompt_Liver_disease, ocr_response.pages[0].markdown)
        print("Extracted data:", extracted_data)
        if 'error' in extracted_data:
            return jsonify({
                "error": "Could not extract data from image. Please enter manually.",
                "show_form": True,
                # "form_data": feature_averages
            }), 200

        final_data = {**feature_averages, **extracted_data}
        print("Report flag:", report_analysis.get('is_Liver_report'))
        
        return jsonify({
            "success": True,
            "is_Liver_report": report_analysis.get('is_Liver_report', False),
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

@app.route('/predict/liver_deases', methods=['POST'])
def liver_predict():
    print("Predict function called for liver disease")
    data = request.json.get('form_data', {})
    
    # Create validated data with fallback to averages
    validated_data = {}
    for key in feature_averages:
        value = data.get(key, None)
        try:
            validated_data[key] = float(value) if value not in [None, ''] else feature_averages[key]
        except (TypeError, ValueError):
            validated_data[key] = feature_averages[key]
    
    # Convert to numpy array in CORRECT ORDER
    input_values = np.array([
        validated_data['Age'],
        validated_data['Gender'],
        validated_data['Total_Bilirubin'],
        validated_data['Direct_Bilirubin'],
        validated_data['Alkaline_Phosphotase'],
        validated_data['Alamine_Aminotransferase'],
        validated_data['Aspartate_Aminotransferase'],
        validated_data['Total_Protiens'],
        validated_data['Albumin'],
        validated_data['Albumin_and_Globulin_Ratio']
    ], dtype=np.float64)
    
    input_data = input_values.reshape(1, -1)
    
    # Predict using the reconstructed model instance
    prediction = model_instance.predict(input_data)[0]
    result = "Liver Disease" if prediction == 1 else "No Liver Disease"
    
    return jsonify({
        "prediction": result,
        "used_values": validated_data
    })
