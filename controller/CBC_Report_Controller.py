from app import app
from flask import request, jsonify
import joblib
import numpy as np
import os
from utilits.Groq import GroqLLM
from services.OCR_Service import extrict_Data_From_image_using_OCR
from services.PromptServices import system_prompt_CBC, system_prompt_for_CBC_Report  # Updated prompts

# Load CBC model and preprocessing objects
cbc_model = joblib.load('TrainModelFiles\CBC_medical_model.pkl')  # Assuming this is your CBC model
cbc_scaler = cbc_model['scaler']
cbc_label_encoder = cbc_model['label_encoder']
cbc_model_instance = cbc_model['model']

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
UPLOAD_FOLDER = 'uploads'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# CBC Configuration
CBC_FEATURES = [
    'WBC Count', 'RBC Count', 'Haemoglobin', 'Hematocrit',
    'MCV', 'MCH', 'MCHC', 'Platelet Count', 'Neutrophils',
    'Lymphocytes', 'Monocytes', 'Eosinophil', 'Basophils'
]

cbc_feature_averages = {
    "WBC Count": 7.0,
    "RBC Count": 4.7,
    "Haemoglobin": 14.0,
    "Hematocrit": 42.0,
    "MCV": 90.0,
    "MCH": 27.0,
    "MCHC": 33.0,
    "Platelet Count": 250.0,
    "Neutrophils": 55.0,
    "Lymphocytes": 35.0,
    "Monocytes": 7.0,
    "Eosinophil": 2.0,
    "Basophils": 1.0
}

@app.route('/api/analyze-image/cbc', methods=['POST'])
def cbc_analyze_image():
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
            system_prompt_for_CBC_Report, 
            ocr_response.pages[0].markdown,
            report_key="is_CBC_report"
        )
        print("CBC report analysis:", report_analysis)

        # Data Extraction
        extracted_data = groq_llm.extract_medical_data(system_prompt_CBC, ocr_response.pages[0].markdown)
        print("Extracted CBC data:", extracted_data)
        
        if 'error' in extracted_data:
            return jsonify({
                "error": "Could not extract CBC data from image. Please enter manually.",
                "show_form": True,
            }), 200

        final_data = {**cbc_feature_averages, **extracted_data}
        
        return jsonify({
            "success": True,
            "is_CBC_report": report_analysis.get('is_CBC_report', False),
            "extracted_data": extracted_data,
            "form_data": final_data
        })
        
    except Exception as e:
        return jsonify({
            "error": "CBC image processing failed. Please enter data manually.",
            "show_form": True,
        }), 200

@app.route('/predict/cbc', methods=['POST'])
def cbc_predict():
    data = request.json.get('form_data', {})
    
    # Validate and prepare input data
    validated_data = {}
    for feature in CBC_FEATURES:
        value = data.get(feature)
        try:
            validated_data[feature] = float(value) if value not in [None, ''] else cbc_feature_averages[feature]
        except (ValueError, TypeError):
            validated_data[feature] = cbc_feature_averages[feature]

    # Convert to array and scale
    input_values = np.array([validated_data[feat] for feat in CBC_FEATURES]).reshape(1, -1)
    scaled_input = cbc_scaler.transform(input_values)
    
    # Make prediction
    prediction = cbc_model_instance.predict(scaled_input)
    diagnosis = cbc_label_encoder.inverse_transform(prediction)[0]
    
    return jsonify({
        "diagnosis": diagnosis,
        "used_values": validated_data,
        "status": "success"
    })