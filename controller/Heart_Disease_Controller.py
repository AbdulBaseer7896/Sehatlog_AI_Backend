
from app import app
from flask import  request, jsonify
import joblib
import numpy as np
import os
from utilits.Groq import GroqLLM
from services.OCR_Service import extrict_Data_From_image_using_OCR
from services.PromptServices import system_prompt_heart_disease


model = joblib.load('./TrainModelFiles/heart_disease_model.pkl')

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

UPLOAD_FOLDER = 'uploads'

feature_averages = {
    'age': 54.37,
    'sex': 0.68,
    'cp': 0.97,
    'trestbps': 131.62,
    'chol': 246.26,
    'fbs': 0.15,
    'restecg': 0.53,
    'thalach': 149.65,
    'exang': 0.33,
    'oldpeak': 1.04,
    'slope': 1.40,
    'ca': 0.73,
    'thal': 2.31
}



def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    print("This si the analyze Image function")
    if 'medical_image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    print("test 1")
    file = request.files['medical_image']
    print("test 2")
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400
    print("test 3")

    try:
        print("test 4")

        # OCR Processing
        ocr_response = extrict_Data_From_image_using_OCR(file)
        print("test 5")

        # Data Extraction
        groq_llm = GroqLLM(api_key=os.environ["GROQ_API_KEY"])
        print("test 6")
        extracted_data = groq_llm.extract_medical_data(  system_prompt_heart_disease , ocr_response.pages[0].markdown)
        print("test 7")
        # Merge with default values
        final_data = {**feature_averages, **extracted_data}
        print("This ithe 8")
        return jsonify({
            "success": True,
            "extracted_data": extracted_data,
            "default_values": feature_averages,
            "form_data": final_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/health_deases', methods=['POST'])
def predict():
    print("THis is the predict funciton")
    data = request.json.get('form_data', {})
    
    # Create validated data with fallback to averages
    validated_data = {}
    for key in feature_averages:
        # Get value from request or use default average
        value = data.get(key, None)
        
        # Convert to float if possible, otherwise use average
        try:
            validated_data[key] = float(value) if value not in [None, ''] else feature_averages[key]
        except (TypeError, ValueError):
            validated_data[key] = feature_averages[key]
    
    # Convert to numpy array in CORRECT ORDER
    input_values = np.array([
        validated_data['age'],
        validated_data['sex'],
        validated_data['cp'],
        validated_data['trestbps'],
        validated_data['chol'],
        validated_data['fbs'],
        validated_data['restecg'],
        validated_data['thalach'],
        validated_data['exang'],
        validated_data['oldpeak'],
        validated_data['slope'],
        validated_data['ca'],
        validated_data['thal']
    ], dtype=np.float64)
    
    input_data = input_values.reshape(1, -1)
    
    # Predict
    prediction = model.predict(input_data)[0]
    result = "Heart Disease" if prediction == 1 else "No Heart Disease"
    
    return jsonify({
        "prediction": result,
        "used_values": validated_data  # Show final values used
    })
