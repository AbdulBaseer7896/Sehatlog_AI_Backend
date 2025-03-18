
from app import app
from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('./TrainModelFiles/heart_disease_model.pkl')

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

@app.route('/predict/health_deases', methods=['POST'])
def predict():
    data = request.json
    
    # Fill missing values with averages
    filled_data = {
        key: data.get(key, feature_averages[key]) 
        for key in feature_averages
    }
    
    # Convert to numpy array in CORRECT ORDER
    input_values = np.array([filled_data[key] for key in feature_averages], dtype=np.float64)
    input_data = input_values.reshape(1, -1)
    
    # Predict
    prediction = model.predict(input_data)[0]
    result = "Heart Disease" if prediction == 1 else "No Heart Disease"
    
    return jsonify({
        "prediction": result,
        "filled_data": filled_data  # Optional: Show values used
    })