system_prompt_heart_disease = """Extract EXACTLY these numerical values from the medical report:
{
    "age": number,
    "sex": number,
    "cp": number,
    "trestbps": number,
    "chol": number,
    "fbs": number,
    "restecg": number,
    "thalach": number,
    "exang": number,
    "oldpeak": number,
    "slope": number,
    "ca": number,
    "thal": number
}
Return ONLY valid JSON. Use null for missing values."""