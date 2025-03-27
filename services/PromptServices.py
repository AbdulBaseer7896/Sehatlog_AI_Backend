system_prompt_heart_disease = """Extract EXACTLY these numerical values:
{
    "age": number,
    "sex": number (0 for male, 1 for female),
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