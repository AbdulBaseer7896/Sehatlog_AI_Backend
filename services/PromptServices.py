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





system_prompt_Liver_disease = """Extract EXACTLY these numerical values:
{
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
Return ONLY valid JSON. Use null for missing values."""




system_prompt_for_Heart_Report = """Determine if this medical report contains cardiac-related information. 
Respond ONLY with JSON: {"is_heart_report": boolean}"""
        


system_prompt_for_Liver_Report = """Determine if this medical report contains Liver diseas (Liver Function Test (LFT)) information. 
Respond ONLY with JSON: {"is_Liver_report": boolean}"""
        