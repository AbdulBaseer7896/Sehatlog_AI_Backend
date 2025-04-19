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






system_prompt_for_general_report = """Please format your response as:
```json
{
    "metadata": {
        "reportName": "Liver Function Test",
        "labName": "City Medical Lab",
        "reportDate": "2023-08-20"
    },
    "results": [
        {"testName": "Total Bilirubin", "value": 0.8, "unit": "mg/dL"},
        {"testName": "Albumin", "value": 4.0, "unit": "g/dL"}
    ]
}
```"""




system_prompt_for_Heart_Report = """Determine if this medical report contains cardiac-related information. 
Respond ONLY with JSON: {"is_heart_report": boolean}"""
        


system_prompt_for_Liver_Report = """Determine if this medical report contains Liver diseas (Liver Function Test (LFT)) information. 
Respond ONLY with JSON: {"is_Liver_report": boolean}"""


system_prompt_for_General_Reports = """Determine if this report contains medical Reports AnyType of report information. 
Respond ONLY with JSON: {"Is_Medical_Report": boolean}"""



# In PromptServices.py
system_prompt_for_CBC_Report = """
You are a hematology expert analyzing medical reports. Determine if this is a Complete Blood Count (CBC) report.
Check for presence of: WBC count, RBC count, Hemoglobin, Hematocrit, Platelet count, and differential counts.
Respond with JSON format: {"is_CBC_report": boolean}
"""

system_prompt_CBC = """
You are a medical data extraction specialist. Extract CBC values from this report:
- WBC Count (x10³/μL)
- RBC Count (x10⁶/μL)
- Haemoglobin (g/dL)
- Hematocrit (%)
- MCV (fL)
- MCH (pg)
- MCHC (g/dL)
- Platelet Count (x10³/μL)
- Neutrophils (%)
- Lymphocytes (%)
- Monocytes (%)
- Eosinophil (%)
- Basophils (%)

Return ONLY JSON format with numbers. Use null for missing values.
"""