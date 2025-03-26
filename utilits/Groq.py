# groq.py
from typing import List, Optional, Dict, Any
from pydantic import Field
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from groq import Groq
import re
import json

class GroqLLM(BaseLLM):
    client: Groq = Field(default=None, exclude=True)
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 500

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.3, max_tokens: int = 500):
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        object.__setattr__(self, "client", Groq(api_key=api_key))

    def _call(self, prompt: str, stop: Optional[List[str]] = None, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()


    def analyze_medical_report(self, ocr_text: str) -> Dict[str, Any]:
        system_prompt = """You are a cardiac medical report analyzer. Your task is to:
        1.content exists information in this JSON format:  and keep the is_heart_report : true
        {
            "is_heart_report": true,
            "cardiac_data": {
                "patient_info": { /* Any demographic info related to heart health */ },
                "measurements": { /* All cardiovascular metrics found */ },
                "findings": [ /* List of clinical observations */ ],
                "diagnosis": [ /* Cardiac-related diagnoses */ ],
                "recommendations": [ /* Heart-specific medical advice */ ]
            }
        }
        Include any relevant details - preserve original values and units from the report. 
        Return ONLY valid JSON, no commentary."""
        try:
            # Get raw LLM response
            response = self._call(
                prompt=f"MEDICAL REPORT CONTENT:\n{ocr_text}",
                system_prompt=system_prompt
            )
            # print("This is the repsone from llm = " , response)
            # Improved JSON extraction
            try:
                # Find JSON boundaries
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON found in response")
                
                json_str = response[json_start:json_end]
                
                # Clean common formatting issues
                json_str = json_str.replace('\\', '') \
                                .replace('\n', ' ') \
                                .replace('  ', ' ') \
                                .replace("‘", "'") \
                                .replace("’", "'")
                
                result = json.loads(json_str)
                
                # Validate structure
                if not result.get('is_heart_report', False):
                    print("its working")
                    return {"is_heart_report": False}
            
                print("This is = to the result = " , result)
                    
                if not result.get('data'):
                    raise ValueError("Missing data field in heart report")
                    
                return result
                
            except json.JSONDecodeError as e:
                return {
                    "error": f"JSON parsing failed: {str(e)}",
                    "raw_response": response,
                    "json_attempt": json_str
                }
            except Exception as e:
                return {"error": f"Validation failed: {str(e)}"}
                
        except Exception as e:
            return {"error": f"API request failed: {str(e)}"}
    



    def extract_medical_data(self, system_prompt ,  ocr_text: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ocr_text}
                ],
                temperature=0.1
            )

            # Extract JSON from response
            json_str = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL).group()
            return json.loads(json_str)
            
        except Exception as e:
            return {"error": str(e)}


    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "groq"