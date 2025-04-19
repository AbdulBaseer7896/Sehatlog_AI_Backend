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




    def analyze_medical_report(
            self,
            system_prompt: str,
            ocr_text: str,
            report_key: str = "is_heart_report"
        ) -> Dict[str, bool]:
            try:
                response = self._call(
                    prompt=f"MEDICAL REPORT CONTENT:\n{ocr_text}",
                    system_prompt=system_prompt
                )
                # Simplified JSON extraction
                json_str = response.replace("'", '"').strip()
                if json_str.startswith("```json"):
                    json_str = json_str[7:-3].strip()

                try:
                    result = json.loads(json_str)
                    # Return a dictionary using the provided report_key
                    return {report_key: result.get(report_key, False)}
                except json.JSONDecodeError:
                    # Fallback pattern matching (case insensitive)
                    if f'"{report_key}": true' in json_str.lower():
                        return {report_key: True}
                    if f'"{report_key}": false' in json_str.lower():
                        return {report_key: False}
                    return {report_key: False}

            except Exception as e:
                # Fail-safe return
                return {report_key: False}
            
            


    def extract_medical_data(self, system_prompt ,  ocr_text: str) -> Dict[str, Any]:
        try:
            print("Tis is the orc text = = = " , ocr_text)
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


    def extract_medical_data_for_General_Report(self, system_prompt, ocr_text: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ocr_text}
                ],
                temperature=0.1
            )

            print("This is the response from the llm = " , response)
            # Get the raw response content
            response_content = response.choices[0].message.content
        
            
            # Improved JSON extraction with error handling
            json_match = re.search(r'```json(.*?)```', response_content, re.DOTALL)
            if not json_match:
                # Fallback to search for JSON without code blocks
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            
            if not json_match:
                return {"error": "No JSON data found in LLM response"}
            
            json_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
            
            # Attempt to parse JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON format: {str(e)}"}
                
        except Exception as e:
            return {"error": f"Extraction failed: {str(e)}"}


    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "groq"