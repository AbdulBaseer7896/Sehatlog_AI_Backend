from typing import List, Optional
from pydantic import Field
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from groq import Groq

class GroqLLM(BaseLLM):
    # Declare client as a field; exclude it from model serialization.
    client: Groq = Field(default=None, exclude=True)
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 500

    def __init__(self, api_key: str, model_name: str, temperature: float = 0.3, max_tokens: int = 500):
        # Initialize the base fields
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        # Set the client attribute using object.__setattr__ to bypass immutability
        object.__setattr__(self, "client", Groq(api_key=api_key))
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "max_tokens", max_tokens)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are an assistant for question-answering tasks. Use the context provided to answer concisely."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "groq"
