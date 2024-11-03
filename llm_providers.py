import os
from abc import ABC, abstractmethod
import ollama
import openai
from anthropic import Anthropic

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class OllamaProvider(LLMProvider):
    def __init__(self, model="llama3.2"):
        self.model = model
        
    def generate(self, prompt: str) -> str:
        response = ollama.generate(model=self.model, prompt=prompt)
        return response['response']

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai.api_key = self.api_key
        
    def generate(self, prompt: str) -> str:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class ClaudeProvider(LLMProvider):
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)
        
    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

def create_llm_provider(provider_name: str) -> LLMProvider:
    providers = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "claude": ClaudeProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Valid options are: {list(providers.keys())}")
    
    return providers[provider_name]()
