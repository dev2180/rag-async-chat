"""
MODULE: app/llm/ollama_client.py

Handles communication with local Ollama server.
Compatible with Ollama >= 0.14 (chat API).
"""

import requests
import json
from app.llm.base import BaseLLM


DEFAULT_MODEL = "qwen2.5:7b-instruct"


class OllamaClient(BaseLLM):

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=180,
        )

        response.raise_for_status()
        return response.json()["message"]["content"]

    def stream_generate(self, prompt: str):
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            stream=True,
            timeout=180,
        )

        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "message" in data:
                    yield data["message"]["content"]
