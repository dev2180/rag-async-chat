"""
MODULE: app/llm/ollama_client.py

Handles communication with local Ollama server.
Compatible with Ollama >= 0.14 (chat API).
"""

import logging
import requests
import json
from app.llm.base import BaseLLM
from app.config import OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLM):

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        logger.info(f"OllamaClient initialized: model={model}, url={base_url}")

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Model '{self.model}' not found on Ollama")
                raise RuntimeError(
                    f"Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                )
            raise

    def stream_generate(self, prompt: str):
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                },
                stream=True,
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data:
                        yield data["message"]["content"]

        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Start it with: ollama serve"
            )
