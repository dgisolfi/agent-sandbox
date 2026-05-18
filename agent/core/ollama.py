"""Ollama

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import requests


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                return False
            models = [m.get("name", "") for m in response.json().get("models", [])]
            if not any(m == self.model or m.startswith(self.model + ":") for m in models):
                print(f"[Agent] Model '{self.model}' is not pulled. Run: ollama pull {self.model}")
                return False
            return True
        except requests.RequestException:
            return False
