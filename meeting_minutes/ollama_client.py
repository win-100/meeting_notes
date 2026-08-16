import requests


class Ollama:
    """Small client for the local Ollama HTTP API."""

    def __init__(self, url):
        self.url = url.rstrip("/")

    def models(self):
        """Return the names of locally installed models."""
        response = requests.get(f"{self.url}/api/tags", timeout=3)
        return [model["name"] for model in response.json().get("models", [])]

    def generate(self, model, prompt, *, think=False):
        """Generate meeting minutes with ``model``, optionally disabling reasoning."""
        response = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": think,
            },
            timeout=1800,
        )
        response.raise_for_status()
        return response.json()["response"]

    def unload_all(self):
        """Release Ollama models from VRAM before ASR or diarization."""
        try:
            response = requests.get(f"{self.url}/api/ps", timeout=3)
            for model in response.json().get("models", []):
                requests.post(
                    f"{self.url}/api/generate",
                    json={"model": model["name"], "keep_alive": 0},
                    timeout=10,
                )
        except requests.RequestException:
            pass
