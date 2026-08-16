from meeting_minutes.ollama_client import Ollama


def test_generate_disables_reasoning(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "Compte-rendu"}

    def post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("meeting_minutes.ollama_client.requests.post", post)

    assert Ollama("http://ollama.local").generate("qwen", "prompt") == "Compte-rendu"
    assert captured["json"]["think"] is False
