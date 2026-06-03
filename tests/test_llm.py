"""The LLM module must import WITHOUT an OpenAI key (lazy client construction)."""
import importlib


def test_import_does_not_construct_client_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import resolve.llm as m
    importlib.reload(m)  # re-run module-level code with no key present
    assert hasattr(m, "prompt_gpt")
    assert callable(m.prompt_gpt)


def test_unavailable_model_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    import resolve.llm as m
    importlib.reload(m)
    import pytest
    with pytest.raises(ValueError):
        m.prompt_gpt("hi", model="not-a-real-model")
