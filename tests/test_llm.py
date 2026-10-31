import pytest

import llm


def test_get_chat_model_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        llm.get_chat_model()


def test_get_chat_model_uses_default_model_and_openrouter_base_url(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    model = llm.get_chat_model()
    assert model.model_name == "openai/gpt-4o-mini"
    assert "openrouter.ai" in str(model.openai_api_base)


def test_get_embeddings_returns_384_dim_vector():
    vec = llm.get_embeddings().embed_query("React frontend developer")
    assert len(vec) == 384
