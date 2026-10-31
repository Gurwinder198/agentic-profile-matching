"""OpenRouter chat model and local embeddings factory.

Every other module that needs an LLM or embeddings imports from here
instead of constructing its own client, so the OpenRouter/local-embedding
configuration lives in exactly one place.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set — copy .env.example to .env and fill in your key"
        )
    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached local embeddings model (no API key required)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
