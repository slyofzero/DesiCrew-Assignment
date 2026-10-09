import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Robustly load .env from backend/ or root
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
load_dotenv(find_dotenv())

AI_TOKEN = os.getenv("AI_TOKEN") or os.getenv("AI_PIPE_TOKEN")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://aipipe.org/openrouter/v1")


def get_idp_llm(
    model: str = "amazon/nova-lite-v1",
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> ChatOpenAI:
    """
    Instantiate a LangChain ChatOpenAI instance routed through AIPipe
    using the ultra-budget amazon/nova-lite-v1 model ($0.06/M tokens)
    for pure-text OCR reasoning, typo correction, and Chain of Thought reflection.
    """
    token = os.getenv("AI_TOKEN") or os.getenv("AI_PIPE_TOKEN") or AI_TOKEN
    if not token:
        if env_path.exists():
            load_dotenv(env_path, override=True)
        token = os.getenv("AI_TOKEN") or os.getenv("AI_PIPE_TOKEN")

    if not token:
        raise ValueError("AI_TOKEN is not set in backend/.env")

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(token),
        base_url=AI_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
