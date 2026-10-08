import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

AI_TOKEN = os.getenv("AI_TOKEN")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://aipipe.org/openrouter/v1")


def get_idp_llm(
    model: str = "amazon/nova-lite-v1",
    temperature: float = 0.0,
    max_tokens: int = 350,
) -> ChatOpenAI:
    """
    Instantiate a LangChain ChatOpenAI instance routed through AIPipe
    using the ultra-budget amazon/nova-lite-v1 model ($0.06/M tokens)
    for pure-text OCR reasoning, typo correction, and Chain of Thought reflection.
    """
    if not AI_TOKEN:
        raise ValueError("AI_TOKEN is not set in backend/.env")

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(AI_TOKEN),
        base_url=AI_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
