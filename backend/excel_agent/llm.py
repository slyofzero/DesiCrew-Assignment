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


def get_llm(model: str = "openai/gpt-4o", temperature: float = 0.0) -> ChatOpenAI:
    """Instantiate a LangChain ChatOpenAI instance routed through AIPipe."""
    token = os.getenv("AI_TOKEN") or os.getenv("AI_PIPE_TOKEN") or AI_TOKEN
    if not token:
        # Retry explicit reload if missing
        if env_path.exists():
            load_dotenv(env_path, override=True)
        token = os.getenv("AI_TOKEN") or os.getenv("AI_PIPE_TOKEN")

    if not token:
        raise ValueError("AI_TOKEN is not set. Please set AI_TOKEN or AI_PIPE_TOKEN in backend/.env")

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(token),
        base_url=AI_BASE_URL,
        temperature=temperature,
    )


if __name__ == "__main__":
    print(f"Connecting to AIPipe at {AI_BASE_URL}...")
    llm = get_llm(model="openai/gpt-4o")
    prompt = "What's a cow?"
    print(f"Sending prompt: {prompt}")
    response = llm.invoke(prompt)
    print("\n--- LLM Response ---")
    print(response.content)
