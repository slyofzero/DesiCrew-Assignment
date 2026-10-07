import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

AI_TOKEN = os.getenv("AI_TOKEN")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://aipipe.org/openrouter/v1")


def get_llm(model: str = "openai/gpt-4o", temperature: float = 0.0) -> ChatOpenAI:
    """Instantiate a LangChain ChatOpenAI instance routed through AIPipe."""
    if not AI_TOKEN:
        raise ValueError("AI_TOKEN is not set. Please set AI_TOKEN in backend/.env")

    return ChatOpenAI(
        model=model,
        api_key=SecretStr(AI_TOKEN),
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
