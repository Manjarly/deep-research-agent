"""LLM factory supporting Gemini, OpenAI, Claude, and local models."""
import os
from typing import Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel

from .config import config


def extract_text_content(content: Any) -> str:
    """Robustly extract plain text string from model responses (handles str, lists, dicts)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(parts)
    return str(content)


def get_llm(
    provider: str = "gemini",
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None
) -> BaseChatModel:
    """Instantiate a chat model based on provider and settings."""
    temp = temperature if temperature is not None else config.MODEL_TEMPERATURE
    provider = provider.lower()

    if provider == "gemini":
        key = api_key or config.GOOGLE_API_KEY
        if not key or key == "your_gemini_api_key_here":
            raise ValueError(
                "Google Gemini API key not found. Please set GOOGLE_API_KEY in your .env file "
                "or pass it via the UI/CLI. (Get one free at https://aistudio.google.com/app/apikey)"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = model_name or config.GEMINI_MODEL or "gemini-3.5-flash-lite"
        # Auto-map deprecated 1.5/2.0 names to the active 3.5-flash-lite
        if "1.5" in model or "2.0" in model or "2.5" in model:
            model = "gemini-3.5-flash-lite"

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=temp,
            convert_system_message_to_human=True,
        )

    elif provider == "openai":
        key = api_key or config.OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment or .env file.")
        from langchain_openai import ChatOpenAI

        model = model_name or "gpt-4o-mini"
        return ChatOpenAI(model=model, api_key=key, temperature=temp)

    elif provider == "anthropic":
        key = api_key or config.ANTHROPIC_API_KEY
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment or .env file.")
        from langchain_anthropic import ChatAnthropic

        model = model_name or "claude-3-5-sonnet-20241022"
        return ChatAnthropic(model=model, api_key=key, temperature=temp)

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        model = model_name or "llama3"
        return ChatOllama(model=model, temperature=temp)

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Choose 'gemini', 'openai', 'anthropic', or 'ollama'.")
