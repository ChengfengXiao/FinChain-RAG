from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from src.settings import get_env, llm_provider, provider_base_url, require_provider_api_key


@dataclass(frozen=True)
class ChatProviderConfig:
    provider: str
    model: str
    base_url: str | None


def resolve_chat_config(provider: str | None = None, model: str | None = None) -> ChatProviderConfig:
    selected_provider = (provider or llm_provider()).lower().strip()
    if selected_provider not in {"openai", "deepseek", "minimax"}:
        raise RuntimeError("provider must be one of: openai, deepseek, minimax.")
    selected_model = model or chat_model_for_provider(selected_provider)
    return ChatProviderConfig(
        provider=selected_provider,
        model=selected_model,
        base_url=provider_base_url(selected_provider),
    )


def chat_model_for_provider(provider: str) -> str:
    defaults = {
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-v4-flash",
        "minimax": "MiniMax-M3",
    }
    env_names = {
        "openai": "OPENAI_CHAT_MODEL",
        "deepseek": "DEEPSEEK_CHAT_MODEL",
        "minimax": "MINIMAX_CHAT_MODEL",
    }
    return get_env(env_names[provider], defaults[provider]) or defaults[provider]


def create_chat_client(provider: str | None = None) -> OpenAI:
    config = resolve_chat_config(provider=provider)
    api_key = require_provider_api_key(config.provider)
    if config.base_url:
        return OpenAI(api_key=api_key, base_url=config.base_url)
    return OpenAI(api_key=api_key)
