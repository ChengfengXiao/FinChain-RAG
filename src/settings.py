from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPANIES_PATH = PROJECT_ROOT / "data" / "processed" / "companies.jsonl"

load_dotenv(PROJECT_ROOT / ".env")


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def require_provider_api_key(provider: str) -> str:
    provider = provider.lower().strip()
    env_map = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "minimax": "MINIMAX_API_KEY",
    }
    env_name = env_map.get(provider)
    if not env_name:
        raise RuntimeError(f"Unsupported LLM provider '{provider}'. Use openai, deepseek, or minimax.")
    api_key = get_env(env_name)
    if not api_key:
        raise RuntimeError(
            f"{env_name} is missing. Set it in .env or switch LLM_PROVIDER to a configured provider."
        )
    return api_key


def chat_model() -> str:
    provider = llm_provider()
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


def llm_provider() -> str:
    provider = (get_env("LLM_PROVIDER", "deepseek") or "deepseek").lower().strip()
    if provider not in {"openai", "deepseek", "minimax"}:
        raise RuntimeError("LLM_PROVIDER must be one of: openai, deepseek, minimax.")
    return provider


def provider_base_url(provider: str) -> str | None:
    provider = provider.lower().strip()
    if provider == "openai":
        return get_env("OPENAI_BASE_URL")
    if provider == "deepseek":
        return get_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if provider == "minimax":
        return get_env("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
    raise RuntimeError(f"Unsupported LLM provider '{provider}'.")
