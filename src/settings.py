from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw" / "liquid_cooling_docs"
COMPANIES_PATH = PROJECT_ROOT / "data" / "processed" / "companies.jsonl"

load_dotenv(PROJECT_ROOT / ".env")


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def require_openai_api_key() -> str:
    api_key = get_env("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create a .env file from .env.example "
            "and set OPENAI_API_KEY before running ingestion or question answering."
        )
    return api_key


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


def chroma_db_dir() -> str:
    value = get_env("CHROMA_DB_DIR", "chroma_db") or "chroma_db"
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def collection_name() -> str:
    return get_env("CHROMA_COLLECTION_NAME", "liquid_cooling_industry") or "liquid_cooling_industry"


def embedding_model() -> str:
    return get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small"


def embedding_provider() -> str:
    provider = (get_env("EMBEDDING_PROVIDER", "local") or "local").lower().strip()
    if provider not in {"local", "openai"}:
        raise RuntimeError("EMBEDDING_PROVIDER must be one of: local, openai.")
    return provider


def local_embedding_model() -> str:
    return (
        get_env("LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


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
