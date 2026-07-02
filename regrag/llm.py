"""Provider-agnostic chat client.

Targets any OpenAI-compatible endpoint (OpenRouter by default, Together AI, or
others) through the `openai` SDK with a per-provider base URL. Keys are read
from the environment (.env). For reproducible evaluation on OpenRouter, routing
can be pinned to a single upstream provider with fallbacks disabled, so a run
cannot silently switch vendors mid-eval.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from regrag import config

# Load .env from the repo root regardless of the current working directory.
load_dotenv(config.REPO_ROOT / ".env")


def _client(provider: str) -> OpenAI:
    try:
        base_url = config.PROVIDER_BASE_URLS[provider]
        key_env = config.PROVIDER_API_KEY_ENV[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown provider '{provider}'") from exc

    api_key = os.getenv(key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing {key_env}. Set it in .env (copy from .env.example)."
        )

    default_headers = {"X-Title": "RegRAG-AML"} if provider == "openrouter" else None
    return OpenAI(base_url=base_url, api_key=api_key, default_headers=default_headers)


def chat(
    messages: list[dict],
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    pin_routing: bool = False,
) -> str:
    """Send a chat completion and return the assistant's text.

    `pin_routing=True` (OpenRouter only) forces a single upstream provider with
    fallbacks disabled, for reproducible eval. It has no effect if
    OPENROUTER_PIN_PROVIDER is unset.
    """
    provider = provider or config.GENERATOR_PROVIDER
    model = model or config.GENERATOR_MODEL

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if provider == "openrouter" and pin_routing and config.OPENROUTER_PIN_PROVIDER:
        kwargs["extra_body"] = {
            "provider": {
                "order": [config.OPENROUTER_PIN_PROVIDER],
                "allow_fallbacks": config.OPENROUTER_ALLOW_FALLBACKS,
            }
        }

    response = _client(provider).chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""
