"""Anthropic API wrapper for ICRL generation."""

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.environ.get("ICRL_MODEL", "claude-opus-4-20250514")
JUDGE_MODEL = os.environ.get("ICRL_JUDGE_MODEL", DEFAULT_MODEL)


def get_client():
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("Install anthropic: pip install anthropic") from exc
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ANTHROPIC_API_KEY in environment or .env file")
    return anthropic.Anthropic(api_key=api_key)


def complete(
    client,
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    model = model or DEFAULT_MODEL
    last_err = None
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parts = [b.text for b in msg.content if hasattr(b, "text")]
            return "".join(parts).strip()
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Anthropic API failed after {max_retries} tries: {last_err}")


def complete_json(
    client,
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 256,
) -> dict[str, Any]:
    text = complete(client, system, user, model=model, max_tokens=max_tokens, temperature=0.0)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Expected JSON in response: {text[:200]}")
    return json.loads(match.group())
