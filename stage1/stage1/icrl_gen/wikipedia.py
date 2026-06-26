"""Sample seed paragraphs from English Wikipedia (50–200 words)."""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

USER_AGENT = "ValueAxisStage1/0.1 (research; contact: local)"


def _get(url: str, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _trim_to_word_range(text: str, min_words: int = 50, max_words: int = 200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
        text = " ".join(words)
    if len(words) < min_words and len(text) > 0:
        return text
    if len(words) < 20:
        return ""
    return text


def fetch_random_paragraph(timeout: float = 15.0) -> str:
    """Return one Wikipedia extract trimmed to roughly 50–200 words."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": "1",
            "exsection": "0",
            "generator": "random",
            "grnnamespace": "0",
            "grnlimit": "1",
        }
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    data = _get(url, timeout=timeout)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract", "")
        paragraph = _trim_to_word_range(extract)
        if paragraph:
            return paragraph
    return ""


def fetch_paragraphs(
    n: int,
    rng,
    *,
    max_attempts_per_paragraph: int = 8,
    on_retry: Callable[[str], None] | None = None,
) -> list[str]:
    """Fetch n distinct non-empty Wikipedia paragraphs."""
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        got = ""
        for attempt in range(max_attempts_per_paragraph):
            try:
                got = fetch_random_paragraph()
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if on_retry:
                    on_retry(f"Wikipedia fetch failed ({exc}); retrying...")
                time.sleep(0.5 * (attempt + 1))
                continue
            if got and got not in seen:
                break
            time.sleep(0.2)
        if not got or got in seen:
            raise RuntimeError(f"Could not fetch Wikipedia paragraph ({len(out)}/{n} so far)")
        seen.add(got)
        out.append(got)
    return out
