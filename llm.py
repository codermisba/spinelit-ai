"""
llm.py
======

Thin, dependency-light client for the LLM backend used by every agent.

Supported backends (chosen via `config.LLM_PROVIDER`):

- "gemini": Google Gemini free API (REST over stdlib `urllib`).
            Set GEMINI_API_KEY. Best accuracy / reliability for agents.
- "ollama": Local Ollama server (JSON-over-HTTP, no API key).
            Requires a running `ollama serve` + a pulled model.

Both backends expose a uniform `LLMClient` interface:

    client.complete_json(prompt)      -> parsed Python dict/list
    client.complete_text(prompt)      -> raw text string

Every agent gets a fresh client so prompts/roles stay isolated and
reproducible (low temperature, deterministic tools).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
import urllib.error

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_REST_URL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)


class LLMConfigError(RuntimeError):
    """Raised when the configured LLM backend is not usable."""


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` fences and surrounding prose a model might add."""
    text = text.strip()
    # JSON object / array fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # If the model wrapped it in a code block with language, strip that too
    return text


def _extract_json(text: str):
    """Best-effort parse of JSON from an LLM completion."""
    text = _strip_markdown_fences(text)

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Pull the largest balanced {...} or [...] span
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not parse JSON from model output:\n{text[:500]}")


class _BaseClient:
    temperature = LLM_TEMPERATURE

    def _chat(self, prompt: str, response_schema: str | None = None) -> str:
        raise NotImplementedError

    def complete_text(self, prompt: str) -> str:
        return self._chat(prompt).strip()

    def complete_json(self, prompt: str):
        raw = self._chat(prompt)
        return _extract_json(raw)


class GeminiClient(_BaseClient):
    """Google Gemini free-tier REST client (stdlib only)."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        # Read from the live environment at call time so `$env:GEMINI_API_KEY`
        # (or a saved .env style export) works without editing config.py.
        live_key = os.environ.get("GEMINI_API_KEY", "")
        self.api_key = (api_key or live_key or GEMINI_API_KEY).strip()
        self.model = model or GEMINI_MODEL
        if not self.api_key:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set. Get a free key from "
                "https://aistudio.google.com/apikey and set it in your "
                "environment or config.py."
            )

    def _chat(self, prompt: str, response_schema: str | None = None) -> str:
        url = (
            f"{GEMINI_REST_URL}{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        generation = {"temperature": self.temperature}
        if response_schema:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = response_schema

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation,
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise LLMConfigError(
                f"Gemini API error {exc.code}: {body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMConfigError(f"Network error calling Gemini: {exc}") from exc

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMConfigError(
                f"Unexpected Gemini response: {json.dumps(data)[:500]}"
            ) from exc


class OllamaClient(_BaseClient):
    """Local Ollama server client (JSON over HTTP)."""

    def __init__(self, host: str | None = None, model: str | None = None):
        import urllib.parse

        self.host = (host or OLLAMA_HOST).rstrip("/")
        self.model = model or OLLAMA_MODEL
        self._quote = urllib.parse.quote

    def _chat(self, prompt: str, response_schema: str | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.host + "/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=LLM_TIMEOUT) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMConfigError(
                f"Cannot reach Ollama at {self.host} ({exc}). Start it with "
                "`ollama serve` and pull a model, or switch LLM_PROVIDER."
            ) from exc
        return result.get("response", "")


def make_client(provider: str | None = None) -> _BaseClient:
    """Factory returning the configured LLM client for an agent."""
    provider = (provider or LLM_PROVIDER).strip().lower()
    if provider == "gemini":
        return GeminiClient()
    if provider == "ollama":
        return OllamaClient()
    raise LLMConfigError(f"Unknown LLM_PROVIDER: {provider!r}")


if __name__ == "__main__":
    import sys

    try:
        client = make_client()
        answer = client.complete_json(
            'Return JSON: {"echo": "hello agentic pipeline"}'
        )
        print("Provider     :", LLM_PROVIDER)
        print("Model        :", getattr(client, "model", "?"))
        print("JSON response:", answer)
    except LLMConfigError as exc:
        print("Not configured:", exc)
        sys.exit(1)
