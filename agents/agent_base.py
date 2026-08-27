"""
agents/agent_base.py
====================
Shared helpers for every LLM agent: a role prompt wrapper and a
tolerant JSON decoder using the LLMClient.
"""

from __future__ import annotations

from llm import LLMConfigError, make_client
from llm import _extract_json


class AgentError(RuntimeError):
    """Raised when an agent cannot complete its task."""


def run_json_agent(
    role_description: str,
    task: str,
    provider: str | None = None,
    temperature: float | None = None,
):
    """
    Run an LLM agent that must return JSON, with robust parsing.

    Returns the parsed Python object (dict/list) or raises AgentError.
    """
    try:
        client = make_client(provider)
    except LLMConfigError as exc:
        raise AgentError(str(exc)) from exc
    if temperature is not None:
        client.temperature = temperature

    prompt = (
        f"{role_description}\n\n"
        f"Return your answer as VALID JSON only (no markdown fences, no "
        f"extra prose).\n\nTASK:\n{task}"
    )
    try:
        raw = client.complete_text(prompt)
        return _extract_json(raw)
    except (ValueError, LLMConfigError) as exc:
        raise AgentError(str(exc)) from exc


def run_text_agent(
    role_description: str,
    task: str,
    provider: str | None = None,
    temperature: float | None = None,
) -> str:
    try:
        client = make_client(provider)
    except LLMConfigError as exc:
        raise AgentError(str(exc)) from exc
    if temperature is not None:
        client.temperature = temperature
    prompt = (
        f"{role_description}\n\nTASK:\n{task}"
    )
    try:
        return client.complete_text(prompt)
    except LLMConfigError as exc:
        raise AgentError(str(exc)) from exc
