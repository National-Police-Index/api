"""OpenAI wrapper for agency validation.

The client is constructed lazily (on first call), so importing this module never
requires OPENAI_API_KEY — keeping the rest of the pipeline import-clean and testable.
Ported from legacy resolve/src/llm.py (standard OpenAI, caching disabled, jittered retry).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from shared.env import load_env

load_env()

logger = logging.getLogger(__name__)

AVAILABLE_MODELS = ["gpt-4.1-mini", "gpt-4.1", "gpt-5.4-nano"]
GPT4_TURBO_SEED = 1701180024

_client = None


class ContextLengthExceededError(Exception):
    pass


@dataclass(frozen=True)
class FewShotExample:
    prompt: str
    response: str


def _get_client():
    """Lazily build the OpenAI client (so import never needs a key)."""
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def prompt_gpt(
    prompt,
    few_shot_examples: Optional[List[FewShotExample]] = None,
    tools: Optional[List[Dict]] = None,
    model: str = "gpt-4.1-mini",
    debug: bool = False,
    cached: bool = True,  # accepted for back-compat; caching disabled
    logger: Optional[logging.Logger] = None,
):
    log = logger or logging.getLogger(__name__)

    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model} is not available. Available models: {AVAILABLE_MODELS}")

    from openai import BadRequestError

    client = _get_client()

    # Reasoning / gpt-5 models reject custom sampling params.
    restricted = "o1" in model or "o3" in model or model.startswith("gpt-5")
    temperature = 1 if ("o1" in model or "o3" in model) else 0

    few = few_shot_examples or []
    messages = []
    for fse in few:
        messages.append({"role": "user", "content": fse.prompt})
        messages.append({"role": "assistant", "content": fse.response})
    messages.append({"role": "user", "content": prompt})

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(multiplier=1, min=4, max=90),
        retry=retry_if_not_exception_type((ContextLengthExceededError, ValueError)),
    )
    def _call(messages):
        kwargs = dict(model=model, messages=messages)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if not restricted:
            kwargs.update(temperature=temperature, top_p=1, frequency_penalty=0,
                          presence_penalty=0, seed=GPT4_TURBO_SEED)
        try:
            completion = client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if getattr(e, "code", None) == "context_length_exceeded":
                raise ContextLengthExceededError(e)
            raise
        msg = completion.choices[0].message
        if tools:
            try:
                return json.loads(msg.tool_calls[0].function.arguments)
            except json.JSONDecodeError:
                log.error(f"JSON decode error: {completion}")
                return None
        return msg.content

    return _call(messages)
