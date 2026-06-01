from dataclasses import dataclass
from typing import Optional, List, Dict

import tiktoken

from openai import OpenAI, BadRequestError
import json
from pprint import pprint
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    before_log,
    after_log,
    before_sleep_log,
    retry_if_not_exception_type,
)
import os
from dotenv import load_dotenv
load_dotenv()

import logging
logger = logging.getLogger('idonea.' + __name__)

tenacity_logger = logging.getLogger('tenacity')
tenacity_logger.setLevel(logging.DEBUG)

# Available models list
AVAILABLE_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5.4-nano",
]

# Switched from Azure to the standard OpenAI API: the Azure endpoint
# (clean-models.openai.azure.com) was decommissioned and no longer resolves.
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai_client = OpenAI(api_key=OPENAI_API_KEY)

logger.info("Using standard OpenAI endpoint for OpenAI services.")

GPT4_TURBO_SEED = 1701180024


class ContextLengthExceededError(Exception):
    pass


@dataclass(frozen=True)
class FewShotExample:
    prompt: str
    response: str


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)


def prompt_gpt(
        prompt,
        few_shot_examples: Optional[list[FewShotExample]] = None,
        tools: Optional[List[Dict]] = None,
        images: Optional[List[str]] = None,
        model='gpt-4.1-mini',  # Default to gpt-4.1-mini as shown in documentation
        debug=False,
        cached=True,  # Accepted for backwards compatibility; caching is disabled.
        logger=None,
    ) -> str or Dict or List:
    if logger is None:
        logger = logging.getLogger('idonea.' + __name__)

    # Validate model is available
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model} is not available. Available models: {AVAILABLE_MODELS}")

    gpt4_client = openai_client

    # Reasoning-style models (o1/o3, gpt-5 family) only accept the default
    # temperature and reject custom sampling params (temperature/top_p/seed).
    restricted_params = ('o1' in model or 'o3' in model or model.startswith('gpt-5'))
    temperature = 1 if ('o1' in model or 'o3' in model) else 0

    if few_shot_examples is None:
        few_shot_examples = []
    # Convert FewShotExample to dict to be compatible with OpenAI chat completions API
    few_shot_examples_dicts = []
    for fse in few_shot_examples:
        few_shot_examples_dicts.append({"role": "user", "content": fse.prompt})
        few_shot_examples_dicts.append({"role": "assistant", "content": fse.response})
    
    messages = few_shot_examples_dicts + [{"role": "user", "content": prompt}]

    # Handle images in the content
    if images:
        # The content becomes an array of text and image objects
        content = []
        # Add the text prompt
        content.append({"type": "text", "text": prompt})
        
        # Add each image
        for image_base64 in images:
            if not image_base64.startswith("data:"):
                # Ensure image has proper data URL format
                image_base64 = f"data:image/jpeg;base64,{image_base64}"
            
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_base64
                }
            })
        
        messages = [{"role": "user", "content": content}]
    
    if debug:
        logger.debug(
            f"Prompting GPT ({model} at {gpt4_client.base_url}) with prompt: \n# Few shot learning\n{json.dumps(few_shot_examples_dicts, indent=2)[:100]}...\n# Prompt\n{prompt[:1000]}...")

    @retry(
        stop=stop_after_attempt(6),
        # Randomized exponential backoff adds jitter so concurrent callers don't
        # retry in lockstep (the thundering-herd that caused APIConnectionError
        # under high validation concurrency).
        wait=wait_random_exponential(multiplier=1, min=4, max=90),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        retry=retry_if_not_exception_type((ContextLengthExceededError, ValueError)),
    )
    def prompt_model_endpoint(messages):
        logger.debug(f"~~~~~~~~ PROMPTING GPT ({model} at {gpt4_client.base_url}) with ~{len(json.dumps(messages))} characters of input data. ~~~~~~~~")
        
        # Use the model name directly as the deployment name
        api_model_string = model
        
        create_kwargs = dict(
            model=api_model_string,
            messages=messages,
        )
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "auto"
        # Reasoning/gpt-5 models reject these sampling params; only send them
        # for the classic chat models that support them.
        if not restricted_params:
            create_kwargs.update(
                temperature=temperature,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                seed=GPT4_TURBO_SEED,
            )

        try:
            chat_completion = gpt4_client.chat.completions.create(**create_kwargs)
        except BadRequestError as e:
            if e.code == 'context_length_exceeded':
                raise ContextLengthExceededError(e)
            else:
                raise e

        logger.info(f"Prompted GPT system: {chat_completion.system_fingerprint}")

        message = chat_completion.choices[0].message

        if tools:
            try:
                response_message = json.loads(message.tool_calls[0].function.arguments)
            except json.JSONDecodeError:
                response_message = None
                logger.error(f"JSON decode error: {chat_completion}")
        else:
            response_message = message.content

        if debug:
            logger.debug(response_message)

        return response_message

    # LLM response caching is intentionally disabled: we always want a fresh
    # call so results reflect the current model/prompt. The `cached` argument is
    # accepted for backwards compatibility but ignored.
    return prompt_model_endpoint(messages)