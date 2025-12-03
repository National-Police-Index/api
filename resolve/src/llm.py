from dataclasses import dataclass
from typing import Optional, List, Dict

import tiktoken

from openai import OpenAI, BadRequestError
import json
from diskcache import Cache
import blake3
from pprint import pprint
from openai.lib.azure import AzureOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
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
]

AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
API_VERSION = os.getenv('API_VERSION')


azure_gpt_client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=API_VERSION
)

logger.info("Using Azure endpoint for OpenAI services.")

GPT4_TURBO_SEED = 1701180024

CACHE_DIR = "llm-responses.cache"

llm_cache = Cache(CACHE_DIR)


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
        cached=True,  # Default to True, can be made configurable if needed
        cache=llm_cache,
        logger=None,
    ) -> str or Dict or List:
    if logger is None:
        logger = logging.getLogger('idonea.' + __name__)

    # Validate model is available
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model} is not available. Available models: {AVAILABLE_MODELS}")

    gpt4_client = azure_gpt_client

    temperature = 0  # Default to 0 for most models
    if 'o1' in model or 'o3' in model:
        temperature = 1

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

    serialized_messages = json.dumps(messages, sort_keys=True)
    if tools:
        serialized_messages += json.dumps(tools, sort_keys=True)
    hash_value = blake3.blake3(serialized_messages.encode()).hexdigest()
    cache_key = f"llm_response-{model}:{hash_value}"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=10, max=80),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        retry=retry_if_not_exception_type((ContextLengthExceededError, ValueError)),
    )
    def prompt_model_endpoint(messages):
        logger.debug(f"LLM Cache MISS for {cache_key}\n~~~~~~~~ PROMPTING GPT ({model} at {gpt4_client.base_url}) with ~{len(json.dumps(messages))} characters of input data. ~~~~~~~~")
        
        # Use the model name directly as the deployment name
        api_model_string = model
        
        try:
            if tools:
                chat_completion = gpt4_client.chat.completions.create(
                    model=api_model_string,
                    messages=messages,
                    tools=tools,
                    tool_choice='auto',
                    temperature=temperature,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    seed=GPT4_TURBO_SEED
                )
            else:
                chat_completion = gpt4_client.chat.completions.create(
                    model=api_model_string,
                    messages=messages,
                    temperature=temperature,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    seed=GPT4_TURBO_SEED
                )
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

        if response_message:
            llm_cache.set(cache_key, response_message)
        return response_message

    if cached:
        response_message = llm_cache.get(cache_key)
        if response_message:
            logger.debug(f"LLM Cache HIT for {cache_key}")
        # Because there are responses in the cache from before we created the current key syntax...
        if not response_message and model == 'gpt-4-1106-preview':
            old_cache_key = f"prompt_gpt4_turbo:{hash_value}"
            response_message = llm_cache.get(old_cache_key)  # <- Old key
            if response_message:
                logger.debug(f"LLM Cache HIT for {old_cache_key}")
        if response_message:
            if debug:
                logger.debug(response_message)
            return response_message
        else:
            return prompt_model_endpoint(messages)
    else:
        return prompt_model_endpoint(messages)