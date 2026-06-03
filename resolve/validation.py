"""Agency validation: deterministic non-LE guard + LLM tie-breaker.

The guard (resolve.agency) runs first and is import-clean. The LLM is only consulted
when the guard doesn't decide; `llm_fn` is injectable so this is testable offline and
the OpenAI client is imported lazily (not at module load).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

from resolve.agency import parse_agencies_to_check, non_le_guard

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """<task>
Determine if the POST agency matches any of the provided agencies. Agencies match ONLY if they refer to the EXACT SAME organization.
</task>

<matching_criteria>
Agencies match when:
- They are the same organization with different abbreviations (e.g., "Sacramento PD" vs "Sacramento Police Department")
- They have minor spelling differences or punctuation (e.g., "Sheriff's Office" vs "Sheriffs Office")
- They have slightly different formatting but same meaning (e.g., "Napa County Sheriff's Office" vs "Napa County Sheriff")

Agencies DO NOT match when:
- They are different police departments, even in nearby cities (e.g., "Corona Police Department" vs "Riverside Police Department")
- They are different sheriff departments from different counties
- They are different state prisons or correctional facilities
- One is a police department and another is a sheriff's department, even in the same area
</matching_criteria>

<post_agency>
{post_agency}
</post_agency>

<agencies_to_compare>
{agencies_list}
</agencies_to_compare>

<instructions>
Return ONLY "MATCH" if the POST agency matches any of the agencies to compare (the EXACT SAME organization).
Return ONLY "NO_MATCH" if it does not match any of them.
Do not include any explanation or other text.
</instructions>"""


def _default_llm(prompt: str, model: str):
    from resolve.llm import prompt_gpt
    return prompt_gpt(prompt, model=model, logger=logger)


def validate_agency_match(
    mention_agency: str,
    mentioned_agencies,
    post_agency: str,
    threshold: float = 0.8,  # accepted for back-compat; unused
    llm_fn: Optional[Callable] = None,
    model: str = "gpt-5.4-nano",
) -> Tuple[bool, str]:
    """Return (is_valid, reason). Validates whether `post_agency` refers to the same
    organization as the source agency or any mentioned agency."""
    if not post_agency:
        return False, "POST agency is empty"

    agencies = parse_agencies_to_check(mention_agency, mentioned_agencies)
    if not agencies:
        return False, "No agencies to compare against"

    blocked, reason = non_le_guard(agencies, post_agency)
    if blocked:
        return False, reason

    prompt = _PROMPT_TEMPLATE.format(
        post_agency=post_agency,
        agencies_list="\n".join(f"- {a}" for a in agencies),
    )

    call = llm_fn or (lambda p: _default_llm(p, model))
    try:
        response = call(prompt)
    except Exception as e:  # noqa: BLE001 - validation must never crash the pipeline
        logger.error(f"LLM agency validation error: {e}")
        return False, f"Validation error: {e}"

    cleaned = str(response).strip().upper()
    if "MATCH" in cleaned and "NO_MATCH" not in cleaned:
        return True, ""
    return False, "Agency cannot be validated"
