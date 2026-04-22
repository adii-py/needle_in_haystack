"""
evals/mrcr/prompt_builder.py

Build the prompt for MRCR evaluation in the OFFICIAL format:
    → Raw messages sent as a chat conversation (not a flat string).

Official implementation (from openai/mrcr dataset card):
    messages = json.loads(row["prompt"])          # already a list of {role, content}
    client.chat.completions.create(model=..., messages=messages)

The dataset's `prompt` field already contains the full conversation including
the final retrieval question as the last user turn.  No extra instruction needs
to be appended — the question is already there.

OpenCompass PromptList format:
    Each item is a dict {"role": "HUMAN"|"BOT"|"SYSTEM", "prompt": "<content>"}
    The OpenAISDK._preprocess_messages() method converts these to the OpenAI
    messages format: HUMAN→user, BOT→assistant, SYSTEM→system.
"""

import logging
from typing import Dict, List

from opencompass.utils.prompt import PromptList

logger = logging.getLogger(__name__)

# Role mapping: MRCR dataset roles → OpenCompass PromptList roles
_ROLE_MAP = {
    "user":      "HUMAN",
    "assistant": "BOT",
    "system":    "SYSTEM",
}


def build_prompt_list(messages: List[Dict[str, str]]) -> PromptList:
    """Convert raw MRCR messages to OpenCompass PromptList format.

    Each message dict {"role": "user"/"assistant", "content": "..."}
    is converted to {"role": "HUMAN"/"BOT", "prompt": "..."} so the
    OpenAISDK._preprocess_messages() method can map it to the OpenAI
    chat API format correctly.

    Args:
        messages: List of {role, content} dicts from the MRCR dataset.
                  The last message is the user's retrieval question.

    Returns:
        List of {"role": ..., "prompt": ...} dicts (PromptList-compatible).
    """
    prompt_list = PromptList()
    for msg in messages:
        oc_role = _ROLE_MAP.get(msg["role"].lower(), "HUMAN")
        prompt_list.append({"role": oc_role, "prompt": msg["content"]})
    return prompt_list


def build_prompt(
    messages: List[Dict[str, str]],
    prefix: str,
    target_text: str,
    topic=None,          # kept for API compatibility, unused in official format
) -> PromptList:
    """Build the full MRCR prompt as a PromptList.

    The dataset already includes the retrieval instruction as the final
    user turn (e.g. "Prepend mWEa9 to the 4th essay about leaders...").
    No additional instruction is appended — doing so would duplicate/
    contradict the dataset's own question.

    Args:
        messages:    Conversation messages (possibly truncated).
        prefix:      Prefix string (unused here, already in last user msg).
        target_text: Target text (unused here, already in last user msg).
        topic:       Unused, kept for backward compatibility.

    Returns:
        PromptList-compatible list of {"role", "prompt"} dicts.
    """
    return build_prompt_list(messages)
