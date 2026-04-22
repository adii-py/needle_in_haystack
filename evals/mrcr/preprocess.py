"""
evals/mrcr/preprocess.py

Token-aware truncation for MRCR samples.

Strategy — protect ALL needle occurrences:
    1. Always keep the final user message (the retrieval question).
    2. Find EVERY assistant message whose content matches target_text
       and protect ALL of them.  This is the critical MRCR fix:
       dropping any earlier copy shifts the occurrence count, making
       "prepend to the 2nd occurrence" point to the wrong message.
    3. Also keep the user turn immediately before each needle occurrence.
    4. If protecting everything exceeds the budget, fall back to keeping
       only the TARGET occurrence + final user message, and warn.
    5. Drop remaining messages oldest-first until the budget is met.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Tokenizer ────────────────────────────────────────────────────────────────

_TOKENIZER_CACHE: Dict[str, object] = {}


def get_tokenizer(encoding_name: str = "cl100k_base"):
    """Return a cached tiktoken tokenizer."""
    if encoding_name not in _TOKENIZER_CACHE:
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for token counting. "
                "Install it with: pip install tiktoken"
            ) from e
        _TOKENIZER_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _TOKENIZER_CACHE[encoding_name]


def count_tokens(text: str, tokenizer) -> int:
    """Count the number of tokens in *text* using *tokenizer*."""
    return len(tokenizer.encode(text, disallowed_special=()))


# ─── Message formatting ────────────────────────────────────────────────────────

def format_message(msg: Dict[str, str]) -> str:
    """Format a single message as a labelled line."""
    role = msg["role"].upper()
    content = msg["content"]
    if role == "USER":
        return f"[USER]: {content}"
    if role == "ASSISTANT":
        return f"[ASSISTANT]: {content}"
    return f"[{role}]: {content}"


# ─── Target-turn resolution ────────────────────────────────────────────────────

def resolve_target_message_index(
    messages: List[Dict[str, str]],
    target_turn_index: int,
) -> int:
    """Return the absolute message index of the *N*-th assistant turn.

    target_turn_index is 0-based among assistant messages.
    Returns -1 if not found.
    """
    assistant_count = 0
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            if assistant_count == target_turn_index:
                return i
            assistant_count += 1
    return -1


# ─── Needle occurrence finder ─────────────────────────────────────────────────

def find_all_needle_indices(
    messages: List[Dict[str, str]],
    target_text: str,
    match_prefix_len: int = 60,
) -> List[int]:
    """Return absolute indices of ALL assistant messages that match target_text.

    Matching uses the first *match_prefix_len* characters of target_text as a
    prefix check — fast and robust to minor trailing whitespace differences.

    Args:
        messages:         Full message list.
        target_text:      The needle text to search for.
        match_prefix_len: How many leading chars to use for matching.

    Returns:
        Sorted list of absolute message indices (assistant turns only).
    """
    if not target_text:
        return []

    prefix = target_text[:match_prefix_len]
    return [
        i
        for i, msg in enumerate(messages)
        if msg["role"] == "assistant" and msg["content"].startswith(prefix)
    ]


# ─── Truncation ────────────────────────────────────────────────────────────────

def truncate_messages(
    messages: List[Dict[str, str]],
    target_turn_index: int,
    max_context_tokens: int,
    prompt_overhead_tokens: int = 300,
    tokenizer=None,
    target_text: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], bool, int]:
    """Truncate *messages* to fit within the token budget.

    Protects ALL occurrences of the needle so the occurrence count in the
    conversation stays consistent with the retrieval instruction.

    Args:
        messages:               Full conversation turn list.
        target_turn_index:      0-based index among assistant messages (the needle).
        max_context_tokens:     Hard token limit for the entire prompt.
        prompt_overhead_tokens: Tokens reserved for instructions + formatting.
        tokenizer:              tiktoken-compatible tokenizer.
        target_text:            Needle text — used to find ALL occurrences.

    Returns:
        (truncated_messages, was_truncated, tokens_removed)
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    available = max_context_tokens - prompt_overhead_tokens

    # Per-message token counts (+1 for newline separator)
    msg_tokens = [
        count_tokens(format_message(m), tokenizer) + 1 for m in messages
    ]
    total_tokens = sum(msg_tokens)

    if total_tokens <= available:
        return messages, False, 0

    n = len(messages)

    # ── Step 1: locate the primary needle ─────────────────────────────────────
    target_idx = resolve_target_message_index(messages, target_turn_index)

    # ── Step 2: find ALL needle occurrences (the MRCR fix) ────────────────────
    if target_text:
        all_needle_indices = find_all_needle_indices(messages, target_text)
        logger.debug(
            f"Needle occurrences in conversation: {len(all_needle_indices)} "
            f"at indices {all_needle_indices}"
        )
    else:
        all_needle_indices = [target_idx] if target_idx >= 0 else []

    # ── Step 3: build the full protected set ──────────────────────────────────
    protected: set = set()

    # 3a. Final user message (the retrieval question — always keep)
    for i in range(n - 1, -1, -1):
        if messages[i]["role"] == "user":
            protected.add(i)
            break

    # 3b. ALL needle occurrences + their preceding user turn
    for idx in all_needle_indices:
        protected.add(idx)
        if idx > 0 and messages[idx - 1]["role"] == "user":
            protected.add(idx - 1)

    # 3c. Ensure the primary target is always protected (even if target_text
    #     matching missed it due to a minor content difference)
    if target_idx >= 0:
        protected.add(target_idx)
        if target_idx > 0 and messages[target_idx - 1]["role"] == "user":
            protected.add(target_idx - 1)

    # ── Step 4: check if protected set alone fits ─────────────────────────────
    protected_tokens = sum(msg_tokens[i] for i in protected)

    if protected_tokens > available:
        # Can't keep everything — fall back to target + final user only
        logger.warning(
            f"All needle occurrences ({len(all_needle_indices)}) + final user "
            f"message = {protected_tokens} tokens > budget {available}. "
            "Falling back to target occurrence + final user only."
        )
        # Rebuild protected with just target + final user
        protected = set()
        for i in range(n - 1, -1, -1):
            if messages[i]["role"] == "user":
                protected.add(i)
                break
        if target_idx >= 0:
            protected.add(target_idx)
            if target_idx > 0 and messages[target_idx - 1]["role"] == "user":
                protected.add(target_idx - 1)
        protected_tokens = sum(msg_tokens[i] for i in protected)

        if protected_tokens > available:
            logger.warning(
                f"Even minimal protected set ({protected_tokens} tok) exceeds "
                f"budget ({available}). Returning protected messages only."
            )
            kept = sorted(protected)
            removed = total_tokens - sum(msg_tokens[i] for i in kept)
            return [messages[i] for i in kept], True, removed

    # ── Step 5: drop oldest non-protected messages until budget is met ─────────
    kept = list(range(n))
    running_total = total_tokens

    for i in range(n):
        if running_total <= available:
            break
        if i in protected:
            continue
        kept.remove(i)
        running_total -= msg_tokens[i]

    tokens_removed = total_tokens - running_total
    was_truncated = tokens_removed > 0

    if was_truncated:
        remaining_needles = sum(
            1 for i in kept if i in set(all_needle_indices)
        )
        logger.debug(
            f"Truncated {tokens_removed} tokens | "
            f"kept {len(kept)}/{n} messages | "
            f"needle occurrences preserved: {remaining_needles}/{len(all_needle_indices)}"
        )

    return [messages[i] for i in kept], was_truncated, tokens_removed
