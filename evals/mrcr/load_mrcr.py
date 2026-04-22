"""
evals/mrcr/load_mrcr.py

Load and sample the MRCR dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/openai/mrcr

Actual schema (verified against the live dataset):
    - prompt:                  str  — JSON-encoded list of {role, content} dicts
    - answer:                  str  — prefix + target_text (the full expected output)
    - random_string_to_prepend: str  — the random prefix (e.g. "mWEa9DrPT3")
    - desired_msg_index:       int  — absolute index of the USER message that
                                      *precedes* the needle; needle is at index+1
    - n_needles:               int  — number of needle occurrences in the convo
    - total_messages:          int  — total message count
    - n_chars:                 int  — character count
    - date_added:              str

Normalised output dict (used throughout the pipeline):
    - messages:          List[{role, content}]  — parsed from prompt
    - target_turn_index: int   — 0-based index among assistant messages (needle)
    - target_text:       str   — exact text the model must retrieve
    - prefix:            str   — random prefix to prepend
"""

import json
import logging
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_mrcr_samples(
    num_samples: Optional[int] = 20,
    seed: int = 42,
    dataset_name: str = "openai/mrcr",
    split: str = "train",
    n_needles: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load MRCR samples from HuggingFace, shuffle, and return a fixed slice.

    Shuffling is done with a fixed seed so the same seed always returns the
    same unique set of samples — but different seeds yield different sets,
    preventing cross-run sample reuse.

    Args:
        num_samples:  How many samples to return.
        seed:         Random seed for reproducible shuffling.
        dataset_name: HuggingFace dataset identifier.
        split:        Dataset split ("train" — the only available split).
        n_needles:    If set (2, 4, or 8), only use samples with this many
                      needle occurrences. None = use all samples.

    Returns:
        List of normalised sample dicts with keys:
            messages, target_turn_index, target_text, prefix
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required. "
            "Install it with: pip install datasets"
        ) from e

    logger.info(f"Loading dataset '{dataset_name}' split='{split}' ...")
    dataset = load_dataset(dataset_name, split=split)
    logger.info(f"Dataset loaded: {len(dataset)} total samples")

    # Filter by n_needles if requested
    if n_needles is not None:
        dataset = dataset.filter(lambda row: row["n_needles"] == n_needles)
        logger.info(
            f"Filtered to n_needles={n_needles}: {len(dataset)} samples remain"
        )

    # Shuffle indices with fixed seed → unique, reproducible sample per seed
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    # num_samples=None → return the full (shuffled) dataset
    if num_samples is None or num_samples >= len(dataset):
        if num_samples is not None and num_samples > len(dataset):
            logger.warning(
                f"Requested {num_samples} samples but dataset only has "
                f"{len(dataset)}. Using all available samples."
            )
        selected = indices
    else:
        selected = indices[:num_samples]

    samples = [_normalize_sample(dataset[i]) for i in selected]

    logger.info(f"Loaded {len(samples)} examples (seed={seed})")
    return samples


def _normalize_sample(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw MRCR row into the normalised dict used by the pipeline.

    Field mapping
    -------------
    raw["prompt"]                  → JSON string → parsed as messages list
    raw["random_string_to_prepend"] → prefix
    raw["answer"]                  → prefix + target_text
                                     so target_text = answer[len(prefix):]
    raw["desired_msg_index"]       → absolute index of the USER message that
                                     PRECEDES the needle assistant turn.
                                     Needle is at desired_msg_index + 1.
    """
    # ── Parse conversation ────────────────────────────────────────────────────
    messages: List[Dict[str, str]] = []
    for msg in json.loads(raw["prompt"]):
        messages.append(
            {
                "role": str(msg["role"]).lower(),
                "content": str(msg["content"]),
            }
        )

    # ── Locate needle ─────────────────────────────────────────────────────────
    # desired_msg_index points to the user turn; needle is the next message.
    needle_abs_idx = int(raw["desired_msg_index"]) + 1

    # Guard against edge cases (needle beyond message list)
    if needle_abs_idx >= len(messages):
        logger.warning(
            f"needle_abs_idx={needle_abs_idx} >= len(messages)={len(messages)}. "
            "Clamping to last message."
        )
        needle_abs_idx = len(messages) - 1

    # Convert absolute index → 0-based assistant-turn index
    # (counts only assistant messages that appear before the needle)
    target_turn_index = sum(
        1
        for i in range(needle_abs_idx)
        if messages[i]["role"] == "assistant"
    )

    # ── Extract prefix and target text ────────────────────────────────────────
    prefix = str(raw["random_string_to_prepend"])
    answer = str(raw["answer"])
    target_text = answer[len(prefix):]  # strip the prefix from the front

    return {
        "messages": messages,
        "target_turn_index": target_turn_index,
        "target_text": target_text,
        "prefix": prefix,
    }
