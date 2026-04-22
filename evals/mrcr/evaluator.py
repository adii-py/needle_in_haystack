"""
evals/mrcr/evaluator.py

Official MRCR evaluation metric — SequenceMatcher ratio.

Grading rule (from openai/mrcr dataset card):
    1. If the response does NOT start with the prefix → score = 0.0
    2. If it does → strip prefix from both response and ground-truth,
       then compute difflib.SequenceMatcher ratio.

This is a FUZZY match, not exact-match.  A score of 1.0 means verbatim
reproduction; lower scores reflect minor whitespace or formatting differences.

is_correct is reported at two thresholds:
    - strict:  score == 1.0  (byte-perfect)
    - lenient: score >= 0.8  (allows minor diffs)
"""

import logging
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Per-sample grading ────────────────────────────────────────────────────────

def grade(
    response: str,
    answer: str,
    prefix: str,
) -> float:
    """Official MRCR grading function.

    Args:
        response: Raw model output.
        answer:   Full ground-truth string (prefix + target_text).
        prefix:   The alphanumeric hash the model must prepend.

    Returns:
        Float in [0, 1].  0 if prefix absent; SequenceMatcher ratio otherwise.
    """
    if not response.startswith(prefix):
        return 0.0

    stripped_response = response.removeprefix(prefix)
    stripped_answer   = answer.removeprefix(prefix)
    return float(SequenceMatcher(None, stripped_response, stripped_answer).ratio())


def evaluate_prediction(
    prediction: str,
    prefix: str,
    target_text: str,
) -> Dict[str, Any]:
    """Evaluate a single model prediction.

    Args:
        prediction:  Raw model output string.
        prefix:      The prefix that must start the response.
        target_text: The exact target response (without prefix).

    Returns:
        Dict with:
            score        (float)  — SequenceMatcher ratio [0, 1]
            is_correct   (bool)   — score == 1.0 (strict exact match)
            is_acceptable (bool)  — score >= 0.8 (lenient)
            has_prefix   (bool)   — whether model included the prefix
            expected     (str)    — ground truth (prefix + target_text)
    """
    expected = prefix + target_text
    has_prefix = prediction.startswith(prefix)
    score = grade(prediction, expected, prefix)

    if not has_prefix:
        logger.debug(
            f"MISS (no prefix) | prefix={prefix!r} | "
            f"pred_start={prediction[:40]!r}"
        )
    elif score < 1.0:
        logger.debug(
            f"PARTIAL score={score:.3f} | "
            f"pred[:80]={prediction[:80]!r}"
        )

    return {
        "score": score,
        "is_correct": score == 1.0,
        "is_acceptable": score >= 0.8,
        "has_prefix": has_prefix,
        "expected": expected,
    }


# ─── Aggregate metrics ─────────────────────────────────────────────────────────

def _safe_mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def compute_metrics(
    results: List[Dict[str, Any]],
    run_id: str = "",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute aggregate metrics from per-sample result dicts.

    Args:
        results: List of dicts produced by runner.py (must include
                 score, is_correct, is_acceptable, truncated,
                 prompt_tokens, tokens_removed, target_turn_index).
        run_id:  Run identifier.
        config:  Config stored for reproducibility.

    Returns:
        Metrics dict.
    """
    total = len(results)
    if total == 0:
        logger.warning("compute_metrics called with empty results.")
        return {"avg_score": 0.0, "total": 0, "run_id": run_id}

    scores       = [r["score"] for r in results]
    correct      = [r for r in results if r["is_correct"]]
    acceptable   = [r for r in results if r["is_acceptable"]]
    has_prefix   = [r for r in results if r["has_prefix"]]
    skipped      = [r for r in results if r.get("skipped", False)]
    truncated    = [r for r in results if r.get("truncated", False)]
    non_trunc    = [r for r in results if not r.get("truncated", False)
                    and not r.get("skipped", False)]

    prompt_tokens     = [r["prompt_tokens"] for r in results]
    original_tokens   = [r.get("original_tokens", r["prompt_tokens"]) for r in results]
    tokens_removed    = [r.get("tokens_removed", 0) for r in results]
    coverages         = [r.get("coverage", 1.0) for r in results]

    def _acc(subset):
        return sum(1 for r in subset if r["is_acceptable"]) / len(subset) if subset else None

    # Depth quartile breakdown
    turn_indices = [r.get("target_turn_index", 0) for r in results]
    max_turn = max(turn_indices) if turn_indices else 1

    depth_scores: Dict[str, List[float]] = {
        "0-25%": [], "25-50%": [], "50-75%": [], "75-100%": []
    }
    for r in results:
        pct = 100 * r.get("target_turn_index", 0) / max(max_turn, 1)
        if pct <= 25:
            depth_scores["0-25%"].append(r["score"])
        elif pct <= 50:
            depth_scores["25-50%"].append(r["score"])
        elif pct <= 75:
            depth_scores["50-75%"].append(r["score"])
        else:
            depth_scores["75-100%"].append(r["score"])

    avg_score_by_depth = {
        k: _safe_mean(v) for k, v in depth_scores.items()
    }

    return {
        "run_id": run_id,
        # ── Primary metric (official) ───────────────────────────────────
        "avg_score": _safe_mean(scores),
        "median_score": sorted(scores)[total // 2],
        # ── Discrete thresholds ─────────────────────────────────────────
        "exact_accuracy": len(correct) / total,
        "lenient_accuracy": len(acceptable) / total,
        "prefix_hit_rate": len(has_prefix) / total,
        "num_exact": len(correct),
        "num_acceptable": len(acceptable),
        "total": total,
        # ── Truncation-stratified accuracy ──────────────────────────────
        "accuracy_truncated":     _acc(truncated),
        "accuracy_non_truncated": _acc(non_trunc),
        "truncation_rate": len(truncated) / total,
        # ── Skipped / truncated counts ──────────────────────────────────
        "num_skipped": len(skipped),
        "num_truncated": len(truncated),
        # ── Token / coverage stats ──────────────────────────────────────
        "avg_prompt_tokens":    _safe_mean(prompt_tokens),
        "max_prompt_tokens":    max(prompt_tokens),
        "min_prompt_tokens":    min(prompt_tokens),
        "avg_original_tokens":  _safe_mean(original_tokens),
        "avg_tokens_removed":   _safe_mean(tokens_removed),
        "avg_coverage":         _safe_mean(coverages),
        # ── Depth breakdown ─────────────────────────────────────────────
        "avg_score_by_depth": avg_score_by_depth,
        # ── Reproducibility ─────────────────────────────────────────────
        "config": config or {},
    }
