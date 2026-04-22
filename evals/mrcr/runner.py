"""
evals/mrcr/runner.py

Main MRCR evaluation pipeline — aligned with the official OpenAI implementation.

Official flow (from openai/mrcr dataset card):
    1. Load dataset rows
    2. Count tokens; SKIP rows that exceed MAX_CONTEXT_WINDOW
    3. Send raw messages list directly to chat API
    4. Grade with SequenceMatcher ratio

Our flow:
    1. Load & shuffle dataset (seed for reproducibility)
    2. Count tokens:
       - If fits → send as-is  (official behaviour)
       - If over  → truncate   (optional fallback, config: allow_truncation)
    3. Convert messages to OpenCompass PromptList → model.generate()
    4. Grade with SequenceMatcher ratio
    5. Save predictions (NIAH-style) + results + metrics

Usage:
    python -m evals.mrcr.runner
    python -m evals.mrcr.runner --num-samples 5 --n-needles 2
    python -m evals.mrcr.runner --seed 99 --no-truncation
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ─── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.mrcr.evaluator import compute_metrics, evaluate_prediction
from evals.mrcr.load_mrcr import load_mrcr_samples
from evals.mrcr.preprocess import (
    count_tokens,
    get_tokenizer,
    truncate_messages,
    find_all_needle_indices,
)
from evals.mrcr.prompt_builder import build_prompt

logger = logging.getLogger(__name__)


# ─── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ─── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logger.debug(f"Config: {cfg}")
    return cfg


# ─── Model ─────────────────────────────────────────────────────────────────────

def load_model(config: Dict[str, Any]):
    """Initialise LiteLLM/OpenAI compatible model."""
    # OpenCompass removed litellm_api, use OpenAISDK as the standard
    from opencompass.models.openai_api import OpenAISDK

    model = OpenAISDK(
        path=config.get("model_name", os.getenv("LITE_LLM_MODEL", "gpt-4")),
        max_seq_len=config.get("max_context_tokens", 190_000),
        temperature=config.get("temperature", 0),
        query_per_second=config.get("query_per_second", 1),
        retry=config.get("retry", 3),
        timeout=config.get("timeout", 3600),
        tokenizer_path=config.get("tokenizer_path", "gpt-4"),
        max_workers=config.get("max_workers", 5),
        openai_api_base=os.getenv("LITE_LLM_URL", "https://api.openai.com/v1/"),
        key=os.getenv("LITE_LLM_API_KEY", "ENV"),
    )
    logger.info(f"Model ready | path={getattr(model, 'path', 'unknown')}")
    return model


# ─── Token counting for message lists ─────────────────────────────────────────

def count_messages_tokens(messages: List[Dict[str, str]], tokenizer) -> int:
    """Sum token count across all message contents."""
    return sum(count_tokens(m["content"], tokenizer) for m in messages)


# ─── Prediction post-processing ────────────────────────────────────────────────

_THINK_TAG = "</think>"

def postprocess_prediction(raw: str) -> str:
    """Strip OpenCompass think-tag wrapping from model output.

    OpenCompass constructs the returned string as:
        reasoning_content + '</think>' + content

    The actual answer is everything after '</think>', with leading whitespace
    stripped (the API typically adds a newline/space before the answer).

    If '</think>' is absent the raw string is returned unchanged.
    """
    idx = raw.find(_THINK_TAG)
    if idx == -1:
        return raw
    return raw[idx + len(_THINK_TAG):].lstrip()


# ─── Per-sample processing ─────────────────────────────────────────────────────

def process_sample(
    sample: Dict[str, Any],
    config: Dict[str, Any],
    tokenizer,
    allow_truncation: bool = True,
) -> Dict[str, Any]:
    """Prepare a single sample: count tokens, skip or truncate, build prompt.

    Official behaviour: skip samples that exceed MAX_CONTEXT_WINDOW.
    Our fallback:       truncate if allow_truncation=True (config-driven).

    Returns a dict with:
        prompt           — PromptList (list of {role, prompt} dicts)
        original_tokens  — token count BEFORE any truncation
        prompt_tokens    — token count of the final prompt (after truncation)
        coverage         — prompt_tokens / original_tokens  [0, 1]
        skipped          — True if sample was skipped (over limit, no truncation)
        truncated        — True if messages were truncated
        tokens_removed   — how many tokens were dropped
        ... plus original sample fields
    """
    messages      = sample["messages"]
    target_tix    = sample["target_turn_index"]
    target_text   = sample["target_text"]
    prefix        = sample["prefix"]
    max_ctx       = config.get("max_context_tokens", 190_000)
    overhead      = 300  # instruction + formatting overhead (tokens)

    raw_tokens = count_messages_tokens(messages, tokenizer)

    # ── Over-limit handling ───────────────────────────────────────────────────
    was_truncated = False
    tokens_removed = 0

    if raw_tokens > max_ctx - overhead:
        if not allow_truncation:
            # Official behaviour: skip this sample
            logger.info(
                f"SKIP sample (tokens={raw_tokens:,} > limit={max_ctx:,})"
            )
            return {
                "prompt":                 None,
                "original_tokens":        raw_tokens,
                "prompt_tokens":          raw_tokens,
                "coverage":               1.0,  # nothing removed yet — just skipped
                "skipped":                True,
                "truncated":              False,
                "tokens_removed":         0,
                "num_messages_original":  len(messages),
                "num_messages_truncated": len(messages),
                "prefix":                 prefix,
                "target_text":            target_text,
                "target_turn_index":      target_tix,
            }

        # Truncation fallback — protect ALL needle occurrences
        messages, was_truncated, tokens_removed = truncate_messages(
            messages=messages,
            target_turn_index=target_tix,
            max_context_tokens=max_ctx,
            prompt_overhead_tokens=overhead,
            tokenizer=tokenizer,
            target_text=target_text,
        )

    # ── Build PromptList (official: raw messages → chat API) ─────────────────
    prompt = build_prompt(messages, prefix, target_text)
    prompt_tokens = count_messages_tokens(messages, tokenizer)
    coverage = prompt_tokens / raw_tokens if raw_tokens > 0 else 1.0

    return {
        "prompt":                 prompt,
        "original_tokens":        raw_tokens,
        "prompt_tokens":          prompt_tokens,
        "coverage":               coverage,
        "skipped":                False,
        "truncated":              was_truncated,
        "tokens_removed":         tokens_removed,
        "num_messages_original":  len(sample["messages"]),
        "num_messages_truncated": len(messages),
        "prefix":                 prefix,
        "target_text":            target_text,
        "target_turn_index":      target_tix,
    }


# ─── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model,
    processed: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> List[Optional[str]]:
    """Run model.generate() on all non-skipped prompts.

    Skipped samples get None as their prediction.
    """
    max_out_len = config.get("max_out_len", 8192)
    temperature = config.get("temperature", 0)

    # Separate runnable from skipped; preserve order
    runnable_idx = [i for i, p in enumerate(processed) if not p["skipped"]]
    prompts = [processed[i]["prompt"] for i in runnable_idx]

    logger.info(
        f"Running inference on {len(prompts)} prompts "
        f"({len(processed) - len(prompts)} skipped) ..."
    )

    if not prompts:
        return [None] * len(processed)

    responses = model.generate(
        inputs=prompts,
        max_out_len=max_out_len,
        temperature=temperature,
    )
    logger.info("Inference complete.")

    # Re-insert None for skipped positions
    predictions: List[Optional[str]] = [None] * len(processed)
    for idx, resp in zip(runnable_idx, responses):
        predictions[idx] = resp

    return predictions


# ─── Output ────────────────────────────────────────────────────────────────────

def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_dir: str,
    run_id: str,
) -> tuple:
    """Save NIAH-style predictions, detailed results, and metrics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # NIAH-style predictions (keyed by string index)
    predictions_niah: Dict[str, Any] = {}
    for r in results:
        predictions_niah[str(r["id"])] = {
            "origin_prompt": r.get("origin_prompt", []),
            "prediction":    r["prediction"] or "",
            "gold":          r["ground_truth"],
            "score":         r["score"],
            "prefix":        r["prefix"],
            "target_text":   r["target_text"],
            "is_correct":    r["is_correct"],
            "is_acceptable": r["is_acceptable"],
            "has_prefix":    r["has_prefix"],
            "prompt_tokens": r["prompt_tokens"],
            "skipped":       r["skipped"],
            "truncated":     r["truncated"],
            "target_turn_index": r["target_turn_index"],
        }

    preds_path   = out / f"{run_id}_predictions.json"
    results_path = out / f"{run_id}_results.json"
    metrics_path = out / f"{run_id}_metrics.json"

    with open(preds_path, "w") as f:
        json.dump(predictions_niah, f, indent=2, ensure_ascii=False)
    logger.info(f"Predictions → {preds_path}")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results     → {results_path}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics     → {metrics_path}")

    return str(preds_path), str(results_path), str(metrics_path)


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def run_mrcr_eval(
    config_path: str,
    output_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    seed: Optional[int] = None,
    n_needles: Optional[int] = None,
    allow_truncation: Optional[bool] = None,
    max_context_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
    samples: Optional[List[Dict[str, Any]]] = None,
    model=None,
) -> tuple:
    """Full MRCR evaluation pipeline.

    Args:
        config_path:        Path to config.yaml.
        output_dir:         Override output directory.
        num_samples:        Override num_samples.
        seed:               Override seed.
        n_needles:          Filter by needle count (2, 4, or 8). None = all.
        allow_truncation:   Override allow_truncation from config.
        max_context_tokens: Override context window size (for multi-context runs).
        samples:            Pre-loaded samples (skips dataset loading if provided).
        model:              Pre-loaded model (skips model init if provided).

    Returns:
        (results, metrics)
    """
    config = load_config(config_path)

    # CLI overrides
    if output_dir:                config["output_dir"]          = output_dir
    if num_samples:               config["num_samples"]         = num_samples
    if seed is not None:          config["seed"]                = seed
    if n_needles:                 config["n_needles"]           = n_needles
    if allow_truncation is not None:
        config["allow_truncation"] = allow_truncation
    if max_context_tokens is not None:
        config["max_context_tokens"] = max_context_tokens

    cfg_output_dir    = config.get("output_dir",         "outputs/mrcr")
    cfg_num_samples   = config.get("num_samples",        20)
    cfg_seed          = config.get("seed",               42)
    cfg_n_needles     = config.get("n_needles",          None)
    cfg_allow_trunc   = config.get("allow_truncation",   True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(
        f"MRCR Eval | run_id={run_id} | samples={cfg_num_samples} | "
        f"seed={cfg_seed} | n_needles={cfg_n_needles} | "
        f"allow_truncation={cfg_allow_trunc} | "
        f"max_context_tokens={config.get('max_context_tokens')}"
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = get_tokenizer(config.get("tokenizer_encoding", "cl100k_base"))

    # ── Dataset (reuse pre-loaded if provided) ────────────────────────────────
    if samples is None:
        samples = load_mrcr_samples(
            num_samples=cfg_num_samples,
            seed=cfg_seed,
            dataset_name=config.get("dataset_name", "openai/mrcr"),
            split=config.get("dataset_split", "train"),
            n_needles=cfg_n_needles,
        )

    # ── Min-tokens filter ─────────────────────────────────────────────────────
    if min_tokens is not None:
        tokenizer_for_filter = get_tokenizer(
            config.get("tokenizer_encoding", "cl100k_base")
        )
        before = len(samples)
        samples = [
            s for s in samples
            if count_messages_tokens(s["messages"], tokenizer_for_filter) >= min_tokens
        ]
        logger.info(
            f"--min-tokens {min_tokens:,}: kept {len(samples)}/{before} samples "
            f"(dropped {before - len(samples)} with < {min_tokens:,} tokens)"
        )

    # ── Model (reuse pre-loaded if provided) ──────────────────────────────────
    if model is None:
        model = load_model(config)

    # ── Preprocess ────────────────────────────────────────────────────────────
    logger.info("Preprocessing samples ...")
    processed: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples):
        proc = process_sample(sample, config, tokenizer, cfg_allow_trunc)
        processed.append(proc)

        status = "SKIP" if proc["skipped"] else (
            "TRUNC" if proc["truncated"] else "OK"
        )
        logger.info(
            f"  [{i+1:>3}/{cfg_num_samples}] {status:<6} "
            f"tokens={proc['prompt_tokens']:>7,} | "
            f"msgs {proc['num_messages_original']}→{proc['num_messages_truncated']}"
        )

    # ── Inference ─────────────────────────────────────────────────────────────
    predictions = run_inference(model, processed, config)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Evaluating predictions ...")
    results: List[Dict[str, Any]] = []

    for i, (proc, pred) in enumerate(zip(processed, predictions)):
        if proc["skipped"] or pred is None:
            result: Dict[str, Any] = {
                "id":                    i,
                "origin_prompt":         proc.get("prompt") or [],
                "prediction":            None,
                "ground_truth":          proc["prefix"] + proc["target_text"],
                "target_text":           proc["target_text"],
                "prefix":                proc["prefix"],
                "score":                 0.0,
                "is_correct":            False,
                "is_acceptable":         False,
                "has_prefix":            False,
                "original_tokens":       proc["original_tokens"],
                "prompt_tokens":         proc["prompt_tokens"],
                "coverage":              proc["coverage"],
                "skipped":               True,
                "truncated":             False,
                "tokens_removed":        0,
                "num_messages_original": proc["num_messages_original"],
                "num_messages_truncated":proc["num_messages_truncated"],
                "target_turn_index":     proc["target_turn_index"],
            }
        else:
            # Strip reasoning content (everything before </think>) added by
            # OpenCompass when the model returns a separate reasoning_content
            # field alongside the answer content.
            pred = postprocess_prediction(pred)
            ev = evaluate_prediction(pred, proc["prefix"], proc["target_text"])
            result = {
                "id":                    i,
                "origin_prompt":         proc["prompt"],
                "prediction":            pred,
                "ground_truth":          ev["expected"],
                "target_text":           proc["target_text"],
                "prefix":                proc["prefix"],
                "score":                 ev["score"],
                "is_correct":            ev["is_correct"],
                "is_acceptable":         ev["is_acceptable"],
                "has_prefix":            ev["has_prefix"],
                "original_tokens":       proc["original_tokens"],
                "prompt_tokens":         proc["prompt_tokens"],
                "coverage":              proc["coverage"],
                "skipped":               False,
                "truncated":             proc["truncated"],
                "tokens_removed":        proc["tokens_removed"],
                "num_messages_original": proc["num_messages_original"],
                "num_messages_truncated":proc["num_messages_truncated"],
                "target_turn_index":     proc["target_turn_index"],
            }
        results.append(result)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(results, run_id=run_id, config=config)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 62)
    logger.info("MRCR Evaluation Complete")
    logger.info(f"  Avg score (SequenceMatcher) : {metrics['avg_score']:.3f}")
    logger.info(f"  Exact accuracy  (score=1.0) : {metrics['exact_accuracy']:.1%}")
    logger.info(f"  Lenient accuracy (≥0.8)     : {metrics['lenient_accuracy']:.1%}")
    logger.info(f"  Prefix hit rate             : {metrics['prefix_hit_rate']:.1%}")
    logger.info(f"  Total / Skipped / Truncated : "
                f"{metrics['total']} / {metrics['num_skipped']} / {metrics['num_truncated']}")
    logger.info(f"  Avg prompt tokens           : {metrics['avg_prompt_tokens']:,.0f}")
    logger.info("=" * 62)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(results, metrics, cfg_output_dir, run_id)

    return results, metrics


# ─── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MRCR evaluation — official metric (SequenceMatcher ratio)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",       default=str(Path(__file__).parent / "config.yaml"))
    p.add_argument("--output-dir",   default=None)
    p.add_argument("--num-samples",  type=int,   default=None)
    p.add_argument("--seed",         type=int,   default=None)
    p.add_argument("--n-needles",    type=int,   default=None,
                   choices=[2, 4, 8], help="Filter by needle count")
    p.add_argument("--no-truncation", action="store_true",
                   help="Skip over-limit samples instead of truncating (official behaviour)")
    p.add_argument("--min-tokens",   type=int,   default=None, metavar="N",
                   help="Only evaluate samples with >= N tokens")
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv
        dp = find_dotenv(usecwd=True)
        if dp:
            load_dotenv(dp, override=False)
    except ImportError:
        pass

    args = _build_parser().parse_args()
    setup_logging(args.log_level)

    run_mrcr_eval(
        config_path=args.config,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        n_needles=args.n_needles,
        allow_truncation=not args.no_truncation,
        min_tokens=args.min_tokens,
    )


if __name__ == "__main__":
    main()
