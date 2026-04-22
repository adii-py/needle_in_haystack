"""
evals/mrcr/run_multi_context.py

Multi-context-length MRCR evaluation.

Runs the SAME sample set across multiple context window sizes and produces
per-context accuracy, truncation, and coverage metrics — analogous to
OpenAI's MRCR analysis across context lengths.

Design:
  - Dataset and model are loaded ONCE and shared across all context sizes.
  - For each context size, only the truncation/skip logic changes.
  - Output files: {output_dir}/{64k,128k,192k}.json  +  multi_context_summary.json

Usage:
    python -m evals.mrcr.run_multi_context
    python -m evals.mrcr.run_multi_context --context-sizes 64000 128000 192000
    python -m evals.mrcr.run_multi_context --num-samples 100 --n-needles 4
    python -m evals.mrcr.run_multi_context --no-truncation --seed 99
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ─── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.mrcr.evaluator import evaluate_prediction
from evals.mrcr.load_mrcr import load_mrcr_samples
from evals.mrcr.preprocess import get_tokenizer
from evals.mrcr.runner import (
    count_messages_tokens,
    load_config,
    load_model,
    postprocess_prediction,
    process_sample,
    run_inference,
    setup_logging,
)

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_SIZES = [64_000, 128_000, 192_000]


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _label(size: int) -> str:
    """64000 → '64k', 128000 → '128k', 192000 → '192k'."""
    return f"{size // 1000}k"


# ─── Per-context metrics ───────────────────────────────────────────────────────

def _compute_context_metrics(
    results: List[Dict[str, Any]],
    context_size: int,
) -> Dict[str, Any]:
    """Compute aggregated metrics for a single context-size run."""
    total = len(results)
    if total == 0:
        return {"context_size": context_size, "accuracy": None, "total": 0}

    runnable  = [r for r in results if not r["skipped"]]
    truncated = [r for r in runnable if r["truncated"]]
    non_trunc = [r for r in runnable if not r["truncated"]]
    skipped   = [r for r in results if r["skipped"]]

    def _acc(subset):
        if not subset:
            return None
        return sum(1 for r in subset if r["is_acceptable"]) / len(subset)

    def _mean(vals):
        return sum(vals) / len(vals) if vals else None

    runnable_scores = [r["score"] for r in runnable]

    return {
        "context_size":            context_size,
        "context_label":           _label(context_size),
        "total":                   total,
        "num_runnable":            len(runnable),
        "num_truncated":           len(truncated),
        "num_skipped":             len(skipped),
        # ── Accuracy ──────────────────────────────────────────────────────────
        "accuracy":                _acc(runnable),           # lenient (score ≥ 0.8)
        "exact_accuracy":          (
            sum(1 for r in runnable if r["is_correct"]) / len(runnable)
            if runnable else None
        ),
        "avg_score":               _mean(runnable_scores),
        # ── Truncation-stratified ─────────────────────────────────────────────
        "accuracy_truncated":      _acc(truncated),
        "accuracy_non_truncated":  _acc(non_trunc),
        # ── Rates ─────────────────────────────────────────────────────────────
        "truncation_rate":         len(truncated) / max(len(runnable), 1),
        "skip_rate":               len(skipped) / total,
        # ── Coverage ──────────────────────────────────────────────────────────
        "avg_coverage":            _mean([r["coverage"] for r in results]),
        "avg_original_tokens":     _mean([r["original_tokens"] for r in results]),
        "avg_used_tokens":         _mean([r["prompt_tokens"] for r in results]),
    }


# ─── Single-context run ────────────────────────────────────────────────────────

def run_for_context(
    context_size: int,
    samples: List[Dict[str, Any]],
    model,
    config: Dict[str, Any],
    tokenizer,
    allow_truncation: bool,
    output_dir: str,
    run_id: str,
) -> Dict[str, Any]:
    """Run the full eval pipeline for one context size.

    Saves results to {output_dir}/{label}.json and returns the payload dict.
    """
    label = _label(context_size)
    logger.info("=" * 62)
    logger.info(f"Context size: {label} ({context_size:,} tokens)")
    logger.info("=" * 62)

    # Config copy with overridden context window
    ctx_cfg = dict(config)
    ctx_cfg["max_context_tokens"] = context_size

    # ── Preprocess ─────────────────────────────────────────────────────────────
    logger.info("Preprocessing samples ...")
    processed: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples):
        proc = process_sample(sample, ctx_cfg, tokenizer, allow_truncation)
        processed.append(proc)
        status = "SKIP" if proc["skipped"] else ("TRUNC" if proc["truncated"] else "OK")
        logger.info(
            f"  [{i+1:>3}/{len(samples)}] {status:<6} "
            f"orig={proc['original_tokens']:>7,} → used={proc['prompt_tokens']:>7,} "
            f"cov={proc['coverage']:.3f}"
        )

    # ── Inference ──────────────────────────────────────────────────────────────
    predictions = run_inference(model, processed, ctx_cfg)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    for i, (proc, pred) in enumerate(zip(processed, predictions)):
        if proc["skipped"] or pred is None:
            result: Dict[str, Any] = {
                "id":                i,
                "score":             0.0,
                "is_correct":        False,
                "is_acceptable":     False,
                "has_prefix":        False,
                "prediction":        None,
                "ground_truth":      proc["prefix"] + proc["target_text"],
                "prefix":            proc["prefix"],
                "target_text":       proc["target_text"],
                "original_tokens":   proc["original_tokens"],
                "prompt_tokens":     proc["prompt_tokens"],
                "coverage":          proc["coverage"],
                "skipped":           True,
                "truncated":         False,
                "tokens_removed":    0,
                "target_turn_index": proc["target_turn_index"],
            }
        else:
            pred = postprocess_prediction(pred)
            ev = evaluate_prediction(pred, proc["prefix"], proc["target_text"])
            result = {
                "id":                i,
                "score":             ev["score"],
                "is_correct":        ev["is_correct"],
                "is_acceptable":     ev["is_acceptable"],
                "has_prefix":        ev["has_prefix"],
                "prediction":        pred,
                "ground_truth":      ev["expected"],
                "prefix":            proc["prefix"],
                "target_text":       proc["target_text"],
                "original_tokens":   proc["original_tokens"],
                "prompt_tokens":     proc["prompt_tokens"],
                "coverage":          proc["coverage"],
                "skipped":           False,
                "truncated":         proc["truncated"],
                "tokens_removed":    proc["tokens_removed"],
                "target_turn_index": proc["target_turn_index"],
            }
        results.append(result)

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics = _compute_context_metrics(results, context_size)
    logger.info(
        f"[{label}] accuracy={metrics['accuracy']:.1%}  "
        f"exact={metrics['exact_accuracy']:.1%}  "
        f"trunc_rate={metrics['truncation_rate']:.1%}  "
        f"coverage={metrics['avg_coverage']:.3f}"
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = Path(output_dir) / f"{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "context_size":  context_size,
        "context_label": label,
        "run_id":        run_id,
        "metrics":       metrics,
        "samples":       results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved → {out_path}")

    return payload


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load .env if available
    try:
        from dotenv import find_dotenv, load_dotenv
        dp = find_dotenv(usecwd=True)
        if dp:
            load_dotenv(dp, override=False)
    except ImportError:
        pass

    p = argparse.ArgumentParser(
        description="Multi-context-length MRCR evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",
                   default=str(Path(__file__).parent / "config.yaml"))
    p.add_argument("--output-dir",    default=None,
                   help="Output directory (default: from config.yaml)")
    p.add_argument("--num-samples",   type=int, default=None)
    p.add_argument("--seed",          type=int, default=None)
    p.add_argument("--n-needles",     type=int, default=None, choices=[2, 4, 8])
    p.add_argument("--context-sizes", type=int, nargs="+",
                   default=DEFAULT_CONTEXT_SIZES, metavar="N",
                   help="Context window sizes in tokens (space-separated)")
    p.add_argument("--no-truncation", action="store_true",
                   help="Skip over-limit samples (official) instead of truncating")
    p.add_argument("--min-tokens",     type=int, default=None, metavar="N",
                   help="Only evaluate samples with >= N tokens (global lower bound)")
    p.add_argument("--auto-bin",       action="store_true",
                   help="Per-context bin mode: each context size only runs samples "
                        "that fit in it but NOT in the previous (smaller) context. "
                        "E.g. 128k run only sees samples with 64k-128k tokens. "
                        "Gives a clean per-bin comparison without overlap.")
    p.add_argument("--samples-per-bin", type=int, default=100, metavar="N",
                   help="When --auto-bin is set, load the FULL dataset and sample "
                        "up to N examples from each bin independently. "
                        "Ignored when --auto-bin is not set (use --num-samples instead).")
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    setup_logging(args.log_level)

    # ── Config ─────────────────────────────────────────────────────────────────
    config = load_config(args.config)
    if args.output_dir:       config["output_dir"]  = args.output_dir
    if args.num_samples:      config["num_samples"] = args.num_samples
    if args.seed is not None: config["seed"]        = args.seed
    if args.n_needles:        config["n_needles"]   = args.n_needles

    output_dir    = config.get("output_dir", "outputs/mrcr")
    num_samples   = config.get("num_samples", 20)
    seed          = config.get("seed", 42)
    n_needles     = config.get("n_needles", None)
    allow_trunc    = not args.no_truncation
    min_tokens     = args.min_tokens
    auto_bin       = args.auto_bin
    samples_per_bin = args.samples_per_bin
    context_sizes  = sorted(args.context_sizes)
    run_id         = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(
        f"Multi-context MRCR | run_id={run_id} | "
        f"sizes={[_label(s) for s in context_sizes]} | "
        f"seed={seed} | n_needles={n_needles} | allow_truncation={allow_trunc} | "
        f"min_tokens={min_tokens} | auto_bin={auto_bin} | "
        f"{'samples_per_bin=' + str(samples_per_bin) if auto_bin else 'num_samples=' + str(num_samples)}"
    )

    # ── Shared resources (loaded once) ─────────────────────────────────────────
    tokenizer = get_tokenizer(config.get("tokenizer_encoding", "cl100k_base"))

    # ── Dataset loading ────────────────────────────────────────────────────────
    # In auto-bin mode: load the FULL dataset so each bin can sample up to
    # samples_per_bin independently (avoids bias from small global sample).
    # In normal mode:  load num_samples globally as before.
    load_n = None if auto_bin else num_samples
    logger.info(
        f"Loading {'FULL' if auto_bin else str(load_n)} dataset "
        f"(auto_bin={auto_bin}) ..."
    )
    samples = load_mrcr_samples(
        num_samples=load_n,
        seed=seed,
        dataset_name=config.get("dataset_name", "openai/mrcr"),
        split=config.get("dataset_split", "train"),
        n_needles=n_needles,
    )
    logger.info(f"Loaded {len(samples)} samples from dataset.")

    # ── Pre-compute token counts for all samples (used for filtering/binning) ──
    logger.info("Pre-computing token counts ...")
    sample_tokens = [
        count_messages_tokens(s["messages"], tokenizer) for s in samples
    ]

    # ── Global min-tokens filter ───────────────────────────────────────────────
    if min_tokens is not None:
        before = len(samples)
        pairs = [(s, t) for s, t in zip(samples, sample_tokens) if t >= min_tokens]
        samples       = [p[0] for p in pairs]
        sample_tokens = [p[1] for p in pairs]
        logger.info(
            f"--min-tokens {min_tokens:,}: kept {len(samples)}/{before} samples "
            f"(dropped {before - len(samples)} with < {min_tokens:,} tokens)"
        )

    logger.info("Loading model (shared across all context sizes) ...")
    model = load_model(config)

    # ── Run for each context size ──────────────────────────────────────────────
    rng = random.Random(seed)
    all_metrics: List[Dict[str, Any]] = []
    for i, ctx_size in enumerate(context_sizes):
        if auto_bin:
            # Per-bin sampling: filter to ONLY this bin's token range, then
            # randomly sample up to samples_per_bin from whatever is available.
            prev_size = context_sizes[i - 1] if i > 0 else 0
            bin_all = [
                s for s, t in zip(samples, sample_tokens)
                if prev_size < t <= ctx_size
            ]
            total_in_bin = len(bin_all)
            if total_in_bin > samples_per_bin:
                bin_samples = rng.sample(bin_all, samples_per_bin)
            else:
                bin_samples = bin_all
            logger.info(
                f"Bin {_label(ctx_size)} "
                f"({_label(prev_size) if prev_size else '0'} < tokens <= {_label(ctx_size)}): "
                f"{total_in_bin} available → using {len(bin_samples)} samples"
            )
        else:
            bin_samples = samples

        payload = run_for_context(
            context_size=ctx_size,
            samples=bin_samples,
            model=model,
            config=config,
            tokenizer=tokenizer,
            allow_truncation=allow_trunc,
            output_dir=output_dir,
            run_id=run_id,
        )
        all_metrics.append(payload["metrics"])

    # ── Summary table ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 75)
    logger.info("Multi-Context MRCR Summary")
    logger.info(
        f"{'Context':<10} {'Accuracy':>10} {'Exact':>8} "
        f"{'AccTrunc':>10} {'AccNoTrunc':>12} {'TruncRate':>10} {'Coverage':>10}"
    )
    logger.info("-" * 75)
    for m in all_metrics:
        def _fmt(v): return f"{v:.1%}" if v is not None else "   N/A"
        logger.info(
            f"{m['context_label']:<10}"
            f" {_fmt(m['accuracy']):>10}"
            f" {_fmt(m['exact_accuracy']):>8}"
            f" {_fmt(m['accuracy_truncated']):>10}"
            f" {_fmt(m['accuracy_non_truncated']):>12}"
            f" {_fmt(m['truncation_rate']):>10}"
            f" {m['avg_coverage']:.3f}  " if m['avg_coverage'] is not None
            else f" {'N/A':>10}"
        )
    logger.info("=" * 75)

    # ── Save summary ───────────────────────────────────────────────────────────
    summary = [
        {
            "context":                m["context_size"],
            "context_label":          m["context_label"],
            "accuracy":               m["accuracy"],
            "exact_accuracy":         m["exact_accuracy"],
            "avg_score":              m["avg_score"],
            "accuracy_truncated":     m["accuracy_truncated"],
            "accuracy_non_truncated": m["accuracy_non_truncated"],
            "truncation_rate":        m["truncation_rate"],
            "skip_rate":              m["skip_rate"],
            "avg_coverage":           m["avg_coverage"],
            "num_truncated":          m["num_truncated"],
            "num_skipped":            m["num_skipped"],
            "total":                  m["total"],
            "min_tokens_filter":      min_tokens,
            "auto_bin":               auto_bin,
        }
        for m in all_metrics
    ]
    summary_path = Path(output_dir) / "multi_context_summary.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
