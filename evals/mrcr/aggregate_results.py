"""
evals/mrcr/aggregate_results.py

Load multi-context MRCR result files and produce an aggregated summary.

Usage:
    # Scan directory for all *k.json files
    python -m evals.mrcr.aggregate_results

    # Explicit files
    python -m evals.mrcr.aggregate_results outputs/mrcr/64k.json outputs/mrcr/128k.json

    # Different output dir
    python -m evals.mrcr.aggregate_results --output-dir outputs/mrcr --save summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── Loading ───────────────────────────────────────────────────────────────────

def load_context_file(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _extract_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle both formats: {metrics: {...}} and flat dict."""
    return data.get("metrics", data)


# ─── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(paths: List[Path]) -> List[Dict[str, Any]]:
    """Load result files and return a summary list sorted by context size."""
    rows = []
    for p in paths:
        data = load_context_file(p)
        m = _extract_metrics(data)

        ctx_size = m.get("context_size") or data.get("context_size")
        label    = m.get("context_label") or data.get("context_label") or p.stem

        rows.append({
            "context":                ctx_size,
            "context_label":          label,
            "accuracy":               m.get("accuracy"),
            "exact_accuracy":         m.get("exact_accuracy"),
            "avg_score":              m.get("avg_score"),
            "accuracy_truncated":     m.get("accuracy_truncated"),
            "accuracy_non_truncated": m.get("accuracy_non_truncated"),
            "truncation_rate":        m.get("truncation_rate"),
            "skip_rate":              m.get("skip_rate"),
            "avg_coverage":           m.get("avg_coverage"),
            "avg_original_tokens":    m.get("avg_original_tokens"),
            "avg_used_tokens":        m.get("avg_used_tokens"),
            "num_truncated":          m.get("num_truncated"),
            "num_skipped":            m.get("num_skipped"),
            "total":                  m.get("total"),
            "source_file":            str(p),
        })

    return sorted(rows, key=lambda r: r["context"] or 0)


# ─── Display ───────────────────────────────────────────────────────────────────

def print_table(rows: List[Dict[str, Any]]) -> None:
    def _pct(v) -> str:
        return f"{v:.1%}" if v is not None else "  N/A"

    def _flt(v) -> str:
        return f"{v:.3f}" if v is not None else "   N/A"

    header = (
        f"\n{'Context':<10} {'Accuracy':>10} {'Exact':>8} "
        f"{'AccTrunc':>10} {'AccNoTr':>9} {'TruncRate':>10} {'Coverage':>10} "
        f"{'Skipped':>8} {'Total':>6}"
    )
    sep = "-" * len(header.lstrip("\n"))
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['context_label'] or r['context']:<10}"
            f" {_pct(r['accuracy']):>10}"
            f" {_pct(r['exact_accuracy']):>8}"
            f" {_pct(r['accuracy_truncated']):>10}"
            f" {_pct(r['accuracy_non_truncated']):>9}"
            f" {_pct(r['truncation_rate']):>10}"
            f" {_flt(r['avg_coverage']):>10}"
            f" {str(r['num_skipped'] or 0):>8}"
            f" {str(r['total'] or 0):>6}"
        )
    print()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate multi-context MRCR results into a summary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("files", nargs="*",
                   help="Specific result JSON files (e.g. 64k.json 128k.json). "
                        "If omitted, scans --output-dir for *k.json files.")
    p.add_argument("--output-dir", "-o", default="outputs/mrcr",
                   help="Directory to scan when no files are given")
    p.add_argument("--save", "-s", default=None,
                   help="Path to save aggregated summary JSON "
                        "(default: {output-dir}/aggregated_summary.json)")
    args = p.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        out = Path(args.output_dir)
        paths = sorted(out.glob("*k.json"))
        if not paths:
            print(f"No *k.json files found in {out}")
            return

    print(f"Loading {len(paths)} file(s): {[p.name for p in paths]}")
    rows = aggregate(paths)
    print_table(rows)

    save_path = Path(args.save) if args.save else (
        Path(args.output_dir) / "aggregated_summary.json"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
