#!/usr/bin/env python3
"""Dump the REGRESSION-group answers of the base model and of `adapters/correct`.

Not part of the shipped lab. It exists because of where this run's fine-tune actually
loses. Rubric 3.4 asks for at least two qualitative cases where the fine-tune is worse
than baseline (b) -- and on the target group there are none: 48 wins, 2 ties, 0 losses.
Every loss this run has is in the regression group, which NB5 scores but never writes
out per item, so the report would have had to describe catastrophic forgetting without
being able to show one line of it.

Additive only: it reads the frozen eval file, re-uses `labkit.generate` with the same
greedy settings and the same `system=None` call NB2/NB5 make for this group, and writes
a new artifact. No eval item, no prompt, and no score already recorded is touched.

    python scripts/dump_regression_examples.py   ->  results/regression_examples.json
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labkit import evaluate as ev, generate, report  # noqa: E402
from labkit.config import get_tier  # noqa: E402


def main() -> int:
    tier = get_tier()
    rows = [json.loads(l) for l in
            (ROOT / "data" / "eval_regression.jsonl").open(encoding="utf-8") if l.strip()]
    prompts = [r["instruction"] for r in rows]

    # Same call NB2 and NB5 make for this group: no system prompt, 96 new tokens, greedy.
    model, tok = generate.load_base(tier)
    base_preds, _ = generate.generate_batch(model, tok, prompts, system=None,
                                            max_new_tokens=96, label="base/regression")
    del model
    generate.free_memory()

    from peft import PeftModel
    model, tok = generate.load_base(tier)
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters" / "correct"))
    model.eval()
    ft_preds, _ = generate.generate_batch(model, tok, prompts, system=None,
                                          max_new_tokens=96, label="ft/regression")
    del model
    generate.free_memory()

    out = []
    for i, (r, bp, fp) in enumerate(zip(rows, base_preds, ft_preds)):
        out.append({
            "i": i,
            "instruction": r["instruction"],
            "keywords": r["keywords"],
            "base_pred": " ".join(bp.split()),
            "ft_pred": " ".join(fp.split()),
            "base_score": round(ev.keyword_recall(bp, r["keywords"]), 4),
            "ft_score": round(ev.keyword_recall(fp, r["keywords"]), 4),
        })

    report.write_json(out, "regression_examples.json", results_dir=ROOT / "results")
    lost = [r for r in out if r["ft_score"] < r["base_score"]]
    print(f"{len(out)} items · fine-tune worse on {len(lost)} of them")
    print(f"base mean {sum(r['base_score'] for r in out) / len(out):.4f}  "
          f"ft mean {sum(r['ft_score'] for r in out) / len(out):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
