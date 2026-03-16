"""
analyze_failures.py
====================
Analyze the 13 retrieval failure cases and explain WHY each failed.

Usage:
    python analyze_failures.py
"""

import json
from pathlib import Path
from collections import Counter

failures = json.loads(Path("results/retrieval_failures.json").read_text(encoding="utf-8"))

print(f"\n{'='*70}")
print(f"  RETRIEVAL FAILURE ANALYSIS — {len(failures)} questions failed Answer@5")
print(f"{'='*70}\n")

# ── Per failure details ───────────────────────────────────────────────────────
for i, f in enumerate(failures, 1):
    print(f"[{i:02d}] {f['question_id']} | {f['document_id']}")
    print(f"  Q : {f['question']}")
    print(f"  A : {f['gold_answer'][:120]}")
    print(f"  Failed    : {f['failed_strategies']}")
    print(f"  Passed    : {f['passed_strategies']}")
    print(f"  Top-1 per strategy:")
    for st, result in f['top1_per_strategy'].items():
        print(f"    {st:<28} → {result[:70]}")
    print()

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"{'='*70}")
print("  FAILURE PATTERNS")
print(f"{'='*70}")

# Which strategies failed most
all_failed = []
for f in failures:
    all_failed.extend(f['failed_strategies'])
strat_counts = Counter(all_failed)
print("\n  Failures per strategy:")
for s, c in sorted(strat_counts.items(), key=lambda x: -x[1]):
    print(f"    {s:<28} : {c} failures")

# Which documents failed most
doc_counts = Counter(f['document_id'] for f in failures)
print("\n  Failures per document:")
for d, c in sorted(doc_counts.items(), key=lambda x: -x[1]):
    print(f"    {d:<40} : {c} failures")

# All-strategy failures vs partial
all_fail = [f for f in failures if not f['passed_strategies']]
partial  = [f for f in failures if f['passed_strategies']]
print(f"\n  Failed in ALL strategies : {len(all_fail)}")
print(f"  Failed in SOME strategies: {len(partial)}")

# Answer length analysis
print("\n  Gold answer word count for failed questions:")
for f in failures:
    words = len(f['gold_answer'].split())
    print(f"    {f['question_id']} : {words} words — {f['gold_answer'][:80]}")

print()