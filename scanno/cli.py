"""The `scanno` command.

Deliberately small. At 0.1.0 there is no driver and no report, so the CLI exposes what
actually exists rather than pretending to a pipeline: build a corpus panel, inspect a
store, run the adversarial suite.

Exit codes follow scQC: 0 pass or review, 1 error, 2 refusal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REFUSE = 2


def _selftest(_):
    import subprocess
    t = Path(__file__).resolve().parents[1] / "tests" / "test_adversarial.py"
    if not t.exists():
        print(f"scanno: no test suite at {t}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, str(t)])


def _panel(a):
    """Show the corpus panel for one context — what the untrained path would score on."""
    from .corpus import load_assertions
    try:
        asr = load_assertions(a.db, a.species, a.tissue, a.min_tier)
    except Exception as e:                                          # noqa: BLE001
        print(f"scanno: cannot read {a.db}: {e}", file=sys.stderr)
        return 1
    if not asr:
        print(f"scanno: REFUSE — no assertions for {a.species}/{a.tissue} at "
              f"tier<={a.min_tier}. The corpus cannot speak to this context.",
              file=sys.stderr)
        return REFUSE
    rows = sorted(((c, len(g)) for c, g in asr.items()), key=lambda r: -r[1])
    print(f"{a.species} / {a.tissue}   tier<={a.min_tier}   "
          f"{len(asr)} cell names, {sum(n for _, n in rows):,} (cell, gene) claims")
    for c, n in rows[:a.top]:
        best = sorted(asr[c].items(), key=lambda kv: -kv[1])[:8]
        print(f"  {c[:38]:<40}{n:>5}   {', '.join(g for g, _ in best)}")
    if len(rows) > a.top:
        print(f"  ... and {len(rows) - a.top} more cell names")
    return 0


def _store_info(a):
    import numpy as np
    d = np.load(a.store, allow_pickle=True)
    print(json.dumps({k: (v.tolist() if getattr(v, "ndim", 0) == 0 else f"array{v.shape}")
                      for k, v in d.items()}, indent=2)[:2000])
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="scanno", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="run the adversarial suite")
    s.set_defaults(fn=_selftest)

    s = sub.add_parser("panel", help="show the corpus panel for one species x tissue")
    s.add_argument("--db", required=True, type=Path)
    s.add_argument("--species", required=True)
    s.add_argument("--tissue", required=True)
    s.add_argument("--min-tier", type=int, default=4)
    s.add_argument("--top", type=int, default=20)
    s.set_defaults(fn=_panel)

    s = sub.add_parser("store-info", help="describe a saved profile store")
    s.add_argument("--store", required=True, type=Path)
    s.set_defaults(fn=_store_info)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
