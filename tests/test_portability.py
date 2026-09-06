#!/usr/bin/env python3
"""The leak guard: no site, host, project or cohort identifier anywhere in this repository.

WHY THIS TEST EXISTS

`--group-order`'s help once offered a study's own arm names as its example, and a comment in
`compare.py` read "Measured on <the study>". A tool built against one cohort and separated from
it in custody carries that cohort's names into every default it does not think about.

WHAT IS CHECKED

  1. every text file in the tree — package, tests, setup, jobs, skills, docs — against the
     SHAPES of site leakage (tests/_terms.py) and the TERMS in $SCANNO_FORBIDDEN_TERMS, a file
     outside the repository, so this guard never spells what it guards against
  2. the arguments that describe a DESIGN carry no such term in their help text

This file exempts only itself. Stdlib only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _terms import hits, terms  # noqa: E402

TEXT_EXT = {".py", ".R", ".sh", ".pbs", ".md", ".yml", ".yaml", ".toml", ".cfg", ".txt", ".json",
            ".csv", ".tsv", ".cff", ".ipynb"}
SKIP = {".git", "__pycache__", "_data", ".egg-info", "references"}
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def files():
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).parts
        if any(x in SKIP for x in rel):
            continue
        if p.is_file() and (p.suffix in TEXT_EXT or p.name in ("VERSION", "HEAD.txt")) \
                and p.resolve() != Path(__file__).resolve() and p.name != "_terms.py":
            yield p


site = terms()
print(f"\n1 - no site or cohort identifier anywhere in the tree ({len(site)} site term(s) "
      f"{'from ' + os.environ['SCANNO_FORBIDDEN_TERMS'] if site else 'supplied: none, set SCANNO_FORBIDDEN_TERMS to prove more'})")
leaks = []
for p in files():
    for i, why, line in hits(p.read_text(encoding="utf-8", errors="replace"), p.name):
        leaks.append(f"{p.relative_to(ROOT)}:{i} {why}: {line}")
for l in leaks[:20]:
    print("        LEAK " + l)
check("no leak in any text file", not leaks, f"{len(leaks)} leak(s)")

print("\n2 - the arguments that describe a DESIGN take the caller's own names")
cli = (ROOT / "scanno" / "cli.py").read_text(encoding="utf-8")
for flag in ("--group-order", "--factor", "--condition-key", "--sample-key", "--group-key"):
    i = cli.find(f'"{flag}"')
    check(f"{flag} is offered", i > 0)
    if i > 0:
        j = cli.find("s.add_argument(", i)
        blob = cli[i:j if j > i else i + 1200]
        h = hits(blob)
        check(f"{flag}'s help names no cohort", not h, "; ".join(w for _, w, _ in h[:3]))

print("")
if fails:
    print(f"FAIL: {len(fails)}")
    raise SystemExit(1)
print("PASS: no site or cohort identifier in the tree")
