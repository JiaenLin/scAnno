#!/usr/bin/env python3
"""Every subcommand's `--help` runs, and no help text names the unit as cells alone.

THE DEFECT (single-cell-harness gap GAP-C30). `scanno report --help` died with
`ValueError: incomplete format`. argparse expands `%` in every help string, and two of `report`'s
said "report ~100%" and "100% agreement". KNOWN_ISSUES.md sends readers to `--help`, so the page
that explains the cohort report could not be opened. Nothing ran `--help`, so nothing noticed.

WHAT THIS CHECKS, through the real CLI, in a subprocess, as a user's shell runs it:
  A  `scanno --help` lists the subcommands, and each one's `--help` exits 0 with its own usage
     line. The list is READ FROM THE PARSER, so a subcommand added later is checked without
     editing this file.
  B  help is printed before any input is read, so no assay is known when it is. No help text may
     name the unit as "cell(s)" alone (gap GAP-C19): "cells or nuclei" names both, and "cell type"
     is not a unit. tests/_words.py says which phrases are not units.

WHAT IT CANNOT CHECK: that the help is right about what a flag does. That is the code's, and the
suites that exercise each command hold it.

    python tests/test_help.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# By file, not through `tests`: an environment can carry a regular package of that name.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _words import cells_said  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def cli(*args):
    r = subprocess.run([sys.executable, "-m", "scanno.cli", *args], cwd=str(ROOT),
                       capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT), "PATH": "", "MPLBACKEND": "Agg"})
    return r.returncode, r.stdout or "", r.stderr or ""


print("\nA - every subcommand's --help exits 0")
rc, out, err = cli("--help")
check("`scanno --help` exits 0", rc == 0, f"rc={rc} {err.strip().splitlines()[-1:]}")
m = re.search(r"\{([a-z0-9,_-]+)\}", out)
cmds = m.group(1).split(",") if m else []
check("it lists the subcommands", len(cmds) >= 10, f"{len(cmds)} listed")
helps = {"": out}
for c in cmds:
    rc, out, err = cli(c, "--help")
    said = (err.strip().splitlines() or [""])[-1]
    check(f"`scanno {c} --help` exits 0 with its usage line",
          rc == 0 and f"usage: scanno {c}" in out, f"rc={rc} {said}")
    helps[c] = out

print("\nB - no help text names the unit as cells alone: no assay is known when help is printed")
for c, text in helps.items():
    hits = cells_said(text)
    check(f"`scanno {c} --help`".replace("scanno  ", "scanno ") + " says cells or nuclei, never cells alone",
          not hits, f"{len(hits)}: " + " | ".join(hits[:3]))

print("\n" + "=" * 64)
if fails:
    print(f"help: {len(fails)} FAILED - " + "; ".join(fails[:8]))
    raise SystemExit(1)
print(f"help: {len(cmds)} subcommands, every --help runs and names no unit as cells alone")
