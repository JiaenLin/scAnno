"""The reading playbook: the notebook is generated from the markdown, every code cell parses, and
neither names a cohort.

WHY A TEST FOR A DOCUMENT

Two copies of the same instructions drift, and the drift is invisible - the notebook still runs
and still makes figures while teaching a parameter the markdown no longer recommends. So the
notebook is GENERATED and this asserts it is current.

WHAT IT NO LONGER DOES (single-cell-harness B1-039). It pinned about fifty sentences of the
markdown - a string count, an exact list of numbers, a phrase per warning. A test of wording fails
when a sentence is reworded and the document is still right, and passes when a sentence survives
beside code that contradicts it. The document's truth is the docs audit's, and the code a reader
copies out of it is tested by section B: it has to parse.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "READING_THE_OUTPUT.md"
NB = ROOT / "docs" / "reading_the_output.ipynb"
GEN = ROOT / "docs" / "make_notebook.py"

fails = []


def check(n, c, d=""):
    print(f"  {'ok  ' if c else 'FAIL'}  {n}" + (f"   {d}" if d else ""))
    if not c:
        fails.append(n)


print("A. the files exist and the notebook is GENERATED, not maintained")
check("the playbook exists", MD.exists())
check("the notebook exists", NB.exists())
check("the generator exists", GEN.exists())
r = subprocess.run([sys.executable, str(GEN), "--check"], capture_output=True, text=True)
check("the notebook is IN SYNC with the markdown", r.returncode == 0,
      (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "")

nb = json.loads(NB.read_text(encoding="utf-8"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
check("the notebook says it is generated",
      "GENERATED FILE" in json.dumps(nb["metadata"]))

print("\nB. every code cell is valid python")
bad = []
for i, c in enumerate(code):
    try:
        ast.parse("".join(c["source"]))
    except SyntaxError as e:
        bad.append(f"cell {i}: {e}")
check(f"all {len(code)} code cells parse", not bad, "; ".join(bad[:3]))
check("there are enough of them to be a playbook", len(code) >= 25, str(len(code)))

md = MD.read_text(encoding="utf-8")

print("\nC. no cohort is named — this ships in a public repo")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _terms import hits as _hits  # noqa: E402
_h = _hits(md)
check("no site or cohort identifier in the playbook", not _h, "; ".join(f"{i}: {w}" for i, w, _ in _h[:4]))

print("")
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}")
    raise SystemExit(1)
print("playbook OK - generated from the markdown, every code cell parses, no cohort named")
