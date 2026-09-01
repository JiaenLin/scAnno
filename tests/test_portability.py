"""The package names no study, and carries no study's values as its examples.

scAnno ships publicly and is developed against one cohort at a time, which is exactly the
condition under which a project's vocabulary leaks into a tool: an example in a help string, a
default that happens to suit the data in front of you, a comment that names the study it was
measured on. None of it fails a test, none of it breaks a run, and all of it tells the next
user that this tool was built for somebody else.

TWO LEAKS THIS FILE WAS WRITTEN AFTER, both found by grep and neither by any suite:

  * `--group-order`'s help offered `young_chow young_HFD aged_chow aged_HFD` as its example -
    one study's arms, in the help text of a general tool.
  * a comment in `compare.py` read "Measured on SAMBO".

The list below is a RATCHET, in the sense scProfile uses: it may shrink when a term stops being
a risk, and it may never grow to accommodate a new leak. A term that appears here appears
because it once shipped.

    python tests/test_portability.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


#: Study names, cohort identifiers, sample names and file names belonging to one project.
#: Case-insensitive, matched as whole words so ordinary English is not caught.
BANNED = [
    r"sambo", r"aging_?hfd", r"young_?hfd", r"aging[0-9]", r"young[0-9]",
    r"aged_chow", r"young_chow", r"cellmarker3_mouse", r"mouse_heart_tree",
    r"jiaen", r"wangyb", r"100,?713", r"109,?140",
]

print("\n1 - no module names a study, a cohort or one project's files")
mods = sorted(p for p in (ROOT / "scanno").glob("*.py"))
check("there are modules to check", len(mods) > 10, str(len(mods)))
for m in mods:
    src = m.read_text(encoding="utf-8")
    hits = sorted({w for w in BANNED if re.search(rf"\b{w}\b", src, re.I)})
    check(f"{m.name} names no project", not hits, ", ".join(hits))

print("\n2 - nor does any shipped skill or the README")
for f in sorted((ROOT / "skills").rglob("*.md")) + [ROOT / "README.md"]:
    src = f.read_text(encoding="utf-8")
    hits = sorted({w for w in BANNED if re.search(rf"\b{w}\b", src, re.I)})
    check(f"{f.relative_to(ROOT)} names no project", not hits, ", ".join(hits))

print("\n3 - the arguments that describe a DESIGN take the caller's own names")
cli = (ROOT / "scanno" / "cli.py").read_text(encoding="utf-8")
for flag in ("--group-order", "--factor", "--condition-key", "--sample-key", "--group-key"):
    i = cli.find(f'"{flag}"')
    check(f"{flag} is offered", i > 0)
    if i > 0:
        # its help runs to the next add_argument
        j = cli.find("s.add_argument(", i)
        blob = cli[i:j if j > i else i + 1200]
        hits = sorted({w for w in BANNED if re.search(rf"\b{w}\b", blob, re.I)})
        check(f"{flag}'s help carries no study's levels", not hits, ", ".join(hits))

print("\n4 - species and tissue are the caller's, never defaulted to one")
for cmd in ("annotate", "background", "panel"):
    i = cli.find(f'sub.add_parser("{cmd}"')
    j = cli.find("set_defaults", i)
    blob = cli[i:j] if i > 0 and j > i else ""
    if not blob:
        continue
    m = re.search(r'"--species"[^)]*default=("(?!")[^"]*")', blob)
    check(f"{cmd} does not default --species to a value", not m, m.group(1) if m else "")
    m = re.search(r'"--tissue"[^)]*default=("(?!")[^"]*")', blob)
    check(f"{cmd} does not default --tissue to a value", not m, m.group(1) if m else "")

print("\n" + "=" * 64)
if fails:
    print(f"portability: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("portability OK - the package names no study and defaults to none")
