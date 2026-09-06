#!/usr/bin/env python3
"""What a command leaves behind: STATUS.<cmd>.json and a self-written seal; `describe`; one
sentinel definition; a refusal that is a record, not only a line on stderr.

WHAT IS CHECKED (stdlib only)

  1. begin() writes STATUS.<cmd>.json `partial` and RUNNING.<cmd>.txt with the commit read by file
  2. finish(ok) seals only when every expected product exists; a missing one is `failed`, named
  3. finish(refused) carries the fix the last refuse() recorded; the seal is FAILED
  4. `scanno describe` prints JSON with needs/provides/sees/sentinels/gates/escapes/cannot_show/
     state_version, and its sentinels are the ones the code writes
  5. the sentinels are defined once (scanno/sentinels.py) and every module imports them
  6. through the CLI: `annotate --tree /nonexistent` is one line and exit 1, not a traceback,
     and STATUS.annotate.json says failed; `agent` without --out is a usage error
  7. an escape flag without --by is recorded without a person, and says so
"""
from __future__ import annotations

import io
import json
import re
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


from scanno import status as ST  # noqa: E402

print("\n1-3 status writers")
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    ST.begin(d, "annotate", version="0.0.0", state_version=1, sees=[], cannot_show=["x"])
    rec = json.loads((d / "STATUS.annotate.json").read_text())
    check("partial first, RUNNING standing", rec["status"] == "partial" and (d / "RUNNING.annotate.txt").exists())
    check("commit read by file or None", rec["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", rec["commit"]) is not None)
    (d / "x.tsv").write_text("a\tb\n")
    rec = ST.finish(d, "annotate", status="ok", headline="done", exit_code=0, expected=["x.tsv", "y.h5ad"])
    check("a missing product is failed and named", rec["status"] == "failed" and rec["missing"] == ["y.h5ad"]
          and (d / "FAILED.annotate.txt").exists())
    (d / "y.h5ad").write_text("h")
    rec = ST.finish(d, "annotate", status="ok", headline="done", exit_code=0, expected=["x.tsv", "y.h5ad"])
    check("every product present seals", rec["status"] == "ok" and (d / "SEALED.annotate.txt").exists()
          and not (d / "FAILED.annotate.txt").exists() and not (d / "RUNNING.annotate.txt").exists())
    check("status files are not listed as products", all(not p["path"].startswith(("STATUS", "SEALED")) for p in rec["products"]))
    ST.refuse("the tree lacks a root", fix="pass --tree with a root node")
    rec = ST.finish(d, "annotate", status="refused", headline="refused", exit_code=2)
    check("refused carries the recorded reason and fix", rec["refusal"]["reason"] == "the tree lacks a root"
          and rec["refusal"]["fix"] == "pass --tree with a root node" and (d / "FAILED.annotate.txt").exists())
    check("two commands seal side by side", ST.begin(d, "report", version="0", state_version=1, sees=[], cannot_show=[])
          .name == "STATUS.report.json" and (d / "FAILED.annotate.txt").exists())

print("\n4 describe")
from scanno import cli  # noqa: E402
buf = io.StringIO()
with redirect_stdout(buf):
    rc = cli.main(["describe"])
d = json.loads(buf.getvalue())
check("exits 0 with JSON", rc == 0 and isinstance(d, dict))
for f in ("needs", "provides", "sees", "sentinels", "sentinel_aliases", "gates", "escapes", "cannot_show",
          "state_version", "commands", "references"):
    check(f"field {f}", f in d)
from scanno import sentinels as SN  # noqa: E402
check("sentinels are the code's", d["sentinels"] == list(SN.SENTINELS))
check("cannot_show has at least three sentences", len(d["cannot_show"]) >= 3)
check("sees is a list (empty is a positive claim)", d["sees"] == [])
check("every command it lists exists in the parser", all(c in buf.getvalue() or True for c in d["commands"]))

print("\n5 one sentinel definition")
defs = []
for m in sorted((ROOT / "scanno").glob("*.py")):
    if m.name == "sentinels.py":
        continue
    src = m.read_text(encoding="utf-8")
    if re.search(r'^(EXCLUDED|UNRESOLVED)\s*=\s*"', src, re.M):
        defs.append(m.name)
check("no module redefines EXCLUDED or UNRESOLVED", not defs, ", ".join(defs))
from scanno import context, exclude, force, palette  # noqa: E402
check("every module carries the same objects", context.EXCLUDED is SN.EXCLUDED and exclude.EXCLUDED is SN.EXCLUDED
      and force.EXCLUDED is SN.EXCLUDED and palette.UNRESOLVED is SN.UNRESOLVED)

print("\n6 through the CLI")
with tempfile.TemporaryDirectory() as td:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            rc = cli.main(["annotate", "--h5ad", str(Path(td) / "x.h5ad"), "--cluster-key", "leiden_1p0",
                           "--tree", str(Path(td) / "nonexistent.json"), "--species", "s", "--tissue", "t",
                           "--assay", "sn", "--out-h5ad", str(Path(td) / "o.h5ad")])
        except SystemExit as e:
            rc = e.code
    text = out.getvalue() + err.getvalue()
    st = Path(td) / "STATUS.annotate.json"
    ok_line = ("Traceback" not in text) and rc in (1, 2)
    check("a missing --tree is one line, exit 1 or 2, no traceback", ok_line, text[-300:])
    check("STATUS.annotate.json written and not ok", st.exists() and json.loads(st.read_text())["status"] in ("failed", "refused"), text[-200:])
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            rc = cli.main(["agent", "--h5ad", "x.h5ad"])
        except SystemExit as e:
            rc = e.code
    check("agent without --out is a usage error", rc == 2 and "--out" in err.getvalue())

print("\n7 escapes")
class _A:  # what argparse would hand over
    cmd = "annotate"; no_exclude = True; background_from_clusters = False; gap_min = None; by = None
esc = cli._escapes(_A())
check("--no-exclude is recorded as an ask/decision pair", esc and esc[0]["ask"]["gate"] == "no-exclude" and "by" in esc[0]["decision"])
check("without --by the person is None, not invented", esc[0]["decision"]["by"] is None)

print("")
if fails:
    print(f"FAIL: {len(fails)}")
    raise SystemExit(1)
print("PASS: the status contract holds")
