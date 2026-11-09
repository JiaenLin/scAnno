"""The documentation's commands run, as written, on the repository's synthetic cohort.

WHY (single-cell-harness gap GAP-031). Once `--assay` became required, every `annotate` example
in README.md and docs/QUICKSTART.md failed. The README also asked `cluster --split-by` to write a
single `--out`, named a cluster column (`leiden`) that `cluster` never writes, gave `--report` no
path, and passed `--tree` to `rescue`, which has no such option. A reader copying the README got
an argparse error at the first annotation. Nothing ran the examples, so nothing noticed.

WHAT THIS DOES. Each fenced `bash` block of each document is run in bash, in document order, in
one working directory per document, so a later block reads what an earlier one wrote, as a reader
would. `scanno` on the PATH is this checkout. The reader brings three files, and the test builds
them from tests/_cohort.py:
    README.md            qc.h5ad (the QC'd cohort), corpus.db, tree.json
    docs/QUICKSTART.md   sample.h5ad (one sample),   corpus.db, tree.json (its own section 3)
Every other file a command reads must be written by an earlier command of the same document.

WHAT IT DOES NOT RUN, and says so line by line: commands of other programs. `git clone` and
`pip install` need the network. `cd` would leave the working directory. `alias`, and `sch` with
it, belong to the harness, whose suite checks its own commands.

WHAT IT CANNOT SHOW: that an example's output reads as the document says. The synthetic
populations are separated by construction. A green run shows the commands are valid and compose.
It does not show that the labels would be right on real data.

Section D pins each statement GAP-031 found false to the code, as corrected.

    python tests/test_docs_run.py
"""
from __future__ import annotations

import atexit
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


try:
    import anndata as ad
    import numpy as np
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

from tests._cohort import TREE, write_cohort, write_corpus, write_sample, write_tree  # noqa: E402

#: First words of lines this test does not run. Each is printed when it is skipped.
NOT_RUN = {"git": "needs the network", "pip": "needs the network",
           "cd": "would leave the working directory", "alias": "the harness's",
           "sch": "the harness's command; its own suite checks it"}
FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.M | re.S)


def blocks(doc):
    return [(m.group(1), m.group(2)) for m in FENCE.finditer(doc)]


def logical_lines(script):
    """Backslash continuations joined, so one command is one line."""
    out, cur = [], ""
    for ln in script.splitlines():
        if ln.rstrip().endswith("\\"):
            cur += ln.rstrip()[:-1] + " "
            continue
        out.append(cur + ln)
        cur = ""
    if cur:
        out.append(cur)
    return out


def first_word(line):
    s = line.strip()
    if not s or s.startswith("#"):
        return ""
    try:
        return shlex.split(s, comments=True)[0] if shlex.split(s, comments=True) else ""
    except ValueError:
        return s.split()[0]


def run_document(name, brings):
    """Run every bash block of one document.

    Returns (work dir, commands run, lines skipped, one (label, ok, detail) per block). It checks
    nothing itself, so the two documents can run at once and still report in order."""
    text = (ROOT / name).read_text(encoding="utf-8")
    work = Path(tempfile.mkdtemp(prefix="scanno_docs_"))
    # Removed at exit, on a failure too: one run of the README writes the whole cohort chain,
    # and a suite run on a compute node would otherwise leave it in that node's /tmp.
    atexit.register(shutil.rmtree, work, True)
    shim = work / ".bin"
    shim.mkdir()
    (shim / "scanno").write_text(f'#!/bin/sh\nexec "{sys.executable}" -m scanno.cli "$@"\n')
    (shim / "scanno").chmod(0o755)
    for fname, build in brings.items():
        build(work / fname)
    # /usr/sbin for the system tools libraries probe (joblib asks sysctl for the core count).
    # Every cache goes into the working directory, which is removed at exit. Without
    # NUMBA_CACHE_DIR, umap's compiled functions cache beside their installed source - inside
    # whatever environment runs this suite (LAYOUT 3f: point the tool's scratch at the run).
    cache = work / ".cache"
    env = {"PATH": f"{shim}:/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg",
           "HOME": str(work), "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
           "XDG_CACHE_HOME": str(cache), "NUMBA_CACHE_DIR": str(cache / "numba"),
           "MPLCONFIGDIR": str(cache / "mpl"), "PYTHONNOUSERSITE": "1"}
    n_run, skipped, results = 0, [], []
    for i, (lang, body) in enumerate(blocks(text)):
        if lang != "bash":
            continue
        keep = []
        for ln in logical_lines(body):
            w = first_word(ln)
            if w in NOT_RUN:
                skipped.append(f"{w}: {ln.strip()[:60]}   ({NOT_RUN[w]})")
                continue
            keep.append(ln)
        script = "\n".join(keep)
        n_cmd = sum(1 for ln in keep if re.search(r"(^|\s|;|do)\s*scanno\s", " " + ln))
        if not n_cmd:
            continue
        import time
        t0 = time.monotonic()
        r = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=str(work), env=env,
                           capture_output=True, text=True, timeout=900)
        took = time.monotonic() - t0
        said = (r.stdout or "") + (r.stderr or "")
        refusal = [ln for ln in said.splitlines() if "REFUSE" in ln or "error:" in ln
                   or "ERROR" in ln][-2:]
        results.append((f"{name} block {i}: {n_cmd} scanno command(s) exit 0", r.returncode == 0,
                        f"{took:.0f}s" if r.returncode == 0
                        else f"rc={r.returncode} {refusal or said.strip().splitlines()[-3:]}"))
        n_run += n_cmd
    return work, n_run, skipped, results


print("\nA - docs/QUICKSTART.md, from nothing to labelled clusters")
qs = (ROOT / "docs" / "QUICKSTART.md").read_text(encoding="utf-8")
tree_blocks = [json.loads(b) for lang, b in blocks(qs) if lang == "json"]
check("QUICKSTART declares a tree in a json block", bool(tree_blocks))
check("and it is the tree tests/_cohort.py builds the cohort for",
      bool(tree_blocks) and tree_blocks[0] == TREE)


def quickstart_tree(p):
    p.write_text(json.dumps(tree_blocks[0] if tree_blocks else TREE), encoding="utf-8")


# The two documents share nothing, so they run at once: each command is its own process and the
# wall clock is mostly the analysis stack being imported again by every one of them.
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

with ThreadPoolExecutor(max_workers=2) as pool:
    qs_job = pool.submit(run_document, "docs/QUICKSTART.md",
                         {"sample.h5ad": write_sample, "corpus.db": write_corpus,
                          "tree.json": quickstart_tree})
    rd_job = pool.submit(run_document, "README.md",
                         {"qc.h5ad": write_cohort, "corpus.db": write_corpus,
                          "tree.json": write_tree})
    qs_work, n_qs, qs_skipped, qs_results = qs_job.result()
    rd_work, n_rd, rd_skipped, rd_results = rd_job.result()
for r in qs_results:
    check(*r)
check("QUICKSTART: at least four scanno commands ran", n_qs >= 4, str(n_qs))
for s in qs_skipped:
    print(f"        not run  {s}")

print("\nB - README.md, every route it shows")
for r in rd_results:
    check(*r)
check("README: at least twelve scanno commands ran", n_rd >= 12, str(n_rd))
for s in rd_skipped:
    print(f"        not run  {s}")

print("\nC - what the documents say a run leaves behind, read from what it left")
ann = qs_work / "annotated.h5ad"
if ann.exists():
    A = ad.read_h5ad(ann)
    check("QUICKSTART: the object carries the label per cell (scanno_cell_type)",
          "scanno_cell_type" in A.obs)
    check("QUICKSTART: var is re-keyed by symbol, the accession kept in var['gene_id']",
          str(A.var_names[0]).startswith("CD") and "gene_id" in A.var
          and str(A.var["gene_id"].iloc[0]).startswith("ENSG"),
          f"var_names[0]={A.var_names[0]!r}")
    check("QUICKSTART: obsm comes out as it went in", set(A.obsm) >= {"X_pca", "X_umap"},
          str(sorted(A.obsm)))
else:
    check("QUICKSTART: annotated.h5ad was written", False)
check("QUICKSTART: --out wrote the per-cluster table", (qs_work / "labels.tsv").exists())
for f in ("report.html", "report.json", "delivered.h5ad", "joint_route.html", "rescue.json",
          "impact.csv"):
    hit = list(rd_work.rglob(f))
    check(f"README: {f} was written", bool(hit))

print("\nD - the statements GAP-031 found false, pinned to the code as corrected")
from scanno.classify import GAP_CORPUS, classify  # noqa: E402
from scanno.corpus import node_weights  # noqa: E402

genes = np.array([f"G{i}" for i in range(30)])
asr = {"alpha one cell": {f"G{i}": 8.0 for i in range(0, 10)},
       "alpha two cell": {f"G{i}": 8.0 for i in range(10, 20)},
       "beta cell": {f"G{i}": 8.0 for i in range(20, 30)}}
tree = {"children": {"root": ["Alpha", "Beta"], "Alpha": ["Alpha one", "Alpha two"]},
        "patterns": {"Alpha": ["alpha"], "Beta": ["beta"], "Alpha one": ["alpha one"],
                     "Alpha two": ["alpha two"]},
        "members": {}, "genes": genes}
usable = np.ones(len(genes), dtype=bool)
Z = np.zeros((1, len(genes)))
Z[0, :20] = 3.0                              # clearly Alpha; Alpha one and Alpha two tied
row = classify(Z, usable, tree, assertions=asr)[0]
check("README: a call that stops above a leaf stops at its parent", row["path"] == "Alpha",
      row["path"])
check("README: its gap is the DECLINED step's, below the bar",
      row["gap"] == row["trace"][-1]["gap"] and row["gap"] < GAP_CORPUS
      and row["trace"][0]["gap"] >= GAP_CORPUS,
      f"gap={row['gap']:.3f}, root step {row['trace'][0]['gap']:.3f}")

pats = {"Alpha one": ["one", "alpha one"], "Alpha two": ["alpha two", "two"]}
W1 = node_weights(asr, pats, genes, usable)[0]
W2 = node_weights(asr, {k: list(reversed(v)) for k, v in pats.items()}, genes, usable)[0]
check("QUICKSTART: the order of a node's patterns does not matter",
      W1 is not None and W2 is not None and np.allclose(W1, W2))
upper = node_weights(asr, {"Alpha one": ["Alpha one"], "Alpha two": ["Alpha two"]}, genes, usable)
check("QUICKSTART: a pattern is matched against the lower-cased name, so upper case matches nothing",
      upper[0] is None)

print("\n" + "=" * 64)
if fails:
    print(f"docs: {len(fails)} FAILED - " + "; ".join(fails[:10]))
    raise SystemExit(1)
print(f"docs: {n_qs + n_rd} scanno commands from README and QUICKSTART ran as written")
