#!/usr/bin/env python3
"""The same input gives the same labels, and every step's wall time is in the run's record.

THE GAP (single-cell-harness gap GAP-030, goal ANNO.2: "ultra-fast, consistent" annotation). Neither
half was measured. No test annotated anything twice, and STATUS.<cmd>.json held a start and a finish
stamp to the second and no duration, so "fast" had no number behind it and "consistent" had no test.

WHAT THIS CHECKS, through the real CLI, every command its own process:
  A  the synthetic cohort (tests/_cohort.py) is clustered, given one gene background, annotated at
     two resolutions with --resolve, and rescued - TWICE, in two directories. Every obs column of
     every annotated and rescued object is identical between the two runs, by value (NaN equals
     NaN), and so are the per-cluster tables, impact.csv and rescue.json.
  B  each command's STATUS.<cmd>.json carries `timings`: the command's wall time, how many
     observations it read, and its steps in order, each with its own wall time. The steps cannot
     add up to more than the whole, and annotate names its walk and its write.

WHAT IT CANNOT SHOW: that another machine gives the same labels, or how fast a real cohort runs. A
reproduction on the cluster measures the first. The second is what the record now carries on every
run, which is the point of writing it there rather than into a benchmark.

    python tests/test_consistency.py
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


try:
    import anndata as ad
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

# By file, not through `tests`: an environment can carry a regular package of that name.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cohort import SAMPLES, SPECIES, TISSUE, write_corpus, write_sample, write_tree  # noqa: E402


def cli(work, *args):
    """The real CLI in its own process, with every cache inside `work`."""
    cache = Path(work) / ".cache"
    env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "MPLBACKEND": "Agg",
           "HOME": str(work), "XDG_CACHE_HOME": str(cache), "NUMBA_CACHE_DIR": str(cache / "numba"),
           "MPLCONFIGDIR": str(cache / "mpl"), "PYTHONNOUSERSITE": "1"}
    r = subprocess.run([sys.executable, "-m", "scanno.cli", *map(str, args)], cwd=str(work),
                       capture_output=True, text=True, env=env, timeout=900)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def chain(inputs, db, tree, run):
    """One whole run: cluster, background, annotate every sample, rescue. Returns [(step, rc, tail)]."""
    run = Path(run)
    said = []

    def step(name, *args):
        rc, out = cli(run.parent, *args)
        said.append((name, rc, out.strip().splitlines()[-2:]))
        return rc

    step("cluster", "cluster", "--h5ad", *inputs, "--out-dir", run / "clustered",
         "--resolutions", "0.5,1.0", "--seed", "0")
    clustered = sorted((run / "clustered").glob("*_clustered.h5ad"))
    step("background", "background", "--h5ad", *clustered, "--cluster-key", "leiden_1p0",
         "--out", run / "store.npz", "--species", SPECIES, "--tissue", TISSUE, "--assay", "scrna")
    for f in clustered:
        s = f.name.split("_")[0]
        step(f"annotate {s}", "annotate", "--h5ad", f, "--cluster-key", "leiden_1p0",
             "--cluster-key", "leiden_0p5", "--resolve", "--tree", tree, "--db", db,
             "--store", run / "store.npz", "--species", SPECIES, "--tissue", TISSUE,
             "--assay", "scrna", "--out", run / "tables" / f"{s}.tsv",
             "--out-h5ad", run / "annotated" / f"{s}.h5ad", "--status-dir", run / "status" / s)
    step("rescue", "rescue", "--h5ad", *sorted((run / "annotated").glob("*.h5ad")),
         "--label-key", "scanno_resolved_path", "--unit-key", "sample",
         "--sweep-prefix", "scanno_resolved_path_r", "--cluster-prefix", "leiden_",
         "--out-dir", run / "rescued", "--out", run / "rescue" / "rescue.json",
         "--out-table", run / "rescue" / "impact.csv")
    return said


def same_column(a, b):
    """(equal, how many differ). By value: strings exactly, numbers exactly with NaN equal to NaN."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    if len(a) != len(b):
        return False, max(len(a), len(b))
    num = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) \
        and not pd.api.types.is_bool_dtype(a) and not pd.api.types.is_bool_dtype(b)
    if num:
        x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
        diff = ~((x == y) | (np.isnan(x) & np.isnan(y)))
    else:
        diff = a.astype(str).to_numpy() != b.astype(str).to_numpy()
    n = int(diff.sum())
    return n == 0, n


work = Path(tempfile.mkdtemp(prefix="scanno_consistency_"))
try:
    inputs = [write_sample(work / "in" / f"{s}.h5ad", sample=s) for s in SAMPLES]
    db, tree = write_corpus(work / "corpus.db"), write_tree(work / "tree.json")
    runs = [work / "run1", work / "run2"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        said = list(pool.map(lambda r: chain(inputs, db, tree, r), runs))

    print("\nA - the same input, run twice from the start, gives the same labels")
    for r, steps in zip(runs, said):
        bad = [f"{n}: rc={rc} {tail}" for n, rc, tail in steps if rc != 0]
        check(f"{r.name}: every command exits 0 ({len(steps)} commands)", not bad, "; ".join(bad[:2]))
    for sub in ("annotated", "rescued"):
        one = sorted((runs[0] / sub).glob("*.h5ad"))
        two = {f.name: f for f in (runs[1] / sub).glob("*.h5ad")}
        check(f"{sub}: one object per sample in both runs",
              len(one) == len(SAMPLES) and sorted(two) == [f.name for f in one],
              f"{len(one)} and {len(two)}")
        n_cols = 0
        for f in one:
            if f.name not in two:
                continue
            A, B = ad.read_h5ad(f), ad.read_h5ad(two[f.name])
            check(f"{sub}/{f.name}: the same observations, in the same order",
                  list(A.obs_names) == list(B.obs_names))
            check(f"{sub}/{f.name}: the same obs columns",
                  list(A.obs.columns) == list(B.obs.columns),
                  f"only in run 1: {sorted(set(A.obs) - set(B.obs))}; "
                  f"only in run 2: {sorted(set(B.obs) - set(A.obs))}")
            differ = []
            for c in A.obs.columns:
                if c in B.obs:
                    ok, n = same_column(A.obs[c], B.obs[c])
                    n_cols += 1
                    if not ok:
                        differ.append(f"{c} ({n})")
            check(f"{sub}/{f.name}: every obs column identical by value ({len(A.obs.columns)})",
                  not differ, ", ".join(differ[:6]))
        check(f"{sub}: labels were compared at all", n_cols > 0 and any(
            str(c).startswith("scanno_") for c in ad.read_h5ad(one[0]).obs.columns) if one else False)
    for rel in [f"tables/{s}.tsv" for s in SAMPLES] + ["rescue/impact.csv"]:
        p, q = runs[0] / rel, runs[1] / rel
        check(f"{rel}: identical in both runs", p.exists() and q.exists()
              and p.read_bytes() == q.read_bytes())
    p, q = runs[0] / "rescue" / "rescue.json", runs[1] / "rescue" / "rescue.json"
    check("rescue/rescue.json: identical in both runs", p.exists() and q.exists()
          and json.loads(p.read_text()) == json.loads(q.read_text()))

    print("\nB - the run's record carries each step's wall time")
    recs = {"cluster": runs[0] / "clustered" / "STATUS.cluster.json",
            "rescue": runs[0] / "rescued" / "STATUS.rescue.json"}
    recs.update({f"annotate {s}": runs[0] / "status" / s / "STATUS.annotate.json" for s in SAMPLES})
    for name, p in recs.items():
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            check(f"{name}: STATUS was written", False, str(p.relative_to(work)))
            continue
        t = rec.get("timings")
        check(f"{name}: STATUS carries `timings`", isinstance(t, dict),
              f"keys: {', '.join(sorted(rec))[:120]}")
        if not isinstance(t, dict):
            continue
        wall, steps = t.get("wall_s"), t.get("steps") or []
        check(f"{name}: the command's wall time is a positive number of seconds",
              isinstance(wall, (int, float)) and math.isfinite(wall) and wall > 0, repr(wall))
        check(f"{name}: its steps are named, in order, each with a wall time",
              bool(steps) and all(isinstance(s.get("step"), str) and s["step"]
                                  and isinstance(s.get("wall_s"), (int, float))
                                  and s["wall_s"] >= 0 for s in steps),
              json.dumps(steps)[:160])
        total = sum(float(s.get("wall_s") or 0) for s in steps)
        check(f"{name}: the steps add up to no more than the whole ({total:.2f} <= {wall})",
              isinstance(wall, (int, float)) and total <= float(wall) + 1e-3)
        check(f"{name}: it says how many observations it read",
              isinstance(t.get("n_obs"), int) and t["n_obs"] > 0, repr(t.get("n_obs")))
        if name.startswith("annotate"):
            s = name.split()[1]
            n = ad.read_h5ad(runs[0] / "annotated" / f"{s}.h5ad", backed="r").n_obs
            check(f"{name}: n_obs is the object's", t.get("n_obs") == n, f"{t.get('n_obs')} vs {n}")
            names = [x.get("step") for x in steps]
            check(f"{name}: the walk and the write are steps of their own",
                  "walk" in names and "write" in names, ", ".join(map(str, names)))
finally:
    shutil.rmtree(work, ignore_errors=True)

print("\n" + "=" * 64)
if fails:
    print(f"consistency: {len(fails)} FAILED - " + "; ".join(fails[:8]))
    raise SystemExit(1)
print("consistency: two runs from one input give the same labels, and the record times every step")
