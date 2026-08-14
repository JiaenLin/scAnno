"""Whose decision the exclusion is, and what happens when the object cannot prove it.

scAnno does not decide which nuclei are technical. Arming an exclusion automatically only stays
consistent with that if the trigger is a DECLARATION and never a column name, so the line these
tests hold is:

    an object with `cluster_FLAG` and NO declaration gets nothing
    an object with a declaration gets exactly what the declaration names, verified by digest
    an object whose flag no longer matches its declaration is REFUSED, not quietly used

The last one matters most. A column rewritten since upstream wrote it is not upstream's decision
any more, and withholding nuclei on it while citing upstream's provenance would attribute a
choice to a pipeline that did not make it.

    python tests/test_upstream.py
"""
from __future__ import annotations

import sys
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
    import pandas as pd
    import scipy.sparse as sp
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

from scanno import upstream as up  # noqa: E402
from scanno.exclude import flag_digest  # noqa: E402


def toy(n=60, flag=True, declare=True, na=0, schema="scqc/provenance@1"):
    rng = np.random.default_rng(1)
    A = ad.AnnData(X=sp.csr_matrix(rng.random((n, 8)).astype("float32")))
    A.obs_names = [f"c{i}" for i in range(n)]
    A.obs["cluster"] = pd.Categorical([str(i % 3) for i in range(n)])
    if flag:
        col = [bool(i % 6 == 0) for i in range(n)]
        for i in range(na):
            col[-(i + 1)] = pd.NA
        A.obs["cluster_FLAG"] = pd.array(col, dtype="boolean")
    if declare and flag:
        mask = up.as_mask_column(A, "cluster_FLAG")
        A.uns["scqc"] = {
            "schema": schema, "tool": "scQC", "flag_column": "cluster_FLAG",
            "n_obs": int(A.n_obs), "n_flagged": int(mask.sum()),
            "flag_digest": flag_digest(mask), "run_key": "abc123", "commit": "deadbeef",
            "flag_meaning": "step 6 flagged the cluster this nucleus sits in",
        }
    return A


print("\n1 - a declaration arms the exclusion, and names its authority")
A = toy()
d = up.decide(A)
check("armed", d.active and d.source == "scqc", f"source={d.source}")
check("on the declared column", d.column == "cluster_FLAG")
check("withholding exactly the flag", d.n == int(up.as_mask_column(A, "cluster_FLAG").sum()))
check("and says so out loud", any("ARMED" in ln for ln in d.lines), str(d.lines[:1]))
check("naming the digest it verified", any("digest" in ln for ln in d.lines))
check("and that scAnno did not choose them",
      any("did not choose" in ln for ln in d.lines))

print("\n2 - NO declaration means NOTHING, even with the column right there")
B = toy(declare=False)
d = up.decide(B)
check("cluster_FLAG is present", "cluster_FLAG" in B.obs)
check("and is NOT acted on", not d.active and d.source == "none")
check("the reason is stated, not silent",
      any("does not infer a flag from a column name" in ln for ln in d.lines), str(d.lines))

print("\n3 - an altered flag is REFUSED, not quietly used")
C = toy()
C.obs["cluster_FLAG"] = pd.array([True] * C.n_obs, dtype="boolean")
d = up.decide(C)
check("refused", bool(d.refuse) and not d.active)
check("the reason names the digest", "digest" in (d.refuse or ""))
check("and says whose decision it would have misattributed",
      "did not make" in (d.refuse or ""), (d.refuse or "")[:80])

print("\n4 - a subset object is REFUSED: the declaration is not for this population")
D = toy()
d = up.decide(D[:20].copy())
check("refused", bool(d.refuse))
check("...in those terms", "subset" in (d.refuse or ""), (d.refuse or "")[:90])

print("\n5 - an unknown schema is reported and not acted on")
E = toy(schema="scqc/provenance@99")
d = up.decide(E)
check("refused rather than guessed", bool(d.refuse))
check("naming both schemas", "@99" in (d.refuse or "") and "@1" in (d.refuse or ""))

print("\n6 - --no-exclude wins over the declaration, and says what it costs")
F = toy()
d = up.decide(F, disabled=True)
check("nothing is withheld", not d.active)
check("the declaration is still reported", any("declared" in ln for ln in d.lines), str(d.lines))
check("and the consequence is spelled out",
      any("rejected" in ln for ln in d.lines), str(d.lines))

print("\n7 - an explicit --exclude-flag wins over the declaration")
G = toy()
G.obs["mine"] = pd.array([bool(i < 5) for i in range(G.n_obs)], dtype="boolean")
d = up.decide(G, explicit="mine")
check("uses the column named", d.column == "mine" and d.source == "explicit")
check("withholding that column's cells", d.n == 5, str(d.n))
check("and notes the declaration named a different one",
      any("declared" in ln and "cluster_FLAG" in ln for ln in d.lines), str(d.lines))

print("\n8 - an explicit flag that does not exist is refused with the options")
d = up.decide(toy(), explicit="nope")
check("refused", bool(d.refuse))
check("and lists the boolean columns available", "cluster_FLAG" in (d.refuse or ""),
      (d.refuse or "")[:90])

print("\n9 - never-examined is NOT flagged: NA is False, and that is a decision")
H = toy(na=8)
mask = up.as_mask_column(H, "cluster_FLAG")
raw = pd.Series(H.obs["cluster_FLAG"])
check("the NA rows are not withheld", not mask[-8:].any(), str(mask[-8:]))
check("and they are genuinely NA in the object", int(raw.isna().sum()) == 8)
d = up.decide(H)
check("the run still arms and verifies", d.active and not d.refuse)

print("\n10 - a declaration with no flag column withholds nothing, and says so")
I = toy(flag=False, declare=False)
I.uns["scqc"] = {"schema": "scqc/provenance@1", "tool": "scQC", "flag_column": "",
                 "n_obs": int(I.n_obs), "n_flagged": -1, "flag_digest": ""}
d = up.decide(I)
check("nothing withheld", not d.active and not d.refuse)
check("and it is reported rather than silent",
      any("nothing is withheld" in ln for ln in d.lines), str(d.lines))

print("\n" + "=" * 64)
if fails:
    print(f"upstream: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("upstream OK - a declaration arms it, a column name never does, a mismatch refuses")
