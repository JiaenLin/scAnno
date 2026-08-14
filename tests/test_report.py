"""The report: one self-contained file, every number in a payload beside it, and its own limits.

Three properties are asserted because each has a failure mode that looks fine on the page:

  1. SELF-CONTAINED. A document that fetches a stylesheet or a font renders correctly on the
     machine that made it and is a broken page everywhere else, five years from now most of all.
     So: no `src=` or `href=` that is not a data URI.
  2. EVERY SECTION STATES A LIMIT, and the report counts a missing one as a defect on its own
     front page. A section with no limit reads as a section with nothing to qualify, which is
     the more confident claim and the wrong one.
  3. A FIGURE THAT CANNOT BE DRAWN IS A NAMED ABSENCE. A blank space reads as "there was
     nothing to show"; the name and the reason read as what they are.

    python tests/test_report.py
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


try:
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

from scanno import report as rp  # noqa: E402
from scanno import upstream as up  # noqa: E402
from scanno.emit import annotate_obs  # noqa: E402
from scanno.exclude import flag_digest  # noqa: E402

LABELS = ["Cardiomyocyte", "Endothelial", "Fibroblast"]


def calls():
    return [{"cluster": i, "label": v, "path": f"root/{v}", "depth": 2, "gap": 0.4 + i / 20,
             "survival": 0.8, "cover": 0.9, "excluded": False, "trace": []}
            for i, v in enumerate(LABELS)]


def toy(n=90, embedding=True, sample=True, flag=True):
    rng = np.random.default_rng(2)
    A = ad.AnnData(X=sp.csr_matrix((rng.random((n, 12)) * 4).astype("float32")))
    A.obs_names = [f"c{i}" for i in range(n)]
    A.var_names = [f"G{i}" for i in range(12)]
    y = np.array([i % 3 for i in range(n)])
    A.obs["cluster"] = pd.Categorical([str(v) for v in y])
    if sample:
        A.obs["sample"] = pd.Categorical(["s1"] * (n // 2) + ["s2"] * (n - n // 2))
        A.obs["condition"] = pd.Categorical(["ctrl"] * (n // 2) + ["hfd"] * (n - n // 2))
    if embedding:
        A.obsm["X_umap"] = rng.normal(size=(n, 2)).astype("float32")
    mask = None
    if flag:
        mask = np.array([i % 15 == 0 for i in range(n)], dtype=bool)
        A.obs["cluster_FLAG"] = pd.array(mask, dtype="boolean")
        A.uns["scqc"] = {"schema": "scqc/provenance@1", "tool": "scQC",
                         "flag_column": "cluster_FLAG", "n_obs": n,
                         "n_flagged": int(mask.sum()), "flag_digest": flag_digest(mask),
                         "run_key": "rk", "commit": "cm",
                         "flag_meaning": "step 6 flagged the cluster"}
    annotate_obs(A, calls(), y, flag=mask, support={v: 30 + i for i, v in enumerate(LABELS)})
    return A, y


def make(**kw):
    A, y = toy(**kw)
    dec = up.decide(A)
    doc = rp.collect(A, calls(), LABELS, y, label_key="scanno_cell_type", decision=dec,
                     support={v: 30 + i for i, v in enumerate(LABELS)},
                     cluster_key="cluster", sample_key="sample" if kw.get("sample", True) else None,
                     condition_key="condition" if kw.get("sample", True) else None,
                     species="Mouse", tissue="Heart", weights="corpus", background="store",
                     version="0.3.1")
    figs = rp.draw(doc, A, "scanno_cell_type")
    return A, doc, figs


print("\n1 - the payload carries the numbers the page shows")
A, doc, figs = make()
check("schema declared", doc["schema"] == rp.SCHEMA, doc["schema"])
check("headline counts the object", doc["headline"]["n_cells"] == A.n_obs)
tot = sum(e["n"] for e in doc["composition_l1"])
check("composition accounts for every nucleus", tot == A.n_obs, f"{tot} vs {A.n_obs}")
check("EXCLUDED appears as its own level",
      any(e["label"] == "EXCLUDED" for e in doc["composition_l1"]))
check("reliability is broken down by depth", len(doc["reliability"]) >= 1)

print("\n2 - the exclusion is attributed, not just counted")
ex = doc["exclusion"]
check("source recorded", ex["source"] == "scqc", ex["source"])
check("declared_by recorded", ex["declared_by"] == "scQC")
check("digest carried into the report", len(ex["digest"]) == 16, ex["digest"])
check("per-sample breakdown present", len(ex["per_sample"]) == 2)
check("per-condition rates present", "per_condition" in ex and len(ex["per_condition"]) == 2)

print("\n3 - the document is self-contained")
html, payload = rp.build(doc, figs)
external = re.findall(r'src="(?!data:)[^"]+"|href="https?://[^"]+"', html)
check("no external src or href", not external, str(external[:3]))
check("it is a complete document", html.lstrip().startswith("<!doctype html"))
check("styles are inline", "<style>" in html)
check("figures are data URIs", "data:image/png;base64," in html)

print("\n4 - every section states what it cannot show")
n_cannot = html.count('class="cannot"')
check("at least one limit per major section", n_cannot >= 5, f"{n_cannot} limit blocks")
check("and labels are explicitly NOT validated",
      "shows that a label is CORRECT" in html or "label is CORRECT" in html)
check("the payload records zero defects on a complete run", payload["defects"] == [],
      str(payload["defects"]))

print("\n5 - a figure that cannot be drawn is a NAMED absence, never a gap")
A2, doc2, figs2 = make(embedding=False)
html2, payload2 = rp.build(doc2, figs2)
check("A3 is absent", figs2["A3"]["uri"] is None)
check("and says why", "embedding" in (figs2["A3"]["absent"] or ""), figs2["A3"]["absent"])
check("and names what would produce it", "UMAP" in (figs2["A3"]["absent"] or ""))
check("the page shows the absence rather than nothing", "is not drawn" in html2)
check("an explained absence is not a defect", payload2["defects"] == [], str(payload2["defects"]))

print("\n6 - an unexplained absence IS a defect on the front page")
_, p3 = rp.build(doc, {"A9": {"uri": None, "absent": None}})
check("counted", any("A9" in d for d in p3["defects"]), str(p3["defects"]))

print("\n7 - it degrades without a sample column instead of refusing")
A4, doc4, figs4 = make(sample=False)
html4, _ = rp.build(doc4, figs4)
check("still builds", len(html4) > 2000)
check("per-sample section is simply empty", doc4["per_sample"] == [])
check("and the composition figure still draws",
      figs4["A1"]["uri"] is not None or figs4["A1"]["absent"])

print("\n8 - nothing withheld reads as nothing withheld")
A5, y5 = toy(flag=False)
doc5 = rp.collect(A5, calls(), LABELS, y5, label_key="scanno_cell_type", decision=None,
                  cluster_key="cluster", species="Mouse", tissue="Heart")
html5, _ = rp.build(doc5, rp.draw(doc5, A5, "scanno_cell_type"))
check("exclusion is None", doc5["exclusion"] is None)
check("and the page says so plainly", "Nothing was withheld" in html5)

print("\n9 - write() produces both files")
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    h, j = rp.write(Path(tmp) / "sub" / "annotation.html", doc, figs)
    check("html written", Path(h).exists() and Path(h).stat().st_size > 5000)
    check("json written beside it", Path(j).exists() and Path(j).name == "annotation.json")
    import json
    back = json.loads(Path(j).read_text(encoding="utf-8"))
    check("and it round-trips", back["headline"]["n_cells"] == A.n_obs)

print("\n10 - marker panels come from the corpus that was scored on")
panels = rp.marker_panels({"cardiomyocyte cell": {"TNNT2": 8.0, "MYH6": 4.0, "X": 1.0}},
                          {"Cardiomyocyte": ["cardiomyocyte"]}, ["Cardiomyocyte"], top=2)
check("matched by the node's own patterns", "Cardiomyocyte" in panels)
check("best-cited first", panels["Cardiomyocyte"] == ["TNNT2", "MYH6"], str(panels))
check("a node with no pattern yields nothing",
      rp.marker_panels({}, {}, ["Nope"]) == {})

print("\n" + "=" * 64)
if fails:
    print(f"report: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("report OK - self-contained, limits stated, absences named")
