"""The composition section: what leads, and which column each block was drawn from.

Two defects this exists for, both found by reading a real report:

  1. FOUR BLOCKS THAT NAMED NO COLUMN. Every block was described in prose - "the scope
     annotation", "FORCED" - and none said which `obs` column it came from. A report built over
     a JOINT route's own columns and one built over a per-sample route's are then identical on
     the page, and a reader believes a figure shows one annotation while it shows another.
  2. THE ANSWER CAME THIRD. L1 led the section, on the reasoning that a reader meets the
     compartments before the subtypes. But L1 is a CHECK on the delivered call, not a coarser
     view of it, and nothing downstream consumes it - so the annotation the run exists to
     produce appeared below two blocks of context.

    python tests/test_report_order.py
"""
from __future__ import annotations

import re
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
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"SKIP: needs {e.name}")
    raise SystemExit(0)

from scanno.context import Context  # noqa: E402
from scanno.document import write_cohort  # noqa: E402

N = 60
# Four columns that DIFFER, so a block drawn from the wrong one is visible in the output. The
# joint column carries a label no other column has: if the joint block is the forced figure
# reused under a joint heading, that label cannot appear.
PATH = ["Immune/Myeloid/Macrophage"] * 30 + ["UNRESOLVED"] * 10 + ["Stromal/Fibroblast"] * 20
FORCED = ["Immune/Myeloid/Macrophage"] * 40 + ["Stromal/Fibroblast"] * 20
JOINT = ["Immune/Myeloid/Macrophage"] * 35 + ["Neural/OnlyInTheJointColumn"] * 5 \
        + ["Stromal/Fibroblast"] * 20
L1 = ["Immune"] * 40 + ["Stromal"] * 20


class Fake:
    def __init__(self, name):
        self.n_obs = N
        self.obs = pd.DataFrame({
            "sample": ["S1"] * 30 + ["S2"] * 30,
            "group": ["g1"] * 30 + ["g2"] * 30,
            "scanno_path": PATH, "scanno_forced": FORCED,
            "my_joint_col": JOINT, "scanno_l1": L1, "scanno_forced_l1": L1,
            "total_counts": np.arange(N, dtype=float),
        }, index=[f"c{i}" for i in range(N)])
        self.var = pd.DataFrame(index=["G1", "G2"])
        self.obsm, self.layers, self.X = {}, {}, None
        self.var_names = self.var.index


ctx = Context([("S", Fake("S"))], path_key="scanno_path", sample_key="sample",
              group_key="group", forced_key="scanno_forced", l1_key="scanno_l1",
              forced_l1_key="scanno_forced_l1", joint_route_key="my_joint_col")

print("\n1 - the context reads the joint column as its own")
check("it is detected", ctx.has_joint_route)
rows = {r["label"]: r["nuclei"] for r in (ctx.joint_route_rows() or [])}
check("its rows come from that column, not the forced one",
      rows.get("Neural/OnlyInTheJointColumn") == 5, str(rows))
frows = {r["label"]: r["nuclei"] for r in (ctx.forced_scope_rows() or [])}
check("and the forced rows do NOT contain it",
      "Neural/OnlyInTheJointColumn" not in frows, str(frows))
check("a context given no joint column simply has none",
      not Context([("S", Fake("S"))], path_key="scanno_path").has_joint_route)

print("\n2 - the delivered annotation leads and L1 follows")
with tempfile.TemporaryDirectory() as td:
    write_cohort(ctx, Path(td), title="T", version="0")
    html = (Path(td) / "reports" / "cohort.html").read_text(encoding="utf-8")
heads = [re.sub(r"<[^>]+>", "", m).strip()
         for m in re.findall(r"<h3>(.*?)</h3>", html, re.S)]
comp = [h for h in heads if "annotation" in h or "JOINT ROUTE" in h]
check("the four blocks are present", len(comp) >= 4, str(comp))
order = {h: i for i, h in enumerate(comp)}
check("the scope annotation comes before L1",
      order.get("the scope annotation", 99) < order.get("the L1 annotation", -1), str(comp))
check("the FORCED scope comes before L1",
      order.get("the scope annotation, FORCED", 99) < order.get("the L1 annotation", -1))
check("the joint route comes before L1",
      order.get("the JOINT ROUTE", 99) < order.get("the L1 annotation", -1))
check("and the joint route comes after the forced column it corrects",
      order.get("the JOINT ROUTE", -1) > order.get("the scope annotation, FORCED", 99))

print("\n3 - every block NAMES the obs column it was drawn from")
for key in ("scanno_path", "scanno_forced", "my_joint_col", "scanno_l1"):
    check(f"the page names {key!r}", f"<code>{key}</code>" in html)
check("the joint block's own label reaches the page",
      "OnlyInTheJointColumn" in html)

print("\n4 - a report with no joint column does not invent the block")
ctx2 = Context([("S", Fake("S"))], path_key="scanno_path", sample_key="sample",
               group_key="group", forced_key="scanno_forced", l1_key="scanno_l1")
with tempfile.TemporaryDirectory() as td:
    write_cohort(ctx2, Path(td), title="T", version="0")
    html2 = (Path(td) / "reports" / "cohort.html").read_text(encoding="utf-8")
check("no joint heading appears", "the JOINT ROUTE" not in html2)
check("nor its label", "OnlyInTheJointColumn" not in html2)
check("and the rest of the section still renders", "the scope annotation" in html2)

print("\n" + "=" * 64)
if fails:
    print(f"report order: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("report order OK - the delivered annotation leads, and every block names its column")
