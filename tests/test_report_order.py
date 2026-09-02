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
# The rescued column carries its OWN unique label, for the same reason the joint one does: if
# the rescued block is a neighbouring figure reused under a rescued heading, that label cannot
# appear in it. It also differs from the joint column, so the two cannot be confused for each
# other either.
RESCUED = ["Immune/Myeloid/Macrophage"] * 37 + ["Adipose/OnlyInTheRescuedColumn"] * 3 \
          + ["Stromal/Fibroblast"] * 20


class Fake:
    def __init__(self, name):
        self.n_obs = N
        self.obs = pd.DataFrame({
            "sample": ["S1"] * 30 + ["S2"] * 30,
            "group": ["g1"] * 30 + ["g2"] * 30,
            "cell_type": PATH, "cell_type_forced": FORCED,
            "my_joint_col": JOINT, "my_rescue_col": RESCUED,
            "cell_compartment": L1, "cell_compartment_forced": L1,
            "cell_type_gap": ["not a number"] * N,
            "total_counts": np.arange(N, dtype=float),
        }, index=[f"c{i}" for i in range(N)])
        self.var = pd.DataFrame(index=["G1", "G2"])
        self.obsm, self.layers, self.X = {}, {}, None
        self.var_names = self.var.index


ctx = Context([("S", Fake("S"))], path_key="cell_type", sample_key="sample",
              group_key="group", forced_key="cell_type_forced", l1_key="cell_compartment",
              forced_l1_key="cell_compartment_forced", joint_route_key="my_joint_col",
              rescue_key="my_rescue_col")

print("\n1 - the context reads the joint column as its own")
check("it is detected", ctx.has_joint_route)
rows = {r["label"]: r["nuclei"] for r in (ctx.joint_route_rows() or [])}
check("its rows come from that column, not the forced one",
      rows.get("Neural/OnlyInTheJointColumn") == 5, str(rows))
frows = {r["label"]: r["nuclei"] for r in (ctx.forced_scope_rows() or [])}
check("and the forced rows do NOT contain it",
      "Neural/OnlyInTheJointColumn" not in frows, str(frows))
check("a context given no joint column simply has none",
      not Context([("S", Fake("S"))], path_key="cell_type").has_joint_route)

print("\n1b - and the RESCUED column the same way")
check("it is detected", ctx.has_rescue)
rrows = {r["label"]: r["nuclei"] for r in (ctx.rescue_rows() or [])}
check("its rows come from that column, not the forced or joint one",
      rrows.get("Adipose/OnlyInTheRescuedColumn") == 3, str(rrows))
check("and neither neighbour contains it",
      "Adipose/OnlyInTheRescuedColumn" not in frows
      and "Adipose/OnlyInTheRescuedColumn" not in rows)
check("nor does the rescued block carry the joint column's unique label",
      "Neural/OnlyInTheJointColumn" not in rrows, str(rrows))
check("a context given no rescued column simply has none",
      not Context([("S", Fake("S"))], path_key="cell_type").has_rescue)

print("\n1c - the sweep is found whichever walk named it")
# `--resolve` renames the whole per-resolution family, and the reader knows the sweep by
# `path_key`, which names the HONEST column. A run that forced every rung therefore wrote
# `<...>_resolved_path_r*` and the reader looked for `<...>_path_r*`, found nothing, and
# reported an object that had never been swept. The second stem is tried ONLY when the first
# finds nothing, so this can turn an absence into a figure and cannot change one that draws.


class Swept(Fake):
    def __init__(self, name, resolved):
        super().__init__(name)
        stem = "cell_type_forced" if resolved else "cell_type"
        for tag in ("1p0", "2p0"):
            self.obs[f"{stem}_r{tag}"] = FORCED if resolved else PATH


_honest = Context([("S", Swept("S", resolved=False))], path_key="cell_type",
                  forced_key="cell_type_forced")
_forced = Context([("S", Swept("S", resolved=True))], path_key="cell_type",
                  forced_key="cell_type_forced")
check("an honest sweep is found under path_key, as always",
      [c for _r, c, _t in _honest.sweep_keys("S")] == ["cell_type_r1p0", "cell_type_r2p0"],
      str(_honest.sweep_keys("S")))
check("a FORCED sweep is found under forced_key",
      [c for _r, c, _t in _forced.sweep_keys("S")]
      == ["cell_type_forced_r1p0", "cell_type_forced_r2p0"], str(_forced.sweep_keys("S")))
check("an object with neither family returns nothing",
      Context([("S", Fake("S"))], path_key="cell_type",
              forced_key="cell_type_forced").sweep_keys("S") == [])
check("and the honest result does not depend on forced_key being set",
      Context([("S", Swept("S", resolved=False))], path_key="cell_type").sweep_keys("S")
      == _honest.sweep_keys("S"))

print("\n2 - the delivered annotation leads and L1 follows")
with tempfile.TemporaryDirectory() as td:
    write_cohort(ctx, Path(td), title="T", version="0")
    html = (Path(td) / "reports" / "cohort.html").read_text(encoding="utf-8")
heads = [re.sub(r"<[^>]+>", "", m).strip()
         for m in re.findall(r"<h3>(.*?)</h3>", html, re.S)]
comp = [h for h in heads if "annotation" in h or "JOINT ROUTE" in h]
check("the five blocks are present", len(comp) >= 5, str(comp))
check("the RESCUED block is one of them",
      any("RESCUED" in h for h in comp), str(comp))
check("and it names the column it was drawn from",
      "my_rescue_col" in html, "the block must print its own obs column")
check("its unique label reaches the page",
      "Adipose/OnlyInTheRescuedColumn" in html)
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
for key in ("cell_type", "cell_type_forced", "my_joint_col", "cell_compartment"):
    check(f"the page names {key!r}", f"<code>{key}</code>" in html)
check("the joint block's own label reaches the page",
      "OnlyInTheJointColumn" in html)

print("\n4 - a report with no joint column does not invent the block")
ctx2 = Context([("S", Fake("S"))], path_key="cell_type", sample_key="sample",
               group_key="group", forced_key="cell_type_forced", l1_key="cell_compartment")
with tempfile.TemporaryDirectory() as td:
    write_cohort(ctx2, Path(td), title="T", version="0")
    html2 = (Path(td) / "reports" / "cohort.html").read_text(encoding="utf-8")
check("no joint heading appears", "the JOINT ROUTE" not in html2)
check("nor its label", "OnlyInTheJointColumn" not in html2)
check("and the rest of the section still renders", "the scope annotation" in html2)

print("\n5 - a label column is never read as a statistic")
# `"cell_type".replace("_path", "_gap")` is `"cell_type"` - a no-op - so the LABEL column was
# read as every statistic and the run died with `could not convert string to float` from a line
# that looks like it is reading a number. The fixture also carries a `cell_type_gap` holding
# text: a name is not a type, and this package's naming makes the collision likely because a
# label column and its statistics share a stem by design.
check("building the context does not raise on a text 'statistic'", ctx.n == N)
for stat in ("depth", "gap", "support", "survival"):
    got = ctx.P[stat] if stat in ctx.P else None
    check(f"{stat!r} is absent rather than filled from a label",
          got is None or got.notna().sum() == 0, "" if got is None else str(got.head(2).tolist()))

print("\n6 - a figure's TITLE names the column it drew, not the block above it")
# The defect this whole file exists for, in its last hiding place. The figure registry held
# "composition, forced" as F106's description, so every F106 was titled "forced" whatever column
# it was given - and under the JOINT ROUTE heading the picture and the words beside it named
# different annotations, with nothing on the page to say so. The registry now holds the KIND and
# the caller supplies the subject.
blocks = re.split(r"<h3>(?=the )", html)
jr = [b for b in blocks if b.startswith("the JOINT ROUTE")]
fo = [b for b in blocks if b.startswith("the scope annotation, FORCED")]
check("the joint block exists", len(jr) == 1, str([b[:30] for b in blocks]))
check("the forced block exists", len(fo) == 1)
jr_titles = re.findall(r"<h3>(F\d+ · .*?)</h3>", jr[0] if jr else "", re.S)
fo_titles = re.findall(r"<h3>(F\d+ · .*?)</h3>", fo[0] if fo else "", re.S)
check("the joint block's figures are titled", bool(jr_titles), str(jr_titles))
check("and every one of them NAMES the joint column",
      all("my_joint_col" in t for t in jr_titles), str(jr_titles))
check("none of them says 'forced'",
      not any("forced" in t.lower() for t in jr_titles), str(jr_titles))
check("while the forced block's figures DO name the forced column",
      all("cell_type_forced" in t for t in fo_titles), str(fo_titles))
check("so the two blocks' figure titles are not identical",
      set(jr_titles) != set(fo_titles))

print("\n7 - EVERY composition figure names its column, and no two share a title")
# The residual after 6: three of five blocks named their column in the figure title and two did
# not, and inside every block the by-group and by-sample figures were titled identically - two
# adjacent pictures of one kind, indistinguishable in the text and in the figure list.
seen = {}
for b in re.split(r"<h3>(?=the )", html):
    if not b.startswith("the "):
        continue
    col = re.search(r"drawn from <code>(.*?)</code>", b)
    if not col:
        continue
    name = re.sub(r"<[^>]+>", "", re.match(r"(.*?)</h3>", b, re.S).group(1)).strip()
    titles = [re.sub(r"<[^>]+>", "", t).strip()
              for t in re.findall(r"<h3>(F\d+ · .*?)</h3>", b, re.S)]
    # A BLOCK WITH NO FIGURES IS A FAILURE, NOT A SKIP. This read `if not titles: continue`,
    # so when six figures died with a TypeError and their blocks rendered empty, the check
    # walked past them and reported green. A check that passes when the thing it checks is
    # ABSENT is worse than no check: the silence reads as a result.
    check(f"{name}: has figures at all", bool(titles),
          "the block rendered with none - look for FAILED TO DRAW on the page")
    if not titles:
        continue
    check(f"{name}: every figure names {col.group(1)!r}",
          all(col.group(1) in t for t in titles), str(titles))
    check(f"{name}: no two figures share a title",
          len(set(titles)) == len(titles), str(titles))
    for t in titles:
        seen.setdefault(t, []).append(name)
dupes = {t: bs for t, bs in seen.items() if len(set(bs)) > 1}
check("and no title is reused across two different blocks", not dupes, str(dupes))
# A figure that RAISED renders as a defect block rather than an absence, and the two must never
# be confused - so the page is checked for it directly rather than inferred from a count.
check("no figure failed to draw", "FAILED TO DRAW" not in html,
      f'{html.count("FAILED TO DRAW")} figure(s) raised')

print("\n8 - an absent statistic names the columns it looked for")
# The message derived the names by `label_key.replace("_cell_type", "_gap")`, a no-op on a key
# holding neither substring - so on the object this report is actually run over it told the
# reader obs carried none of `cell_type`, `cell_type`, `cell_type` or `cell_type`. The context
# now records every name it searched and the message reads them back.
check("the context recorded what it searched for", bool(ctx.stat_keys_tried),
      str(sorted(ctx.stat_keys_tried)))
check("and never the label column itself",
      "cell_type" not in ctx.stat_keys_tried and "cell_compartment" not in ctx.stat_keys_tried,
      str(sorted(ctx.stat_keys_tried)))
check("the page names them", all(f"`{k}`" in html for k in sorted(ctx.stat_keys_tried))
      or "no per-call statistic" not in html, str(sorted(ctx.stat_keys_tried)))
check("and says WHY a label-only object has none",
      "no per-call statistic" not in html or "drop-list" in html)

print("\n" + "=" * 64)
if fails:
    print(f"report order: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print("report order OK - the delivered annotation leads, and every block names its column")
