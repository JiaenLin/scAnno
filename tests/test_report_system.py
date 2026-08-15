"""The report system: palette, primitives, depth generalisation, and the two documents.

Every test here is a defect that reached a rendered figure or a published page. They are written
as assertions on ARTISTS and DATA, not on pixels: a pixel diff tells you something moved and not
what, and it fails on a matplotlib upgrade for reasons that are nobody's fault.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanno.palette import BASE_HUES, Palette, SENTINELS, _to_hls, depth_of  # noqa: E402
from scanno.primitives import (GEOMETRY, NotDrawable, panel_grid, readable_on,  # noqa: E402
                               unique_ticks)

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# ------------------------------------------------------------------ palette
def test_palette():
    print("palette")
    paths = ["Cardiomyocyte/Working/Atrial", "Cardiomyocyte/Working/Ventricular",
             "Endothelial/Vascular", "Endothelial/Lymphatic",
             "Immune/Myeloid/Macrophage", "Immune/Lymphoid/T cell",
             "Stromal/Fibroblast", "UNRESOLVED", "EXCLUDED"]
    p = Palette(paths)
    check("depth is read from the paths", p.depth == 3, f"got {p.depth}")

    # rule 3: never None, for anything
    check("every lookup has a default",
          all(isinstance(p.of(x), str) and p.of(x).startswith("#")
              for x in ["", "Nope", None, "a/b/c/d/e/f"]))

    # rule 2: two different greys, neither shaded
    check("sentinels are two different greys",
          p.of("EXCLUDED") == SENTINELS["EXCLUDED"]
          and p.of("UNRESOLVED") == SENTINELS["UNRESOLVED"]
          and p.of("EXCLUDED") != p.of("UNRESOLVED"))
    check("a sentinel below the root is still a sentinel",
          Palette(["Immune/EXCLUDED"]).of("Immune/EXCLUDED") == SENTINELS["EXCLUDED"])

    # rule 1: hue is the ancestor's
    for kid in ("Immune/Myeloid", "Immune/Myeloid/Macrophage", "Immune/Lymphoid/T cell"):
        check(f"{kid} keeps its root's hue",
              abs(_to_hls(p.of(kid))[0] - _to_hls(p.of("Immune"))[0]) < 0.01,
              "tolerance is for 8-bit rounding (~0.001 of hue), not for drift")

    # rule 4: no collision across MIXED depths
    nodes = ["Cardiomyocyte", "Cardiomyocyte/Working", "Cardiomyocyte/Working/Atrial",
             "Cardiomyocyte/Working/Ventricular", "Immune", "Immune/Myeloid",
             "Immune/Myeloid/Macrophage", "Immune/Lymphoid", "Immune/Lymphoid/T cell",
             "Endothelial", "Endothelial/Vascular", "Endothelial/Lymphatic",
             "Stromal", "Stromal/Fibroblast", "EXCLUDED", "UNRESOLVED"]
    check("no two labels that appear together share a colour", not p.collisions(nodes),
          str(p.collisions(nodes)[:2]))

    # the two recorded failures of the earlier shading rules
    ls = [_to_hls(p.of(n))[1] for n in nodes if n not in SENTINELS]
    check("nothing is driven to black or white", min(ls) >= 0.29 and max(ls) <= 0.83,
          f"lightness {min(ls):.2f}-{max(ls):.2f}")
    import itertools
    deep = ["/".join(("R",) + c) for c in itertools.product(*[["a", "b"]] * 5)]
    q = Palette(deep)
    allnodes = sorted({"/".join(l.split("/")[:i + 1])
                       for l in deep for i in range(len(l.split("/")))})
    check("depth 6 does not collide", not q.collisions(allnodes),
          f"{len(allnodes)} nodes, {len(q.collisions(allnodes))} clashes")

    # base hues are distinguishable from one another
    worst = min((abs(_to_hls(a)[0] - _to_hls(b)[0]), a, b)
                for a, b in itertools.combinations(BASE_HUES[:8], 2))
    check("the first eight base hues are separated", worst[0] > 0.02,
          f"closest pair {worst[1]} / {worst[2]}")

    # pinning
    pin = Palette(paths, pinned={"Immune": "#7B3FA0"})
    check("a pin wins", pin.of("Immune") == "#7B3FA0")
    check("pinning a lineage recolours its subtree",
          abs(_to_hls(pin.of("Immune/Myeloid/Macrophage"))[0]
              - _to_hls("#7B3FA0")[0]) < 0.01)
    check("a pinned leaf is exactly the pin",
          Palette(paths, pinned={"Immune/Myeloid/Macrophage": "#abcdef"})
          .of("Immune/Myeloid/Macrophage") == "#abcdef")
    check("Palette.load rejects a non-colour",
          _raises(lambda: Palette.load(_tmp_json({"A": "red"}))))


def _raises(fn):
    try:
        fn()
    except Exception:                                                     # noqa: BLE001
        return True
    return False


def _tmp_json(obj):
    import tempfile
    p = Path(tempfile.mkstemp(suffix=".json")[1])
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


# ------------------------------------------------------------------ primitives
def test_primitives():
    print("primitives")
    # the zip()-truncation class of bug: a grid must never be smaller than the series
    for n in (1, 2, 5, 8, 12, 13):
        fig, axes = panel_grid(n)
        check(f"panel_grid({n}) returns exactly {n} axes", len(axes) == n, f"got {len(axes)}")
        import matplotlib.pyplot as plt
        plt.close(fig)

    check("panel_grid refuses zero panels", _raises(lambda: panel_grid(0)))

    # segment labels must be legible on their own segment
    check("label colour flips on a pale segment", readable_on("#f0e0d0") == "#1a1a1a")
    check("label colour flips on a dark segment", readable_on("#20303f") == "#ffffff")

    # a leaf name that is not unique must not collapse two populations into one tick
    t = unique_ticks(["Immune/Myeloid/Mac", "Stromal/Other/Mac", "Immune/Lymphoid/T"])
    check("ambiguous leaves are disambiguated", t[0] != t[1] and t[2] == "T", str(t))

    # saving geometry is part of the figure, and two figures deliberately have no tight box
    check("F133 and F134 save without a tight bounding box",
          GEOMETRY["F133"][1] is None and GEOMETRY["F134"][1] is None)
    check("the sweep and level figures save at 130 dpi",
          GEOMETRY["F100"][0] == 130 and GEOMETRY["F101"][0] == 130)

    # showfliers has no default: it is the only difference between three identical boxplots
    import inspect
    from scanno.primitives import quartile_boxes
    sig = inspect.signature(quartile_boxes)
    check("showfliers is required", sig.parameters["showfliers"].default is inspect._empty)
    from scanno.primitives import umap_scatter, stacked_rows
    check("denominator is required",
          inspect.signature(umap_scatter).parameters["denominator"].default is inspect._empty)
    check("label_floor is required",
          inspect.signature(stacked_rows).parameters["label_floor"].default is inspect._empty)


# ------------------------------------------------------------------ one colour authority
def test_one_colour_rule():
    print("one colour authority")
    src = {p.name: p.read_text(encoding="utf-8")
           for p in (Path(__file__).resolve().parents[1] / "scanno").glob("*.py")}
    import re
    offenders = []
    for name, text in src.items():
        if name in ("palette.py",):
            continue
        for m in re.finditer(r'^[A-Z_]*(PALETTE|HUES|COLOURS|COLORS)\s*=\s*[\[{]', text, re.M):
            offenders.append(f"{name}:{m.group(0)[:30]}")
        if re.search(r'^def shade\(', text, re.M):
            offenders.append(f"{name}: defines its own shade()")
    check("no module outside palette.py defines a hue list or a shade rule",
          not offenders, "; ".join(offenders[:3]))


# ------------------------------------------------------------------ depth generalisation
def test_depth_general():
    print("depth generalisation")
    src = (Path(__file__).resolve().parents[1] / "scanno")
    import re
    bad = []
    for f in ("figures.py", "context.py", "document.py"):
        text = (src / f).read_text(encoding="utf-8")
        # `"/" in label` silently matches level 3 as well as level 2
        # Code, not prose: these module docstrings NAME the patterns in order to forbid them,
        # so a lint that reads the docstring reports the rule as its own violation.
        code = re.sub(r'"""..*?"""', "", text, flags=re.S)
        for m in re.finditer(r'\b(if|elif|and|or|not)\s+[^\n]*"/"\s+in\s+\w+', code):
            bad.append(f"{f}: {m.group(0)[:40]}")
        for m in re.finditer(r'\.split\("/"\)\[0\]', code):
            bad.append(f"{f}: split('/')[0]")
    check("no module detects a level with a separator test", not bad, "; ".join(bad[:3]))
    check("depth_of ignores sentinels",
          depth_of(["A/B/C", "EXCLUDED", "UNRESOLVED"]) == 3)

    from scanno.figures import FIGURES
    for fid in ("F103", "F135", "F136", "F143", "F134"):
        import inspect
        check(f"{fid} takes a depth",
              "depth" in inspect.signature(FIGURES[fid][0]).parameters)


# ------------------------------------------------------------------ named absence
def test_named_absence():
    print("named absence")
    from scanno import figures as F
    check("NotDrawable exists and carries a reason",
          issubclass(F.NotDrawable, Exception))
    doc = (Path(__file__).resolve().parents[1] / "scanno" / "document.py").read_text("utf-8")
    check("absent SECTIONS are named, not omitted", "_absent_section(" in doc)
    check("absent FIGURES are named, not omitted", "class='absent'" in doc)
    check("every figure id has a legend",
          all(fid in doc for fid in F.FIGURES), "")


# ------------------------------------------------------------------ viewer fitness
def test_viewer_audit():
    print("viewer audit")
    from scanno.emit import audit_file, SCRATCH_UNS_PREFIXES
    check("audit_file is importable", callable(audit_file))
    check("scratch uns prefixes are declared", "_tmp" in SCRATCH_UNS_PREFIXES)
    cli = (Path(__file__).resolve().parents[1] / "scanno" / "cli.py").read_text("utf-8")
    check("the lab subcommand is registered", 'sub.add_parser("lab"' in cli)
    check("the lab subcommand can fix in place", '"--fix"' in cli)


if __name__ == "__main__":
    for t in (test_palette, test_primitives, test_one_colour_rule, test_depth_general,
              test_named_absence, test_viewer_audit):
        t()
    print("")
    if FAIL:
        print(f"{len(FAIL)} FAILED: {', '.join(FAIL)}")
        raise SystemExit(1)
    print("all report-system checks passed")
