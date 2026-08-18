"""The delivered cell-type tree with L1 integrated, as a section of the STANDARD report.

WHY THE FIRST TEST IS ABOUT MODULE IDENTITY RATHER THAN RENDERING

The same trap that swallowed the scope section swallows this one. `scanno/report.py` is real and
IS imported — `scanno annotate --report` calls `report.collect`, `scanno report --panels auto`
calls `report.panels_by_depth` — but the cohort document is assembled by
`scanno/document.py:write_all`, and `document.py` does not import `report.py`. A section written
there compiles, the suite passes, and it renders NOWHERE. So the wiring is asserted before
anything about the content.

WHY THE FIXTURE HAS TWO COHORTS

The failure a human does not notice is a section quoting a number that belongs to the cohort it
was written for. So every structural claim is re-run on `OTHER`, a three-sample cohort with
unrelated node names, a taxonomy of a different depth, and a compartment the first cohort has no
counterpart for.

WHAT THE SECTION MUST NOT DO, tested here because each was a real defect elsewhere in this file's
neighbourhood:

  - report agreement between a column and itself (the DERIVED L1 is the deep path truncated, so
    it agrees by construction and measures nothing)
  - render a terminal label with no account of WHY it stops, or read `sealed` as `leaf`
  - assume three levels, ten samples, or any node name
"""
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import pytest
except ImportError:                                                        # noqa: BLE001
    class _Raises:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, t, v, tb):
            if t is None:
                raise AssertionError(f"expected {self.exc.__name__}, nothing was raised")
            return issubclass(t, self.exc)

    class _Pytest:
        raises = staticmethod(_Raises)

        @staticmethod
        def approx(v, abs=1e-9):                                           # noqa: A002
            class _A:
                def __eq__(self, other):
                    return abs >= (other - v if other > v else v - other)
            return _A()

    pytest = _Pytest()

from scanno.scope import why_terminal                                       # noqa: E402


# ------------------------------------------------------------------ fixtures
#
# A real cohort's shape: a sealed node, a FORCE node stranding cells, a depth-1 declared LEAF,
# and a
# three-level lineage. Real names, because a failure message reading "A/B/C" tells the next reader
# nothing about which lineage broke.

TREE = {
    "children": {
        "root": ["Cardiomyocyte", "Stromal", "Endothelial", "Immune", "Mesothelial"],
        "Cardiomyocyte": ["Working cardiomyocyte", "Conduction cardiomyocyte"],
        "Stromal": ["Fibroblast", "Mural"],
        "Fibroblast": ["Matrifibrocyte", "Quiescent fibroblast"],
        "Mural": ["Pericyte", "Smooth muscle"],
        "Endothelial": ["Vascular endothelial", "Endocardial"],
        "Immune": ["Myeloid", "Lymphoid"],
        "Myeloid": ["Macrophage"],
        "Lymphoid": ["B cell", "NK cell"],
    },
}

#: What the cohort was DELIVERED: {sample: [(scope_path, independent_l1), ...]}
DELIVERED = {
    "S1": ([("Cardiomyocyte/Working cardiomyocyte", "Cardiomyocyte")] * 40
           + [("Stromal/Fibroblast", "Stromal")] * 18            # sealed -> stops at the parent
           + [("Stromal/Mural/Pericyte", "Stromal")] * 5
           + [("Endothelial/Vascular endothelial", "Endothelial")] * 17
           + [("Endothelial", "Endothelial")] * 6                # STRANDED on an open node
           + [("Mesothelial", "Mesothelial")] * 3                # a declared LEAF at depth 1
           + [("EXCLUDED", "EXCLUDED")] * 4),
    "S2": ([("Cardiomyocyte/Working cardiomyocyte", "Cardiomyocyte")] * 35
           + [("Stromal/Fibroblast", "Stromal")] * 20
           + [("Stromal/Mural/Smooth muscle", "Stromal")] * 4
           + [("Endothelial/Endocardial", "Endothelial")] * 9
           + [("Mesothelial", "Mesothelial")] * 2
           + [("UNRESOLVED", "UNRESOLVED")] * 2),
}

#: The vote, as `scanno scope --out` writes the part this section reads.
VERDICTS = {
    "root": {"verdict": "KEEP"},
    "Cardiomyocyte": {"verdict": "KEEP"},
    "Stromal": {"verdict": "KEEP"},
    "Stromal/Fibroblast": {"verdict": "SEAL"},
    "Stromal/Mural": {"verdict": "KEEP"},
    "Endothelial": {"verdict": "FORCE", "stranded": {"S1": 6}},
    "Immune": {"verdict": "UNVOTABLE"},
}

OTHER_TREE = {
    "children": {
        "root": ["Alpha", "Gamma"],
        "Alpha": ["Beta"],
        "Beta": ["Delta", "Epsilon"],
        "Delta": ["Zeta"],
    },
}

OTHER_DELIVERED = {
    "P": [("Alpha/Beta/Delta/Zeta", "Alpha")] * 7 + [("Gamma", "Gamma")] * 3,
    "Q": [("Alpha/Beta", "Alpha")] * 5 + [("Gamma", "Gamma")] * 2,
    "R": [("Alpha/Beta/Epsilon", "Alpha")] * 4,
}

OTHER_VERDICTS = {"root": {"verdict": "KEEP"}, "Alpha": {"verdict": "KEEP"},
                  "Alpha/Beta": {"verdict": "SEAL"}, "Alpha/Beta/Delta": {"verdict": "KEEP"}}


class FakeCtx:
    """Only what `taxonomy_section` reads. A real Context needs anndata; this needs pandas only.

    Built from the SAME (path, l1) pairs a real object would carry, and it computes the bands with
    the real `Context.l1_bands` / `Context.l1_concordance` rather than a reimplementation — a
    fixture that reimplements the thing under test keeps passing after that thing changes.
    """

    def __init__(self, delivered, tree, verdicts, l1_key="scanno_path_l1", depth=None):
        import pandas as pd

        from scanno.context import Context
        rows = [{"sample": s, "path": p, "l1": l} for s, v in delivered.items() for p, l in v]
        self.P = pd.DataFrame(rows)
        self.n = len(self.P)
        self.samples = sorted(delivered)
        self.l1_key = l1_key
        self.has_l1 = True
        self.tree = tree
        self.scope = {"nodes": verdicts}
        self.depth = depth or max(len(p.split("/")) for p in self.P["path"]
                                  if p not in ("EXCLUDED", "UNRESOLVED"))
        self.P["L1"] = [p.split("/")[0] for p in self.P["path"]]
        self.l1_bands = lambda: Context.l1_bands(self)
        self.l1_concordance = lambda: Context.l1_concordance(self)

    def colour(self, label):
        return "#888888"


def render(ctx, **kw):
    from scanno.document import taxonomy_section
    return "".join(taxonomy_section(ctx, **kw))


def test_an_internal_frame_key_is_translated_to_the_real_obs_column():
    """`P` uses short keys; `A.obs` uses whatever the annotation wrote. Anything reading the
    OBJECTS must translate, or it finds nothing, measures nothing, and draws an empty grid."""
    ctx = reference_cohort()
    assert ctx.obs_column_for("path") == ctx.path_key
    assert ctx.obs_column_for("nonsense") == "nonsense"      # unknown keys pass through


def test_a_column_no_object_carries_RAISES_rather_than_returning_zeros():
    ctx = reference_cohort()
    genes = list(ctx._gene_index())[:2] if hasattr(ctx, "_gene_index") else []
    if not genes:
        return
    try:
        ctx.expression_by_label(genes, 1, ["x"], col="a_column_nothing_has")
        assert False, "returned zeros instead of raising"
    except KeyError as e:
        assert "no object carries" in str(e), str(e)


def reference_cohort():
    return FakeCtx(DELIVERED, TREE, VERDICTS)


def other():
    return FakeCtx(OTHER_DELIVERED, OTHER_TREE, OTHER_VERDICTS, l1_key="lvl1")


def rows_of(h, needle):
    return [r for r in re.findall(r"<tr.*?</tr>", h, re.S) if needle in r]


# ---------------------------------------------------------------- the wiring, not the rendering

def test_the_section_is_on_the_live_path():
    """`document.py` assembles the cohort document. `report.py` does not, and never did."""
    doc = (ROOT_DIR / "scanno" / "document.py").read_text(encoding="utf-8")
    rpt = (ROOT_DIR / "scanno" / "report.py").read_text(encoding="utf-8")
    cli = (ROOT_DIR / "scanno" / "cli.py").read_text(encoding="utf-8")

    assert "def taxonomy_section(" in doc, "the section must live in the module write_all() calls"
    cohort = doc.split("def write_cohort(", 1)[1].split("\ndef ", 1)[0]
    assert "taxonomy_section(" in cohort, "write_cohort() must CALL it; defining it is not enough"
    assert "def taxonomy_section(" not in rpt, (
        "report.py is not imported by document.py; a section there renders nowhere")
    assert "from .report" not in doc and "import report" not in doc
    assert "from .document import write_all" in cli


def test_the_cli_threads_the_l1_column_and_the_declared_tree_into_the_context():
    cli = (ROOT_DIR / "scanno" / "cli.py").read_text(encoding="utf-8")
    ctxsrc = (ROOT_DIR / "scanno" / "context.py").read_text(encoding="utf-8")
    assert '"--l1-key"' in cli, "the report subcommand must accept --l1-key"
    assert "l1_key=a.l1_key" in cli, "and pass it into the Context"
    assert "l1_key=None" in ctxsrc and "self.l1_key = l1_key" in ctxsrc
    # The declared tree is what separates `leaf` from `sealed`. It used to be loaded only on the
    # --panels auto path, so every terminal label read as a complete call.
    assert "tree=declared_tree" in cli, "the DECLARED tree must reach the Context"
    assert "declared_tree = " in cli


def test_report_json_carries_both_columns_machine_readable():
    doc = (ROOT_DIR / "scanno" / "document.py").read_text(encoding="utf-8")
    payload = doc.split("payload = {", 1)[1].split("(out / \"report.json\")", 1)[0]
    assert '"l1"' in payload, "a consumer must not have to scrape the HTML for the two columns"


# ---------------------------------------------------------------- absence is NAMED

def test_no_independent_column_is_a_named_absence_not_a_missing_section():
    ctx = reference_cohort()
    ctx.has_l1 = False
    h = render(ctx)
    assert "The delivered annotation" in h, "the heading must appear even with no data"
    assert "--l1-key" in h and "--l1-tree" in h, "it must say how to GET the section"
    # and it must say what the reader loses, specifically the trap it protects against
    assert "showing one column twice" in h


# ---------------------------------------------------------------- both columns, in one picture

def test_the_independent_compartment_is_a_BAND_and_the_taxonomy_sits_inside_it():
    h = render(reference_cohort())
    band = rows_of(h, "class='band'")
    names = [re.sub("<.*?>", "", r) for r in band]
    assert any("Cardiomyocyte" in n for n in names), names
    assert any("Mesothelial" in n for n in names), names
    # the deep label is a row of its own, not merged into the band
    assert rows_of(h, "Working cardiomyocyte")


def test_a_terminal_row_carries_BOTH_a_count_and_why_it_stops():
    h = render(reference_cohort())
    r = rows_of(h, ">Fibroblast<")
    assert r, "the sealed label must appear"
    assert "sealed" in r[0], "a sealed node must not read as a complete call"
    assert "18" in r[0] or "20" in r[0] or "38" in r[0], r[0]


def test_the_four_reasons_a_branch_stops_are_distinguished_not_conflated():
    """`leaf`, `sealed`, `stranded` and `unvotable` have unrelated remedies."""
    h = render(reference_cohort())
    # a DECLARED leaf at depth 1 — complete
    assert "leaf" in rows_of(h, ">Mesothelial<")[-1]
    # a SEALED node at depth 2 — recoverable, and the same shape in the label column
    assert "sealed" in rows_of(h, ">Fibroblast<")[0]
    # a node left OPEN that stranded cells — thin evidence, nothing removed
    stranded = [r for r in rows_of(h, "stranded") if "Endothelial" in r]
    assert stranded, "the FORCE node's stranded cells must be marked stranded, not leaf"
    # and every one of them is explained in words on the page
    for word in ("leaf", "sealed", "stranded", "unvotable"):
        assert f"<b>{word}</b>" in h, f"{word} is used as a badge but never defined"


def test_a_depth_one_LEAF_and_a_depth_one_STRANDED_node_are_told_apart():
    """Both are one-word labels with no slash. Only the declared tree separates them."""
    assert why_terminal("Mesothelial", TREE, VERDICTS) == "leaf"
    assert why_terminal("Endothelial", TREE, VERDICTS) == "stranded"
    assert why_terminal("Stromal/Fibroblast", TREE, VERDICTS) == "sealed"
    assert why_terminal("Immune", TREE, VERDICTS) == "unvotable"
    assert why_terminal("EXCLUDED", TREE, VERDICTS) == "sentinel"
    # with no tree at all, nothing may be asserted to be sealed - but nor may the section
    # invent a reason: an unknown node has no declared children, so it is a leaf by the only
    # evidence available, and the CLI is what guarantees the tree is present.
    assert why_terminal("Stromal/Fibroblast", {}, VERDICTS) == "leaf"


def test_an_intermediate_node_nothing_terminates_at_is_kept_as_a_guide_row():
    """Dropping it flattens a three-level lineage into unrelated names."""
    h = render(reference_cohort())
    guide = rows_of(h, "class='guide'")
    assert any("Mural" in re.sub("<.*?>", "", g) for g in guide), (
        "Stromal/Mural has no cells of its own and must still appear")
    assert "branch" in "".join(guide)


def test_terminal_and_subtotal_counts_are_different_columns():
    """A node with both its own cells and descendants must not report one number for both."""
    h = render(reference_cohort())
    r = rows_of(h, ">Endothelial<")
    joined = "".join(r)
    assert "6" in joined, "the 6 stranded cells terminate at Endothelial"
    assert "32" in joined, "and 32 sit at or below it"


# ---------------------------------------------------------------- the concordance is MEASURED

def test_perfect_agreement_is_reported_as_a_measurement_not_a_guarantee():
    h = render(reference_cohort())
    con = reference_cohort().l1_concordance()
    assert con["n_disagree"] == 0
    assert "measured here rather than assumed" in h
    assert "nothing in the pipeline constrains them to" in h


def test_a_disagreement_is_shown_and_NEITHER_column_is_corrected():
    d = {s: list(v) for s, v in DELIVERED.items()}
    # one nucleus the independent walk calls Immune and the deep walk rooted at Stromal
    d["S1"] = d["S1"] + [("Stromal/Fibroblast", "Immune")] * 3
    ctx = FakeCtx(d, TREE, VERDICTS)
    con = ctx.l1_concordance()
    assert con["n_disagree"] == 3
    assert con["pairs"][("Stromal", "Immune")] == 3
    h = render(ctx)
    assert "Neither is corrected to match the other" in h
    # the disagreeing nuclei must appear in the band their INDEPENDENT column names,
    # carrying the full path they were actually given
    band_immune = h.split(">Immune<", 1)
    assert len(band_immune) > 1, "there must be an Immune band"
    assert "Stromal/Fibroblast" in h


def test_a_named_but_absent_column_is_not_perfect_agreement():
    """An l1 column of empty strings must not read as a walk that agreed everywhere."""
    from scanno.context import Context
    ctx = reference_cohort()
    ctx.P["l1"] = ""
    ctx.has_l1 = bool((ctx.P["l1"].astype(str) != "").any())
    assert ctx.has_l1 is False
    assert Context.l1_concordance(ctx) is None


# ---------------------------------------------------------------- nothing about THIS cohort

def test_the_section_renders_on_an_unrelated_cohort_of_a_different_depth():
    h = render(other())
    assert "Alpha" in h and "Gamma" in h
    # depth 4 in this fixture, 3 in the other: no layout assumption about how many levels exist
    assert other().depth == 4
    assert rows_of(h, ">Zeta<"), "the deepest label must render"
    assert "Cardiomyocyte" not in h and "Mesothelial" not in h, (
        "a name from the cohort this was written for has leaked into the output")


def test_no_cohort_size_or_node_name_is_hardcoded_in_the_source():
    doc = (ROOT_DIR / "scanno" / "document.py").read_text(encoding="utf-8")
    sec = doc.split("def taxonomy_section(", 1)[1].split("\n# ====", 1)[0]
    for banned in ("/10", "Cardiomyocyte", "Fibroblast", "Endothelial", "Mesothelial",
                   "Ten ", " ten "):
        assert banned not in sec, f"{banned!r} is hardcoded in taxonomy_section"
    # and the same for the scope section, which HAD a literal cohort size in its lede
    scope_sec = doc.split("def scope_section(", 1)[1].split("\n# ====", 1)[0]
    for banned in ("Ten independent walks", "4/7", "4/10"):
        assert banned not in scope_sec, f"{banned!r} is hardcoded in scope_section"


def test_samples_are_counted_against_the_cohort_not_a_literal():
    h = render(other())
    assert "/3" in h, "the OTHER cohort has three samples and the column must say so"
    assert "/10" not in h


# ---------------------------------------------------------------- the guard on --l1-key

def test_the_cli_refuses_an_l1_key_that_is_the_deep_walk_itself():
    cli = (ROOT_DIR / "scanno" / "cli.py").read_text(encoding="utf-8")
    fn = cli.split("def _report(", 1)[1].split("\ndef ", 1)[0]
    assert "a.l1_key == path_key" in fn, (
        "pointing --l1-key at the deep column would report a column agreeing with itself")
    assert 'REFUSE' in fn and '"/" in str(v)' in fn, (
        "a PATH column passed as --l1-key must be refused: it is not an independent L1")


def test_the_guard_does_not_refuse_the_good_outcome():
    """Perfect agreement is the RESULT, not a defect. It must never be refused."""
    cli = (ROOT_DIR / "scanno" / "cli.py").read_text(encoding="utf-8")
    fn = cli.split("def _report(", 1)[1].split("\ndef ", 1)[0]
    assert "refusing it would refuse the good outcome" in fn


# ---------------------------------------------------------------- what it cannot show

def test_the_cannot_show_note_names_the_three_different_remedies():
    h = render(reference_cohort())
    assert "cannot tell you why a branch stopped" in h
    assert "the remedy differs completely" in h
    assert "no truth set" in h
    # nuclei are not animals, and the samples column is where that shows
    assert "not cells and not animals" in h


def test_sentinels_are_marked_as_not_cell_types():
    h = render(reference_cohort())
    assert "not a cell type" in h
    ex = rows_of(h, "EXCLUDED")
    assert ex, "EXCLUDED must still appear - a silently dropped sentinel loses 4 nuclei"


# ---------------------------------------------------------------- FORCE, in the scope section

def test_the_scope_section_explains_and_counts_FORCE():
    from scanno.document import scope_section
    scope = {"rule": {"min_support": 1.0, "min_reach": 2, "descend_rule": "any",
                      "path_key": "p"},
             "samples": ["S1", "S2"], "n_samples": 2,
             "nodes": {"root": {"verdict": "KEEP", "n_reached": 2, "n_descended": 2,
                                "support": 1.0, "cells": {}, "cells_below": {}},
                       "Endothelial": {"verdict": "FORCE", "n_reached": 2, "n_descended": 2,
                                       "support": 1.0, "stranded": {"S1": 6},
                                       "cells": {"S1": 32}, "cells_below": {"S1": 26},
                                       "children_declared": ["Vascular endothelial",
                                                             "Endocardial"]}},
             "sealed": {}, "removed_labels": {}, "tree_lines": []}
    h = "".join(scope_section(scope))
    assert "FORCE" in h
    assert "nothing may terminate here" in h, "FORCE must be defined, not just printed"
    assert "what each FORCE reassigns" in h
    assert "6" in h, "the number of nuclei a FORCE moves must be on the page"
    # and it must be distinguished from a seal, which is the opposite operation
    assert "nuclei moved, never labels removed" in h


def test_no_FORCE_node_says_so_rather_than_rendering_nothing():
    from scanno.document import scope_section
    scope = {"rule": {"min_support": 1.0}, "samples": ["S1"], "n_samples": 1,
             "nodes": {"root": {"verdict": "KEEP", "n_reached": 1, "n_descended": 1,
                                "support": 1.0, "cells": {}, "cells_below": {}}},
             "sealed": {}, "removed_labels": {}, "tree_lines": []}
    h = "".join(scope_section(scope))
    assert "No node was forced" in h


def _run():
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:                                             # noqa: BLE001
            bad += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - bad} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_run())
