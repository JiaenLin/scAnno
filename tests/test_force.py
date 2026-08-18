"""`scanno annotate --scope`: the vote drives the annotation, and FORCE is honoured.

WHAT THIS PROTECTS

`scanno scope` returns three verdicts. KEEP and SEAL were implemented; FORCE was decided and
then not acted on, so cells stranded on a node the cohort agreed to split kept the name of a
COMPARTMENT — the same string the L1 column uses for every cell beneath it. Two delivered
columns then disagree about what one word means.

ONE PUSH WAS NOT ENOUGH

The first implementation pushed a stranded cell exactly one level. On the cohort this was
written for that left one sample's cluster on the PARENT of two real subtypes — an internal node
— so the compartment name was moved one level down rather than removed. FORCE now means: a cell
may not TERMINATE on an internal node. The push repeats until a leaf, at any depth, and section
8 below is the whole of that guarantee.

The fix has to hold eight things at once, and each has a test named for it:

  1. FORCE lands on the ARGMAX THE WALK ALREADY RECORDED — `trace[-1]["top"]` at the node it
     stopped on. Checked against a REAL `classify()` run, not against a hand-built trace, so
     the reading of `trace` is verified rather than assumed;
  2. after annotation NO cell carries a bare FORCE-node name;
  3. a forced call is marked as such per CELL and its margin travels with it — it is not the
     same evidential claim as a gap-cleared call and must stay filterable;
  4. the L1 column is untouched by seal and by force;
  5. without `--scope` nothing whatsoever changes;
  6. WHICH nodes are FORCE comes from the scope FILE — nothing in the library names a node, a
     sample, a cohort size or a threshold;
  7. `scanno/classify.py` is not touched. The walk, the gap test and GAP_CORPUS are as they
     were; everything here is post-walk or changes only which TREE is handed in;
  8. the push RECURSES to a leaf; each step past the first is really scored, by the same
     machinery the walk uses; the number of steps is recorded per cell; a chain that cannot
     reach a leaf is recorded rather than invented and never looped; and a single-level push is
     bit-for-bit what it was before.

    python tests/test_force.py        # no pytest needed; the shim below stands in
"""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pytest
except ImportError:                                                       # noqa: BLE001
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
        """Just enough of pytest to run this file with the interpreter that has scanno."""
        raises = staticmethod(_Raises)

        @staticmethod
        def approx(v, abs=1e-9):                                          # noqa: A002
            class _A:
                def __eq__(self, other):
                    return abs >= (other - v if other > v else v - other)
            return _A()

    pytest = _Pytest()

from scanno.force import (ASSIGNMENTS, BY_FORCE, BY_GAP, EXCLUDED,  # noqa: E402
                          apply_force, bare_force, check_scope, force_nodes, format_force,
                          internal_terminals, push_to_leaf, scope_verdicts, sealed_nodes)

# ============================================================================ fixtures
#
# Real names, because a failure message saying "A/B/C" tells the next reader nothing about
# which lineage broke. The SHAPE is what is under test; none of these names is in the library.

TREE = {
    "children": {
        "root": ["Cardiomyocyte", "Stromal", "Endothelial"],
        "Cardiomyocyte": ["Working cardiomyocyte", "Conduction cardiomyocyte"],
        "Stromal": ["Fibroblast", "Mural"],
        "Mural": ["Pericyte", "Smooth muscle"],
        "Endothelial": ["Vascular endothelial", "Endocardial"],
    },
    "patterns": {}, "members": {},
}


def scope(**verdicts):
    """A `scanno scope --out` payload carrying exactly the verdicts asked for.

    Keys are node PATHS, exactly as `scanno scope` writes them — `Stromal/Mural`, not `Mural` —
    so a path has to be passed through `**{...}`. Every internal node of TREE gets a verdict,
    because a real scope votes on all of them and a fixture that votes on some would be testing
    a file shape the tool never sees.
    """
    from scanno.scope import internal_nodes
    declared = internal_nodes(TREE)
    unknown = sorted(set(verdicts) - set(declared))
    assert not unknown, f"the fixture names nodes the tree does not declare: {unknown}"
    return {"nodes": {n: {"verdict": verdicts.get(n, "KEEP"), "children_declared": kids}
                      for n, kids in declared.items()},
            "n_samples": 10, "removed_labels": {}, "rule": {}}


def at(node, top, gap, survival=0.5, cover=0.6):
    return {"at": node, "top": top, "gap": gap, "survival": survival, "cover": cover}


def calls(*paths, trace=None):
    """`classify()`-shaped rows, one per cluster in order — what `apply_force` reads.

    The derived fields are DERIVED here exactly as `classify()` derives them, and not declared:

      `gap`                 the gap of the step that ENDED the walk — `trace[-1]`, cleared or not
      `survival` / `cover`  the statistics of the last ACCEPTED step — `trace[depth - 1]`

    A fixture free to disagree with those would let a bug through in the one place the whole
    feature reads.
    """
    out = []
    for i, p in enumerate(paths):
        sentinel = p in ("EXCLUDED", "UNRESOLVED")
        tr = (trace or {}).get(i)
        d = 0 if sentinel else len(p.split("/"))
        if tr is None and not sentinel:
            tr = [at(p.split("/")[-1], "?", 0.7)]
        acc = (tr[d - 1] if tr and d - 1 < len(tr) else (tr[-1] if tr else None))
        out.append({"cluster": i,
                    "label": (p if sentinel else p.split("/")[-1]),
                    "path": p,
                    "depth": d,
                    "gap": (tr[-1]["gap"] if tr else 0.0),
                    "survival": (acc["survival"] if acc else float("nan")),
                    "cover": (acc["cover"] if acc else float("nan")),
                    "excluded": p == "EXCLUDED",
                    "trace": tr or []})
    return out


# ================================= 1. FORCE lands on the argmax the REAL walk recorded
#
# The whole design rests on one reading of `classify()`: the last `trace` entry is written at
# the node the walk stopped on, and its `top` is that node's argmax child. Asserting that
# against a hand-built trace would only test the fixture. So the walk is actually run.

def _walk():
    """A real `classify()` run whose cluster 0 truncates on `Endothelial`. Returns its res."""
    import numpy as np
    from scanno.classify import GAP_PROFILE, classify

    G = 12
    genes = [f"G{i}" for i in range(G)]

    class Store:
        """The smallest thing `profile_weights` accepts: celltypes, mean, grade."""
        celltypes = ["VE", "EC", "CM"]
        mean = np.array([
            [3, 3, 3, 3, 2.0, 1.9, 0, 0, 0, 0, 0, 0],      # VE  \  one endothelial block,
            [3, 3, 3, 3, 1.9, 2.0, 0, 0, 0, 0, 0, 0],      # EC  /  faintly separated
            [0, 0, 0, 0, 0.0, 0.0, 0, 0, 3, 3, 3, 3],      # CM
        ], float)

        def grade(self, i):
            return "A"

    tree = {"children": {"root": ["Endothelial", "Cardiomyocyte"],
                         "Endothelial": ["Vascular endothelial", "Endocardial"]},
            "members": {"Endothelial": ["VE", "EC"], "Cardiomyocyte": ["CM"],
                        "Vascular endothelial": ["VE"], "Endocardial": ["EC"]},
            "genes": genes}
    Z = np.array([[3, 3, 3, 3, 2.00, 1.95, 0, 0, 0, 0, 0, 0],   # endothelial, subtype unclear
                  [0, 0, 0, 0, 0.00, 0.00, 0, 0, 3, 3, 3, 3],   # clearly CM
                  [3, 3, 3, 3, 3.00, 1.00, 0, 0, 0, 0, 0, 0]],  # clearly VE
                 float)
    return classify(Z, np.ones(G, bool), tree, store=Store(), gap_min=GAP_PROFILE), GAP_PROFILE


def test_force_lands_on_the_argmax_the_unchanged_walk_recorded():
    res, gap_min = _walk()

    # what the walk did, before anything touched it
    assert res[0]["path"] == "Endothelial", res[0]["path"]        # stranded on an open node
    assert res[0]["trace"][-1]["at"] == "Endothelial"
    assert res[0]["trace"][-1]["gap"] < gap_min                   # it stopped because of this
    argmax = res[0]["trace"][-1]["top"]
    assert res[2]["path"] == "Endothelial/Vascular endothelial"   # another cluster got there
    assert res[2]["gap"] >= gap_min                               # ... above the bar

    forced, rec = apply_force(res, ["Endothelial"], counts={0: 961, 1: 40715, 2: 17783})

    assert forced[0]["path"] == f"Endothelial/{argmax}"
    assert forced[0]["label"] == argmax
    assert forced[0]["depth"] == 2
    assert forced[0]["assignment"] == BY_FORCE
    assert rec["assigned"]["0"]["to"] == argmax
    assert rec["assigned"]["0"]["n_cells"] == 961
    # the cluster that cleared the bar is NOT touched, and is not marked as forced
    assert forced[2] == dict(res[2], assignment=BY_GAP)
    assert forced[1]["assignment"] == BY_GAP


def test_the_margin_of_a_forced_call_is_the_failing_gap_and_is_not_copied_anywhere():
    """One number, one home. `gap` already holds it, so nothing duplicates it."""
    res, gap_min = _walk()
    margin = res[0]["trace"][-1]["gap"]
    forced, rec = apply_force(res, ["Endothelial"])

    assert forced[0]["gap"] == margin           # classify already stored the failing gap here
    assert rec["assigned"]["0"]["margin"] == margin
    assert margin < gap_min
    # and the statistics move to the step that produced the LABEL - classify.py's own rule
    assert _same(forced[0]["survival"], res[0]["trace"][-1]["survival"])
    assert _same(forced[0]["cover"], res[0]["trace"][-1]["cover"])


def test_the_statistics_beside_a_forced_label_are_the_forcing_steps_not_the_level_aboves():
    """classify.py: "the statistics of the step that produced the LABEL - the last accepted one".

    Forcing makes `trace[-1]` that step. Leaving the previous level's numbers beside the new
    label would describe a decision that was not taken — the failure classify.py's own comment
    was written about.
    """
    res = calls("Endothelial",
                trace={0: [at("root", "Endothelial", 0.90, survival=0.91, cover=0.92),
                           at("Endothelial", "Endocardial", 0.02, survival=0.31, cover=0.32)]})
    assert (res[0]["survival"], res[0]["cover"]) == (0.91, 0.92)     # what the walk reported
    forced, _ = apply_force(res, ["Endothelial"])
    assert (forced[0]["survival"], forced[0]["cover"]) == (0.31, 0.32)


def test_apply_force_does_not_mutate_the_rows_classify_returned():
    """The walk's own output stays on disk-able. A caller can still write the unforced result."""
    res, _ = _walk()
    before = res[0]["path"]
    apply_force(res, ["Endothelial"])
    assert res[0]["path"] == before and "assignment" not in res[0]


# ==================================================== 2. no bare FORCE-node label survives

def test_no_bare_force_node_label_survives():
    res = calls("Endothelial", "Stromal", "Cardiomyocyte/Working cardiomyocyte",
                trace={0: [at("root", "Endothelial", 0.9), at("Endothelial", "Endocardial", .02)],
                       1: [at("root", "Stromal", 0.8), at("Stromal", "Mural", 0.05)]})
    forced, _ = apply_force(res, ["Endothelial", "Stromal"])

    assert bare_force(forced, ["Endothelial", "Stromal"]) == []
    assert [r["path"] for r in forced] == ["Endothelial/Endocardial", "Stromal/Mural",
                                           "Cardiomyocyte/Working cardiomyocyte"]


def test_bare_force_reads_the_finished_result_rather_than_trusting_apply_force():
    """The post-condition is a check on the OUTPUT, so it catches a res built any other way."""
    hand = calls("Endothelial", "Endothelial/Endocardial")
    stuck = bare_force(hand, ["Endothelial"])
    assert [s["cluster"] for s in stuck] == [0]
    assert stuck[0]["node"] == "Endothelial"


def test_a_cluster_the_walk_never_scored_is_not_forced_and_is_named():
    """`node_weights` returning None means no argmax was ever measured. Inventing one is the
    single thing a classifier that truncates must not do — so it refuses, loudly."""
    res = calls("Endothelial", trace={0: [at("root", "Endothelial", 0.9)]})   # broke BEFORE
    forced, rec = apply_force(res, ["Endothelial"], counts={0: 961})

    assert forced[0]["path"] == "Endothelial"          # unchanged: nothing was invented
    assert rec["n_forced"] == 0
    assert rec["unforceable"]["0"]["node"] == "Endothelial"
    assert rec["unforceable"]["0"]["stopped_at"] == "root"
    assert rec["unforceable"]["0"]["n_cells"] == 961
    # and the post-condition catches it, which is what makes the CLI refuse rather than write
    assert bare_force(forced, ["Endothelial"]) != []
    assert "REFUSE" in "\n".join(format_force(rec))


def test_an_excluded_cluster_is_never_forced():
    """Withheld upstream, never walked. `forced` would claim a decision it took no part in."""
    forced, rec = apply_force(calls("EXCLUDED", "Endothelial",
                                    trace={1: [at("root", "Endothelial", .9),
                                               at("Endothelial", "Endocardial", .02)]}),
                              ["Endothelial"])
    assert forced[0]["path"] == "EXCLUDED" and forced[0]["assignment"] == EXCLUDED
    assert forced[1]["assignment"] == BY_FORCE
    assert rec["clusters_by_assignment"] == {BY_GAP: 0, BY_FORCE: 1, EXCLUDED: 1}


# ============================ 3. a forced call is marked per CELL, and stays filterable

def test_every_cell_says_how_it_was_assigned():
    import numpy as np
    from scanno.emit import annotate_obs
    A = _obj(6)
    y = np.array([0, 0, 1, 1, 2, 2])
    res, _ = apply_force(
        calls("Endothelial", "Endothelial/Endocardial", "Cardiomyocyte/Working cardiomyocyte",
              trace={0: [at("root", "Endothelial", .9), at("Endothelial", "Endocardial", .02)]}),
        ["Endothelial"])
    written = annotate_obs(A, res, y, assignment=True)

    assert "scanno_assignment" in written
    assert list(A.obs["scanno_assignment"]) == [BY_FORCE, BY_FORCE, BY_GAP, BY_GAP,
                                                BY_GAP, BY_GAP]
    # the pair that makes a sensitivity check possible: WHICH cells, and on WHAT margin
    forced = A.obs["scanno_assignment"] == BY_FORCE
    assert set(A.obs.loc[forced, "scanno_path"]) == {"Endothelial/Endocardial"}
    assert float(A.obs.loc[forced, "scanno_gap"].iloc[0]) == pytest.approx(0.02, abs=1e-6)
    # ... and the label it would have carried, recoverable exactly, with no column of its own
    assert {p.rsplit("/", 1)[0] for p in A.obs.loc[forced, "scanno_path"]} == {"Endothelial"}


def test_a_flagged_nucleus_inside_a_forced_cluster_is_EXCLUDED_not_forced():
    import numpy as np
    from scanno.emit import annotate_obs
    A = _obj(4)
    y = np.array([0, 0, 1, 1])
    res, _ = apply_force(
        calls("Endothelial", "Cardiomyocyte/Working cardiomyocyte",
              trace={0: [at("root", "Endothelial", .9), at("Endothelial", "Endocardial", .02)]}),
        ["Endothelial"])
    annotate_obs(A, res, y, flag=np.array([False, True, False, False]), assignment=True)
    assert list(A.obs["scanno_assignment"]) == [BY_FORCE, EXCLUDED, BY_GAP, BY_GAP]
    assert list(A.obs["scanno_cell_type"])[1] == "EXCLUDED"


def test_the_provenance_names_the_node_the_child_and_the_margin_per_cluster():
    import numpy as np
    from scanno.emit import annotate_obs, force_provenance
    A = _obj(2)
    res, rec = apply_force(
        calls("Endothelial",
              trace={0: [at("root", "Endothelial", .9), at("Endothelial", "Endocardial", .02)]}),
        ["Endothelial"], counts={0: 961})
    annotate_obs(A, res, np.array([0, 0]), assignment=True)
    key = force_provenance(A, rec, scope="scope.json")

    assert key == "scanno_assignment_provenance"          # sorts beside the column it describes
    prov = A.uns[key]
    assert prov["verdict"] == "FORCE"
    assert prov["column"] == "scanno_assignment"
    assert prov["scope"] == "scope.json"
    assert prov["nodes"] == ["Endothelial"]
    # `steps` and `margins` are LISTS because a push can take more than one step. For a
    # single-step push they hold one entry each, and `margin` still names the FIRST — the margin
    # at the FORCE node — so a reader of the old record reads the same number from the same key.
    assert prov["assigned"]["0"] == {"node": "Endothelial", "to": "Endocardial",
                                     "path": "Endothelial/Endocardial", "margin": 0.02,
                                     "survival": 0.5, "cover": 0.6, "n_cells": 961,
                                     "force_depth": 1,
                                     "steps": ["Endothelial/Endocardial"], "margins": [0.02]}
    assert prov["by_node"] == {"Endothelial": {"Endocardial": 961}}
    assert prov["n_cells_forced"] == 961
    assert prov["clusters_by_force_depth"] == {"1": 1}


def test_the_provenance_holds_no_list_of_dicts_so_the_object_can_be_written():
    """anndata writes `uns` mappings; a list of dicts becomes an object array and raises.

    Found the expensive way once already: a record that is correct in memory and unwritable.
    """
    _, rec = apply_force(calls("Endothelial",
                               trace={0: [at("root", "Endothelial", .9),
                                          at("Endothelial", "Endocardial", .02)]}),
                         ["Endothelial"])

    def walk(v, where):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, f"{where}.{k}")
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                assert not isinstance(x, (dict, list, tuple)), f"{where}[{i}] is nested"
    walk(rec, "record")


def test_the_object_round_trips_through_h5ad_with_the_provenance_intact():
    import tempfile

    import anndata as ad
    import numpy as np
    from scanno.emit import annotate_obs, force_provenance
    A = _obj(2)
    res, rec = apply_force(
        calls("Endothelial",
              trace={0: [at("root", "Endothelial", .9), at("Endothelial", "Endocardial", .02)]}),
        ["Endothelial"], counts={0: 961})
    annotate_obs(A, res, np.array([0, 0]), assignment=True)
    force_provenance(A, rec, scope="scope.json")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "forced.h5ad"
        A.write_h5ad(p)
        B = ad.read_h5ad(p)
    assert list(B.obs["scanno_assignment"]) == [BY_FORCE, BY_FORCE]
    assert B.uns["scanno_assignment_provenance"]["assigned"]["0"]["to"] == "Endocardial"


# ================================================== 4. L1 is untouched by seal and by force

def test_the_l1_column_is_untouched_by_seal_and_by_force():
    """The scope acts strictly BELOW level 1, so neither verdict can reach the L1 tree."""
    from scanno.scope import seal_tree, truncate_tree
    v = scope(**{"Stromal": "SEAL", "Endothelial": "FORCE"})["nodes"]
    assert sealed_nodes(v) == ["Stromal"] and force_nodes(v) == ["Endothelial"]

    l1 = truncate_tree(TREE, 1)
    sealed, _ = seal_tree(TREE, v)
    assert truncate_tree(sealed, 1) == l1                     # a seal cannot reach it
    # and nothing in the L1 tree can be FORCEd: its nodes have no children to be pushed to
    assert all(c not in l1["children"] for c in l1["children"]["root"])
    assert check_scope(scope(root="FORCE"), l1)               # root FORCE is refused, not run


def test_force_is_applied_to_the_deep_walk_and_never_to_the_independent_l1():
    """AST, not behaviour: the guarantee is that the L1 result never reaches `apply_force`."""
    src = (ROOT / "scanno" / "cli.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_annotate")
    used = [n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "apply_force"]
    assert len(used) == 1, f"{len(used)} apply_force() call(s) in _annotate"
    assert getattr(used[0].args[0], "id", "") == "res", ast.unparse(used[0])
    # the second walk still happens, with the same bar and the same withheld set
    two = [n for n in ast.walk(fn)
           if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "classify"]
    assert [c.args[2].id for c in two] == ["tree", "l1_tree"]


def test_the_independent_l1_column_still_says_it_is_independent_under_a_scope():
    import numpy as np
    from scanno.emit import annotate_obs, independent_l1
    A = _obj(4)
    y = np.array([0, 0, 1, 1])
    res, _ = apply_force(
        calls("Endothelial", "Cardiomyocyte/Working cardiomyocyte",
              trace={0: [at("root", "Endothelial", .9), at("Endothelial", "Endocardial", .02)]}),
        ["Endothelial"])
    annotate_obs(A, res, y, assignment=True)
    col, rec = independent_l1(A, calls("Endothelial", "Cardiomyocyte"), y, tree="l1.json")

    assert A.uns[f"{col}_provenance"]["source"] == "independent"
    assert rec["n_disagree"] == 0            # forcing moved level 2, never level 1
    assert list(A.obs["scAnno_L1"]) == ["Endothelial", "Endothelial",
                                        "Cardiomyocyte", "Cardiomyocyte"]
    assert list(A.obs["scAnno_L2"])[0] == "Endothelial/Endocardial"


# ========================================================= 5. absent --scope changes nothing

def test_without_a_scope_the_annotate_path_is_untouched():
    src = (ROOT / "scanno" / "cli.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_annotate")
    body = ast.unparse(fn)
    assert _stmt("scope, force_paths = None, []") in body, "the scope must default to nothing"
    assert "if force_paths:" in body, "the reassignment must be gated on the scope"
    assert _stmt("force_rec = None") in body
    assert "assignment=force_rec is not None" in body, "the column must be gated too"
    # nothing may seal or force outside those two guards
    assert body.count("apply_force(") == 1 and body.count("seal_tree(") == 1

    opts = _annotate_parser()
    assert "--scope" in opts, sorted(opts)
    assert "required" not in opts["--scope"] and "default" not in opts["--scope"]


def test_without_the_flag_no_assignment_column_is_written():
    """A column that can only hold one value is a column a reader checks to learn nothing."""
    import numpy as np
    from scanno.emit import annotate_obs
    A = _obj(2)
    written = annotate_obs(A, calls("Cardiomyocyte/Working cardiomyocyte"), np.array([0, 0]))
    assert written == ["scanno_cell_type", "scanno_path", "scanno_depth", "scanno_gap",
                       "scanno_survival", "scAnno_L1", "scAnno_L2"], written
    assert "scanno_assignment" not in A.obs
    assert not [k for k in A.uns if "assignment" in str(k)]


def test_an_empty_force_set_changes_no_label():
    res = calls("Endothelial", "Cardiomyocyte/Working cardiomyocyte")
    out, rec = apply_force(res, [])
    assert [r["path"] for r in out] == [r["path"] for r in res]
    assert rec["n_forced"] == 0 and rec["nodes"] == []
    assert format_force(rec) == [
        "  FORCE  0 node(s) the cohort agreed to split, on which nothing may terminate: (none)"]


# =========================================== 6. the decision comes from the FILE, not the code

def test_which_nodes_are_forced_is_read_from_the_scope_and_from_nowhere_else():
    assert force_nodes(scope(Endothelial="FORCE")["nodes"]) == ["Endothelial"]
    assert force_nodes(scope(**{"Stromal": "FORCE",
                                "Stromal/Mural": "FORCE"})["nodes"]) == ["Stromal",
                                                                         "Stromal/Mural"]
    assert force_nodes(scope()["nodes"]) == []
    assert sealed_nodes(scope(**{"Stromal/Mural": "SEAL"})["nodes"]) == ["Stromal/Mural"]


# ================================= 10. --resolve: a label column with no holes, and no invention
#
# The walk truncates rather than guessing, so a cohort carries UNRESOLVED cells and cells
# labelled with a compartment. That is the answer. `--resolve` writes a SECOND set of columns in
# which every walked cell sits on a leaf, and the whole of its honesty is the origin column: a
# reader must be able to tell a leaf that was REACHED from one that was ASSIGNED.

def _scorer_for(tree):
    """A scorer that always prefers the FIRST declared child, so the descent is predictable."""
    def score(cid, node):
        kids = tree["children"].get(node)
        if not kids:
            return None
        return at(node, kids[0], 0.11)
    return score


def test_an_unresolved_cluster_is_pushed_from_the_root_to_a_leaf():
    from scanno.force import FROM_ROOT, resolve_to_leaf
    res = calls("UNRESOLVED", trace={0: [at("root", "Cardiomyocyte", 0.02)]})
    rows, rec = resolve_to_leaf(res, tree=TREE, scorer=_scorer_for(TREE))
    r = rows[0]
    assert r["resolved_origin"] == FROM_ROOT, r["resolved_origin"]
    assert r["resolved_path"] == "Cardiomyocyte/Working cardiomyocyte", r["resolved_path"]
    assert r["resolved_label"] == "Working cardiomyocyte"
    # and the path carries NO leading root, which is what joining from the root name invites
    assert not r["resolved_path"].startswith("root")


def test_the_first_step_is_the_argmax_the_walk_already_recorded_at_the_root():
    # Nothing is re-scored to choose the root's child: trace[0]["top"] IS the measurement, and
    # a resolution that picked differently would be inventing a destination.
    from scanno.force import resolve_to_leaf
    res = calls("UNRESOLVED", trace={0: [at("root", "Endothelial", 0.03)]})
    rows, _ = resolve_to_leaf(res, tree=TREE, scorer=_scorer_for(TREE))
    assert rows[0]["resolved_path"].split("/")[0] == "Endothelial"


def test_the_principled_label_is_left_exactly_as_it_was():
    from scanno.force import resolve_to_leaf
    res = calls("UNRESOLVED", trace={0: [at("root", "Stromal", 0.02)]})
    rows, _ = resolve_to_leaf(res, tree=TREE, scorer=_scorer_for(TREE))
    assert rows[0]["label"] == "UNRESOLVED", "the decision NOT to guess must survive"
    assert rows[0]["path"] == "UNRESOLVED"
    assert rows[0]["depth"] == 0


def test_a_cluster_already_on_a_leaf_is_not_touched_and_says_so():
    from scanno.force import FROM_WALK, resolve_to_leaf
    rows, _ = resolve_to_leaf(calls("Stromal/Mural/Pericyte"), tree=TREE,
                              scorer=_scorer_for(TREE))
    assert rows[0]["resolved_origin"] == FROM_WALK
    assert rows[0]["resolved_path"] == "Stromal/Mural/Pericyte"


def test_a_cluster_stranded_on_an_internal_node_is_pushed_from_there():
    from scanno.force import FROM_INTERNAL, resolve_to_leaf
    res = calls("Stromal/Mural", trace={0: [at("root", "Stromal", 0.9),
                                            at("Stromal", "Mural", 0.8),
                                            at("Mural", "Pericyte", 0.05)]})
    rows, _ = resolve_to_leaf(res, tree=TREE, scorer=_scorer_for(TREE))
    assert rows[0]["resolved_origin"] == FROM_INTERNAL
    assert rows[0]["resolved_path"] == "Stromal/Mural/Pericyte"


def test_an_excluded_cluster_is_never_resolved():
    # Withheld BEFORE the walk: no trace to descend, and a leaf here would be pure invention.
    from scanno.force import EXCLUDED, resolve_to_leaf
    rows, _ = resolve_to_leaf(calls("EXCLUDED"), tree=TREE, scorer=_scorer_for(TREE))
    assert rows[0]["resolved_label"] == EXCLUDED
    assert rows[0]["resolved_origin"] == EXCLUDED


def test_a_cluster_the_walk_never_scored_stays_unresolved_and_is_named():
    from scanno.force import UNRESOLVED, resolve_to_leaf
    res = calls("UNRESOLVED", trace={0: []})
    rows, rec = resolve_to_leaf(res, tree=TREE, scorer=_scorer_for(TREE))
    assert rows[0]["resolved_label"] == UNRESOLVED
    assert "0" in rec["unresolvable"], rec["unresolvable"]
    assert "no most-similar child" in rec["unresolvable"]["0"]["why"]


def test_a_descent_that_cannot_reach_a_leaf_invents_nothing():
    from scanno.force import UNRESOLVED, resolve_to_leaf
    res = calls("UNRESOLVED", trace={0: [at("root", "Cardiomyocyte", 0.02)]})
    # internal node, and no scorer to take the next step
    rows, rec = resolve_to_leaf(res, tree=TREE, scorer=None)
    assert rows[0]["resolved_label"] == UNRESOLVED
    assert rec["unresolvable"], "the refusal must be recorded, not silently dropped"


def test_the_record_counts_cells_not_only_clusters():
    from scanno.force import FROM_ROOT, resolve_to_leaf
    res = calls("UNRESOLVED", trace={0: [at("root", "Cardiomyocyte", 0.02)]})
    _, rec = resolve_to_leaf(res, tree=TREE, scorer=_scorer_for(TREE), counts={0: 4321})
    assert rec["cells_by_origin"][FROM_ROOT] == 4321


def test_resolve_does_not_mutate_the_rows_it_was_given():
    from scanno.force import resolve_to_leaf
    res = calls("UNRESOLVED", trace={0: [at("root", "Cardiomyocyte", 0.02)]})
    before = dict(res[0])
    resolve_to_leaf(res, tree=TREE, scorer=_scorer_for(TREE))
    assert res[0] == before, "apply-style functions here return new rows and touch nothing"


def test_every_cell_says_whether_its_leaf_was_reached_or_assigned():
    # The three columns are ONE statement. Without the origin, a consumer reads a root-level
    # guess as a call, which is the whole reason the walk declined to make it.
    import numpy as np
    from scanno.emit import per_cell
    from scanno.force import FROM_ROOT, FROM_WALK, resolve_to_leaf
    res = calls("Stromal/Mural/Pericyte", "UNRESOLVED",
                trace={1: [at("root", "Cardiomyocyte", 0.02)]})
    rows, _ = resolve_to_leaf(res, tree=TREE, scorer=_scorer_for(TREE))
    cols = per_cell(rows, np.array([0, 0, 1]))
    assert list(cols["resolved_origin"]) == [FROM_WALK, FROM_WALK, FROM_ROOT]
    assert cols["resolved"][2] == "Working cardiomyocyte"
    # the ordinary column still holds the refusal
    assert cols["cell_type"][2] == "UNRESOLVED"


def test_a_flagged_nucleus_is_excluded_in_the_resolved_column_too():
    import numpy as np
    from scanno.emit import per_cell
    from scanno.force import EXCLUDED, resolve_to_leaf
    res = calls("UNRESOLVED", trace={0: [at("root", "Cardiomyocyte", 0.02)]})
    rows, _ = resolve_to_leaf(res, tree=TREE, scorer=_scorer_for(TREE))
    cols = per_cell(rows, np.array([0, 0]), flag=np.array([False, True]))
    assert cols["resolved"][1] == EXCLUDED
    assert cols["resolved_origin"][1] == EXCLUDED
    assert cols["resolved"][0] == "Working cardiomyocyte"


def test_per_cell_defaults_the_resolved_column_to_the_walks_own_answer():
    # A run that never called resolve_to_leaf must still produce the columns, equal to the label
    # columns - not empty ones, and not a crash.
    import numpy as np
    from scanno.emit import per_cell
    from scanno.force import FROM_WALK
    cols = per_cell(calls("Stromal/Fibroblast"), np.array([0]))
    assert cols["resolved"][0] == "Fibroblast"
    assert cols["resolved_path"][0] == "Stromal/Fibroblast"
    assert cols["resolved_origin"][0] == FROM_WALK


def test_the_library_names_no_node_no_sample_and_no_cohort_size():
    """Docstrings may cite the cohort this was written for. CODE may not encode it.

    A literal `/10` shipped in `format_report` once: on seven samples it printed "7/10" and
    nobody reading it would have known. This is the check that keeps its kin out.
    """
    cohort = ("Cardiomyocyte", "Endothelial", "Fibroblast", "Matrifibrocyte", "Pericyte",
              "Macrophage", "Mesothelial", "Endocardial", "Lymphoid", "Aging", "Young",
              "SAMBO", "mouse_heart")
    # EVERY module, not a list of four. A guard naming its own files stops covering the package
    # the moment a module is added, and the newest module is the likeliest to carry a leak.
    for mod in sorted(x.name for x in (ROOT / "scanno").glob("*.py")):
        for s in _code_strings(ROOT / "scanno" / mod):
            hit = [w for w in cohort if w in s]
            assert not hit, f"scanno/{mod} encodes {hit} in a code string: {s!r}"

    fn = next(n for n in ast.walk(ast.parse((ROOT / "scanno" / "cli.py").read_text("utf-8")))
              if isinstance(n, ast.FunctionDef) and n.name == "_annotate")
    for n in ast.walk(fn):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            hit = [w for w in cohort if w in n.value]
            assert not hit, f"cli._annotate encodes {hit}: {n.value!r}"


def test_force_declares_no_threshold_at_all():
    """FORCE has no bar to tune. Every value it acts on comes out of the scope file.

    Two shapes are checked, because a hardcoded threshold can wear either: a float anywhere,
    and a comparison against a number. Integers are allowed — they are list indices, and
    `trace[-1]` is not a parameter.
    """
    mod = ast.parse((ROOT / "scanno" / "force.py").read_text(encoding="utf-8"))
    floats = [n.value for n in ast.walk(mod)
              if isinstance(n, ast.Constant) and isinstance(n.value, float)]
    assert floats == [], floats
    for n in ast.walk(mod):
        if isinstance(n, ast.Compare):
            for c in n.comparators:
                assert not (isinstance(c, ast.Constant)
                            and isinstance(c.value, (int, float))
                            and not isinstance(c.value, bool)), ast.unparse(n)

    top = [t.id for s in mod.body if isinstance(s, ast.Assign) for t in s.targets
           if isinstance(t, ast.Name)]
    assert set(top) == {"SEP", "ROOT", "FORCE", "SEAL", "BY_GAP", "BY_FORCE", "EXCLUDED",
                        "ASSIGNMENTS"}, top


def test_a_scope_that_is_not_a_scope_is_refused():
    """`--scope sealed_tree.json` would otherwise seal nothing and force nothing, silently."""
    with pytest.raises(ValueError):
        scope_verdicts(TREE)
    with pytest.raises(ValueError):
        scope_verdicts({"nodes": ["Endothelial"]})


def test_the_root_may_not_be_forced():
    """Root truncation is UNRESOLVED — a different decision, which `scanno scope` never emits."""
    problems = check_scope(scope(root="FORCE"), TREE)
    assert len(problems) == 1 and "ROOT" in problems[0] and "UNRESOLVED" in problems[0]


def test_a_force_node_whose_child_is_also_force_is_refused():
    """Forcing there would land the cell on a bare FORCE name one level down."""
    problems = check_scope(scope(**{"Stromal": "FORCE", "Stromal/Mural": "FORCE"}), TREE)
    assert len(problems) == 1 and "Stromal" in problems[0] and "Mural" in problems[0]
    # the same two nodes are fine as long as the parent-child pair is not both FORCE
    assert check_scope(scope(Stromal="FORCE", Endothelial="FORCE"), TREE) == []


def test_a_force_node_with_nothing_to_push_onto_is_refused():
    leafy = {"children": {"root": ["Endothelial"]}, "patterns": {}, "members": {}}
    problems = check_scope({"nodes": {"root": {"verdict": "KEEP"},
                                      "Endothelial": {"verdict": "FORCE"}}}, leafy)
    assert any("no children" in p for p in problems), problems


def test_a_scope_voted_on_a_different_tree_is_refused_but_re_applying_its_own_is_not():
    """Idempotence matters: `--tree sealed.json --scope its_own_scope.json` must still run."""
    from scanno.scope import seal_tree
    sc = scope(**{"Stromal": "SEAL", "Endothelial": "FORCE"})
    assert check_scope(sc, TREE) == []

    sealed, removed = seal_tree(TREE, sc["nodes"])
    assert removed == {"Stromal": ["Fibroblast", "Mural"]}
    assert check_scope(sc, sealed) == []            # the seal is gone from the tree: still fine
    sealed_twice, again = seal_tree(sealed, sc["nodes"])
    assert again == {} and sealed_twice["children"] == sealed["children"]

    other = {"children": {"root": ["Immune"], "Immune": ["Myeloid", "Lymphoid"]},
             "patterns": {}, "members": {}}
    problems = check_scope(sc, other)
    assert any("voted on a different tree" in p for p in problems), problems


def test_a_name_at_two_positions_is_refused_because_the_trace_is_keyed_by_bare_name():
    dup = {"children": {"root": ["Stromal", "Endothelial"],
                        "Stromal": ["Mural"], "Endothelial": ["Mural"],
                        "Mural": ["Pericyte", "Smooth muscle"]},
           "patterns": {}, "members": {}}
    problems = check_scope({"nodes": {"root": {"verdict": "KEEP"}}}, dup)
    assert any("more than one position" in p for p in problems), problems


# ============================================================ 7. the walk itself is unchanged

#: sha256 of scanno/classify.py as of the commit that added --scope. The requirement is not
#: "classify still works" but "classify was NOT TOUCHED": a FORCE that needed the walk edited
#: would not be the same walk, and every gap in the output would mean two different things.
CLASSIFY_SHA = "9096b9aef108e88c9678009ab6fa522aa1cb9cac11485433db1ccfbaeef2c8ba"


def test_classify_py_is_not_touched_by_the_scope_feature():
    src = (ROOT / "scanno" / "classify.py").read_text(encoding="utf-8")
    got = hashlib.sha256(src.encode()).hexdigest()
    assert got == CLASSIFY_SHA, (
        f"classify.py digest is {got}, pinned {CLASSIFY_SHA}. --scope is post-walk and must "
        f"not change the walk; if this edit was deliberate, say so out loud and re-pin.")

    mod = ast.parse(src)
    fn = next(n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "classify")
    args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert args == ["Z", "usable", "tree", "store", "assertions", "gap_min", "exclude"], args
    for word in ("scope", "force", "FORCE", "SEAL", "assignment"):
        assert word not in src, f"classify.py mentions {word!r} - the walk was parameterised"

    from scanno.classify import GAP_CORPUS, GAP_PROFILE
    assert (GAP_CORPUS, GAP_PROFILE) == (0.30, 0.15)


def test_the_trace_this_feature_reads_is_the_one_classify_writes():
    """`trace[-1]["top"]` is only the argmax if classify still records `order[srt[0]]` there.

    Read out of the source rather than assumed, because the whole feature rests on it and a
    rename inside the walk would leave every forced call pointing at the wrong child while
    every test that used a hand-built trace went on passing.
    """
    src = (ROOT / "scanno" / "classify.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "classify")
    app = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
           and getattr(n.func, "attr", "") == "append"
           and getattr(n.func.value, "id", "") == "trace"]
    assert len(app) == 1, f"{len(app)} trace.append call(s)"
    d = app[0].args[0]
    keys = [k.value for k in d.keys]
    assert keys[:3] == ["at", "top", "gap"], keys
    assert ast.unparse(d.values[keys.index("at")]) == "node"
    assert ast.unparse(d.values[keys.index("top")]) == "order[srt[0]]"

    # ... and the argmax is the FIRST of a descending sort, i.e. the most similar child
    assert "srt = np.argsort(-s)" in src


# ================================================ 8. the push RECURSES, to a LEAF, and says so
#
# The defect this section exists for: a single push landed a stranded cluster on the PARENT of
# two subtypes, so the cell terminated on an internal node carrying a compartment name — the
# thing FORCE removes, moved one level down. Everything below is that guarantee, taken apart.


def _deep():
    """A REAL `classify()` run stranded on a node whose ARGMAX CHILD IS ALSO INTERNAL.

    That is the shape one push cannot fix, and it is built here rather than described: the walk
    truncates at `Endothelial`, whichever child it prefers has children of its own, and only a
    second SCORE can say which of those the cluster resembles — the walk never looked.

    Returns everything a scorer needs: `(res, tree, Z, usable, store, gap_min)`.
    """
    import numpy as np
    from scanno.classify import GAP_PROFILE, classify

    G = 12
    genes = [f"G{i}" for i in range(G)]

    class Store:
        #: genes 0-3 the shared compartment block, 4/5 separate the two subtypes faintly,
        #: 6/7 separate the sub-subtypes cleanly, 8-11 the other compartment. The sub-subtype
        #: genes are ZERO in both subtype profiles, so adding them cannot move the score at the
        #: node above — which is what keeps this fixture's truncation the same one `_walk` has.
        celltypes = ["VE", "EC", "CM", "VEa", "VEb", "ECa", "ECb"]
        mean = np.array([
            [3, 3, 3, 3, 2.0, 1.9, 0, 0, 0, 0, 0, 0],      # VE
            [3, 3, 3, 3, 1.9, 2.0, 0, 0, 0, 0, 0, 0],      # EC
            [0, 0, 0, 0, 0.0, 0.0, 0, 0, 3, 3, 3, 3],      # CM
            [3, 3, 3, 3, 2.0, 1.9, 3, 0, 0, 0, 0, 0],      # VEa
            [3, 3, 3, 3, 2.0, 1.9, 0, 3, 0, 0, 0, 0],      # VEb
            [3, 3, 3, 3, 1.9, 2.0, 3, 0, 0, 0, 0, 0],      # ECa
            [3, 3, 3, 3, 1.9, 2.0, 0, 3, 0, 0, 0, 0],      # ECb
        ], float)

        def grade(self, i):
            return "A"

    tree = {"children": {"root": ["Endothelial", "Cardiomyocyte"],
                         "Endothelial": ["Vascular endothelial", "Endocardial"],
                         "Vascular endothelial": ["Arterial", "Capillary"],
                         "Endocardial": ["Atrial endocardial", "Ventricular endocardial"]},
            "members": {"Endothelial": ["VE", "EC"], "Cardiomyocyte": ["CM"],
                        "Vascular endothelial": ["VE"], "Endocardial": ["EC"],
                        "Arterial": ["VEa"], "Capillary": ["VEb"],
                        "Atrial endocardial": ["ECa"], "Ventricular endocardial": ["ECb"]},
            "genes": genes}
    Z = np.array([[3, 3, 3, 3, 2.00, 1.95, 2, 0, 0, 0, 0, 0],   # subtype unclear, sub-subtype
                  [0, 0, 0, 0, 0.00, 0.00, 0, 0, 3, 3, 3, 3],   #   clear: the stranded one
                  [3, 3, 3, 3, 3.00, 1.00, 3, 0, 0, 0, 0, 0]],  # clear all the way down
                 float)
    usable = np.ones(G, bool)
    store = Store()
    res = classify(Z, usable, tree, store=store, gap_min=GAP_PROFILE)
    return res, tree, Z, usable, store, GAP_PROFILE


def _scorer(table):
    """A `node_scorer`-shaped callable driven by a table `{node: (top, gap)}`.

    Used where the arithmetic is not what is under test. Anything absent from the table returns
    None, which is exactly what `node_scorer` returns for a node it cannot score.
    """
    return lambda cid, node: (at(node, *table[node]) if node in table else None)


def test_a_forced_push_continues_until_it_reaches_a_leaf_of_the_scoped_tree():
    """The guarantee, end to end and with real scoring: no cell terminates on an internal node."""
    from scanno.step import node_scorer
    res, tree, Z, usable, store, gap_min = _deep()

    assert res[0]["path"] == "Endothelial"                     # stranded on an internal node
    first = res[0]["trace"][-1]["top"]
    assert tree["children"][first], f"{first} must be internal or the fixture proves nothing"

    forced, rec = apply_force(res, ["Endothelial"], counts={0: 99, 1: 40715, 2: 17783},
                              tree=tree, scorer=node_scorer(Z, usable, tree, store=store))

    path = forced[0]["path"].split("/")
    assert path[0] == "Endothelial" and path[1] == first
    assert not tree["children"].get(path[-1]), f"{path[-1]} is not a leaf"
    assert forced[0]["depth"] == len(path) == 3
    assert forced[0]["label"] == path[-1]
    assert forced[0]["assignment"] == BY_FORCE
    assert internal_terminals(forced, tree) == []
    assert bare_force(forced, ["Endothelial"]) == []
    assert rec["recursive"] is True
    assert rec["clusters_by_force_depth"] == {"2": 1}


def test_a_forced_step_is_scored_exactly_as_the_walk_would_have_scored_it():
    """The anti-drift guarantee, and the reason a forced step is comparable to a walked one.

    `scanno/step.py` repeats the arithmetic of ONE node of `classify()`'s loop, and a textual
    copy drifts silently. So the two are not compared by reading; they are RUN over the same
    inputs and compared entry for entry, at every node the walk actually visited — where the
    truth is known — before either is trusted at a node it did not.
    """
    from scanno.step import node_scorer
    res, tree, Z, usable, store, _ = _deep()
    step = node_scorer(Z, usable, tree, store=store)

    n = 0
    for r in res:
        for entry in r["trace"]:
            got = step(r["cluster"], entry["at"])
            assert got is not None, f"the walk scored {entry['at']} and step() would not"
            assert got["at"] == entry["at"]
            assert got["top"] == entry["top"], (entry["at"], got["top"], entry["top"])
            assert got["gap"] == pytest.approx(entry["gap"], abs=1e-12)
            assert _same(got["survival"], entry["survival"])
            assert _same(got["cover"], entry["cover"])
            n += 1
    assert n, "the fixture recorded no trace at all"
    # ... and a node the walk never reached is scored, which is the whole point
    deeper = step(0, res[0]["trace"][-1]["top"])
    assert deeper is not None and deeper["top"] in tree["children"][deeper["at"]]


def test_a_sealed_node_is_a_leaf_and_the_recursion_stops_there():
    """A seal deletes a child set, so the sealed node HAS no children and is a leaf by
    construction. Nothing in the recursion reads the verdicts, and nothing needs to."""
    from scanno.scope import seal_tree
    v = scope(**{"Stromal": "FORCE", "Stromal/Mural": "SEAL"})["nodes"]
    sealed, removed = seal_tree(TREE, v)
    assert removed == {"Stromal/Mural": ["Pericyte", "Smooth muscle"]}

    res = calls("Stromal", trace={0: [at("root", "Stromal", 0.9), at("Stromal", "Mural", 0.05)]})
    # the scorer would happily go on if it were asked; it is not asked, because Mural is a leaf
    scorer = _scorer({"Mural": ("Pericyte", 0.4)})
    forced, rec = apply_force(res, ["Stromal"], tree=sealed, scorer=scorer, counts={0: 99})

    assert forced[0]["path"] == "Stromal/Mural" and forced[0]["depth"] == 2
    assert forced[0]["force_depth"] == 1
    assert internal_terminals(forced, sealed) == []
    # and against the UNSEALED tree the same rows would be a defect - which is what makes the
    # check above a statement about the tree in force rather than about the label string
    assert [t["node"] for t in internal_terminals(forced, TREE)] == ["Stromal/Mural"]


def test_the_number_of_forced_steps_is_recorded_and_a_double_push_is_not_a_single_one():
    """Stacked uncertainty must be visible in the OBJECT. Two decisions the walk did not take
    produce a deeper, more confident-looking name, and nothing else in obs says how it got
    there. The count is not a strength: only the first step is below the bar by construction,
    which is why every step's margin is recorded beside it rather than summarised."""
    import numpy as np
    from scanno.emit import annotate_obs, force_provenance
    A = _obj(4)
    res = calls("Stromal", "Endothelial",
                trace={0: [at("root", "Stromal", 0.9), at("Stromal", "Mural", 0.05)],
                       1: [at("root", "Endothelial", 0.9),
                           at("Endothelial", "Endocardial", 0.02)]})
    forced, rec = apply_force(res, ["Stromal", "Endothelial"], counts={0: 99, 1: 961},
                              tree=TREE, scorer=_scorer({"Mural": ("Pericyte", 0.42)}))

    assert forced[0]["path"] == "Stromal/Mural/Pericyte" and forced[0]["force_depth"] == 2
    assert forced[1]["path"] == "Endothelial/Endocardial" and forced[1]["force_depth"] == 1
    assert rec["clusters_by_force_depth"] == {"2": 1, "1": 1}

    written = annotate_obs(A, forced, np.array([0, 0, 1, 1]), assignment=True)
    assert "scanno_force_depth" in written
    assert list(A.obs["scanno_force_depth"]) == [2, 2, 1, 1]
    assert list(A.obs["scanno_assignment"]) == [BY_FORCE] * 4      # identical, and must not be
    # the two are one statement: the column that says HOW and the column that says HOW FAR
    force_provenance(A, rec)
    a = A.uns["scanno_assignment_provenance"]["assigned"]["0"]
    assert a["steps"] == ["Stromal/Mural", "Stromal/Mural/Pericyte"]
    assert a["margins"] == [pytest.approx(0.05), pytest.approx(0.42)]
    assert a["margin"] == pytest.approx(0.05)      # still the margin AT the FORCE node
    assert a["to"] == "Pericyte" and a["force_depth"] == 2


def test_a_gap_cleared_or_excluded_cell_is_recorded_as_zero_steps_forced():
    """Zero is a measurement here, not a missing value — and `assignment` tells the two apart."""
    import numpy as np
    from scanno.emit import annotate_obs
    A = _obj(6)
    res, _ = apply_force(calls("EXCLUDED", "Cardiomyocyte/Working cardiomyocyte", "Stromal",
                               trace={2: [at("root", "Stromal", .9), at("Stromal", "Mural", .05)]}),
                         ["Stromal"], tree=TREE,
                         scorer=_scorer({"Mural": ("Smooth muscle", 0.3)}))
    annotate_obs(A, res, np.array([0, 0, 1, 1, 2, 2]), assignment=True)
    assert list(A.obs["scanno_force_depth"]) == [0, 0, 0, 0, 2, 2]
    assert list(A.obs["scanno_assignment"]) == [EXCLUDED, EXCLUDED, BY_GAP, BY_GAP,
                                                BY_FORCE, BY_FORCE]


def test_a_node_whose_children_cannot_be_scored_is_recorded_rather_than_invented():
    """`node_scorer` returns None where the walk would have broken. There is no measured child
    then, and a half-applied push would leave the cell on the internal node — the defect."""
    res = calls("Stromal", trace={0: [at("root", "Stromal", 0.9), at("Stromal", "Mural", 0.05)]})
    forced, rec = apply_force(res, ["Stromal"], counts={0: 99}, tree=TREE, scorer=_scorer({}))

    assert forced[0]["path"] == "Stromal"          # UNTOUCHED: not moved half way
    assert "force_depth" not in forced[0]
    assert rec["n_forced"] == 0 and rec["n_cells_forced"] == 0
    u = rec["unforceable"]["0"]
    assert u["node"] == "Stromal" and u["reached"] == "Stromal/Mural"
    assert u["n_steps"] == 1 and u["margins"] == [pytest.approx(0.05)]
    assert "cannot be scored" in u["reason"]
    assert u["n_cells"] == 99
    # the post-condition still fires, which is what makes the CLI refuse rather than write
    assert bare_force(forced, ["Stromal"]) != []
    assert "REFUSE" in "\n".join(format_force(rec))


def test_a_tree_that_loops_stops_the_recursion_instead_of_running_forever():
    """A cycle is a taxonomy defect, not a reason to hang. It is named and the row is left be."""
    loop = {"children": {"root": ["Alpha"], "Alpha": ["Beta"], "Beta": ["Alpha"]},
            "patterns": {}, "members": {}}
    res = calls("Alpha", trace={0: [at("root", "Alpha", 0.9), at("Alpha", "Beta", 0.05)]})
    forced, rec = apply_force(res, ["Alpha"], tree=loop,
                              scorer=_scorer({"Beta": ("Alpha", 0.4)}))

    assert forced[0]["path"] == "Alpha" and rec["n_forced"] == 0
    assert "already passed through" in rec["unforceable"]["0"]["reason"]
    assert rec["unforceable"]["0"]["reached"] == "Alpha/Beta"


def test_the_recursion_assumes_no_particular_depth():
    """Three levels is this cohort's tree, not the tool's. A four-step chain must work the same."""
    chain = {"children": {"root": ["N1", "X"], "N1": ["N2", "Y"], "N2": ["N3", "Z"],
                          "N3": ["N4", "W"]},
             "patterns": {}, "members": {}}
    res = calls("N1", trace={0: [at("root", "N1", 0.9), at("N1", "N2", 0.05)]})
    forced, rec = apply_force(res, ["N1"], tree=chain,
                              scorer=_scorer({"N2": ("N3", 0.06), "N3": ("N4", 0.07)}))

    assert forced[0]["path"] == "N1/N2/N3/N4"
    assert forced[0]["depth"] == 4 and forced[0]["force_depth"] == 3
    assert rec["assigned"]["0"]["margins"] == [pytest.approx(x) for x in (0.05, 0.06, 0.07)]
    assert internal_terminals(forced, chain) == []


def test_a_child_that_is_already_a_leaf_is_pushed_once_exactly_as_before():
    """NO REGRESSION. Where the destination is a leaf, recursion must change nothing at all —
    same path, same depth, same statistics, same record — whether or not a scorer is present."""
    res = calls("Endothelial",
                trace={0: [at("root", "Endothelial", .9), at("Endothelial", "Endocardial", .02)]})
    old, rec_old = apply_force(res, ["Endothelial"], counts={0: 961})
    new, rec_new = apply_force(res, ["Endothelial"], counts={0: 961}, tree=TREE,
                               scorer=_scorer({"Endocardial": ("Never asked", 0.9)}))

    assert new[0]["path"] == old[0]["path"] == "Endothelial/Endocardial"
    assert new[0]["depth"] == old[0]["depth"] == 2
    assert new[0] == old[0] and old[0]["force_depth"] == 1
    assert rec_new["assigned"] == rec_old["assigned"]
    assert rec_new["by_node"] == rec_old["by_node"]


def test_without_a_tree_the_push_is_single_level_because_leafness_is_unknowable():
    """NO REGRESSION for a caller holding only the walk's output. `Mural` is internal in TREE and
    the push still stops there, because without a tree nothing in scope KNOWS that."""
    res = calls("Stromal", trace={0: [at("root", "Stromal", .9), at("Stromal", "Mural", .05)]})
    forced, rec = apply_force(res, ["Stromal"], counts={0: 99})
    assert forced[0]["path"] == "Stromal/Mural" and forced[0]["force_depth"] == 1
    assert rec["unforceable"] == {} and rec["recursive"] is False


def test_a_tree_without_a_scorer_refuses_rather_than_stopping_on_an_internal_node():
    """The one combination that must NOT silently do half the job: leafness knowable, the next
    child not measurable. Guessing there is the same invention the walk refuses to make."""
    res = calls("Stromal", trace={0: [at("root", "Stromal", .9), at("Stromal", "Mural", .05)]})
    forced, rec = apply_force(res, ["Stromal"], counts={0: 99}, tree=TREE)
    assert forced[0]["path"] == "Stromal" and rec["n_forced"] == 0
    assert "no scorer was given" in rec["unforceable"]["0"]["reason"]


def test_push_to_leaf_reports_the_chain_it_took_and_not_only_where_it_ended():
    """Read on its own, because the CLI's refusal message and the provenance both need the chain
    — a line reading `A -> C` lets a reader believe one measurement put the cell there."""
    steps, refusal = push_to_leaf(
        0, "Stromal", at("Stromal", "Mural", 0.05), tree=TREE,
        scorer=_scorer({"Mural": ("Pericyte", 0.42)}))
    assert refusal == ""
    assert [s["path"] for s in steps] == ["Stromal/Mural", "Stromal/Mural/Pericyte"]
    assert [s["to"] for s in steps] == ["Mural", "Pericyte"]
    assert [s["margin"] for s in steps] == [pytest.approx(0.05), pytest.approx(0.42)]


def test_the_printed_reassignment_shows_every_step_and_says_when_there_was_more_than_one():
    res = calls("Stromal", trace={0: [at("root", "Stromal", .9), at("Stromal", "Mural", .05)]})
    _, rec = apply_force(res, ["Stromal"], counts={0: 99}, tree=TREE,
                         scorer=_scorer({"Mural": ("Pericyte", 0.42)}))
    text = "\n".join(format_force(rec, gap_min=0.30))
    assert "Stromal -> Mural -> Pericyte" in text
    assert "0.050, 0.420" in text
    assert "MORE THAN ONE step" in text
    assert "force_depth" in text
    # the WEAKEST step, named — a later step may clear the bar the first one failed, and a line
    # calling every step sub-threshold would be as wrong as calling none of them that
    assert "weakest step 0.050" in text
    assert "may" in text and "either side of the bar" in text


def test_internal_terminals_separates_a_broken_recursion_from_a_split_the_scope_left_open():
    """Both are cells sitting on an internal node and their remedies have nothing in common: one
    is a bug in the push, the other is the walk truncating where the scope said it may."""
    res = calls("Cardiomyocyte", "Stromal/Fibroblast",
                trace={0: [at("root", "Cardiomyocyte", .9),
                           at("Cardiomyocyte", "Working cardiomyocyte", .01)]})
    out, _ = apply_force(res, ["Stromal"], tree=TREE)       # Cardiomyocyte is NOT forced
    stopped = internal_terminals(out, TREE)
    assert [t["node"] for t in stopped] == ["Cardiomyocyte"]
    assert stopped[0]["assignment"] == BY_GAP
    assert stopped[0]["children"] == "Working cardiomyocyte, Conduction cardiomyocyte"
    assert [t for t in stopped if t["assignment"] == BY_FORCE] == []


def test_the_cli_refuses_a_forced_cluster_that_stopped_on_an_internal_node():
    """AST, not behaviour: the post-condition must be READ off the finished rows in `_annotate`,
    and a `forced` row on any internal node must reach the same refusal as a bare FORCE name."""
    src = (ROOT / "scanno" / "cli.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_annotate")
    body = ast.unparse(fn)
    assert body.count("internal_terminals(") == 1
    assert body.count("bare_force(") == 1
    assert "node_scorer(" in body, "the recursion needs a real scorer, not trace[-1] twice"
    assert _stmt("stuck += [t for t in inner "
                 "if t['assignment'] == BY_FORCE and t['cluster'] not in seen]") in body
    assert "if stuck:" in body and "return REFUSE" in body


def test_the_scorer_holds_no_bar_and_does_not_walk():
    """`scanno/step.py` computes ONE node. The loop, the truncation rule and the gap bar stay in
    classify.py — a second place that could decide to descend is a second classifier."""
    src = (ROOT / "scanno" / "step.py").read_text(encoding="utf-8")
    mod = ast.parse(src)
    fn = next(n for n in ast.walk(mod) if isinstance(n, ast.FunctionDef) and n.name == "step")
    assert not [n for n in ast.walk(fn) if isinstance(n, (ast.While, ast.For))], "it walks"
    # CODE, not prose: the docstring is allowed to explain what FORCE needs this for.
    for word in ("gap_min", "GAP_PROFILE", "GAP_CORPUS", "FORCE", "SEAL", "descend"):
        assert word not in ast.unparse(fn), f"step.py's step() mentions {word!r}"
    floats = {n.value for n in ast.walk(mod)
              if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    assert floats == {1.0}, floats          # the spread guard classify.py uses, and nothing else


# ------------------------------------------------------------------------------- helpers

def _obj(n):
    import anndata as ad
    import numpy as np
    return ad.AnnData(X=np.zeros((n, 3), dtype=np.float32))


def _annotate_parser():
    """{option: {keyword names}} for every argument the `annotate` subparser declares.

    Located by SOURCE LINE between its own `add_parser` and the next one. `ast.walk` yields
    breadth-first, so a flag that tracked "are we inside annotate yet" over that order would
    collect arguments from whichever subparser happened to sit at the same depth.
    """
    mod = ast.parse((ROOT / "scanno" / "cli.py").read_text(encoding="utf-8"))
    parsers, args = [], []
    for n in ast.walk(mod):
        if not isinstance(n, ast.Call):
            continue
        attr = getattr(n.func, "attr", "")
        if attr == "add_parser" and n.args and isinstance(n.args[0], ast.Constant):
            parsers.append((n.lineno, n.args[0].value))
        elif attr == "add_argument" and n.args and isinstance(n.args[0], ast.Constant):
            args.append((n.lineno, n.args[0].value, {k.arg for k in n.keywords}))
    parsers.sort()
    start = next(ln for ln, name in parsers if name == "annotate")
    end = min((ln for ln, _ in parsers if ln > start), default=float("inf"))
    return {opt: kw for ln, opt, kw in args if start < ln < end}


def _same(a, b):
    """Equal, with NaN equal to NaN — `survival` is NaN on the profile path and must compare."""
    import math
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b)


def _stmt(src):
    """One statement, rendered by the SAME unparser the test reads the source with.

    A hand-typed expectation compared against `ast.unparse` output is an assertion about
    CPython's pretty-printer, not about scanno. CPython 3.11 stopped parenthesising a tuple
    ASSIGNMENT TARGET, so `"(scope, force_paths) = (None, [])"` matched on 3.9, matched
    nothing on 3.12, and reported the strongest guarantee in this file — that absent
    `--scope` the annotate path is byte-identical — as BROKEN on the only interpreter the
    pipeline actually runs on, while `cli.py` was correct throughout. The wrong half of that
    is the one that would have been "fixed".

    Round-tripping the expectation through the same unparser removes the interpreter version
    from the assertion and removes nothing else: the string still has to be present, still
    has to be that exact statement, and a substring that merely resembles it still fails.
    """
    return ast.unparse(ast.parse(src))


def _code_strings(path):
    """Every string CONSTANT that is not a docstring. Prose may cite a cohort; code may not."""
    mod = ast.parse(Path(path).read_text(encoding="utf-8"))
    doc = set()
    for n in ast.walk(mod):
        body = getattr(n, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            doc.add(id(body[0].value))
    return [n.value for n in ast.walk(mod)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc]


if __name__ == "__main__":
    ok, bad, skipped = 0, [], []
    for name, fn in sorted(dict(globals()).items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except ImportError as e:
            print(f"  SKIP  {name}   {e}")
            skipped.append(name)
            continue
        except AssertionError as e:
            print(f"  FAIL  {name}   {e}")
            bad.append(name)
            continue
        except Exception as e:                                            # noqa: BLE001
            print(f"  FAIL  {name}   {type(e).__name__}: {e}")
            bad.append(name)
            continue
        print(f"  PASS  {name}")
        ok += 1
    print(f"\n{ok} passed, {len(bad)} failed, {len(skipped)} skipped")
    if bad:
        print("failures: " + ", ".join(bad))
    raise SystemExit(1 if bad else 0)
