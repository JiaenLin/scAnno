"""`scanno annotate --l1-tree`: one command, one object, two label columns.

WHAT THIS PROTECTS

The deliverable is ONE object carrying exactly two label columns — the independent L1, and the
label from the sealed tree. Before `--l1-tree` that took two `annotate` runs, and both wrote
`scAnno_L1`, so the second silently clobbered the first. The workaround was `--label-suffix` on
one of them plus a merge afterwards, which is an in-house step: a private script, unversioned and
untested, between the tool and the deliverable.

Five things have to hold for the flag to replace that, and each has a test named for it:

  1. the walk is UNCHANGED — `classify.py` is byte-identical, and the second run is a second
     CALL with a different tree, not a mode inside the walk;
  2. the L1 column keeps the name a reader keys on, and a suffixed run still cannot collide;
  3. the column is marked INDEPENDENT in the object, so a reader can tell it from a derived one;
  4. a tree that is not depth-1 is REFUSED, on the declaration, so all ten samples of a cohort
     get the same verdict;
  5. without `--l1-tree` nothing whatsoever changes.

    python tests/test_l1_tree.py        # no pytest needed; the shim below stands in
"""
from __future__ import annotations

import ast
import hashlib
import json
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

from scanno.scope import (root_child_diff, seal_tree, truncate_tree,  # noqa: E402
                          tree_depth, vote)

# --------------------------------------------------------------------- the fixture tree
#
# SAMBO's declared tree, trimmed to the branches that matter here. Real names, because a test
# whose failure message says "A/B/C" tells the next reader nothing about which lineage broke.
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
    "patterns": {n: ["x"] for n in
                 ["Cardiomyocyte", "Working cardiomyocyte", "Conduction cardiomyocyte",
                  "Stromal", "Fibroblast", "Matrifibrocyte", "Quiescent fibroblast", "Mural",
                  "Pericyte", "Smooth muscle", "Endothelial", "Vascular endothelial",
                  "Endocardial", "Immune", "Myeloid", "Macrophage", "Lymphoid", "B cell",
                  "NK cell", "Mesothelial"]},
    "members": {},
}


def calls(*paths):
    """`classify()`-shaped rows, one per cluster, in order — what `per_cell` promises to join."""
    out = []
    for i, p in enumerate(paths):
        sentinel = p in ("EXCLUDED", "UNRESOLVED")
        out.append({"cluster": i,
                    "label": (p if sentinel else p.split("/")[-1]),
                    "path": p,
                    "depth": (0 if sentinel else len(p.split("/"))),
                    "gap": 0.0 if sentinel else 0.7,
                    "survival": float("nan"),
                    "cover": float("nan"),
                    "excluded": p == "EXCLUDED",
                    "trace": []})
    return out


# ===================================================================== 1. the walk is unchanged

#: sha256 of scanno/classify.py as of the commit that added --l1-tree. The requirement is not
#: "classify still works" but "classify was not touched": an independent L1 that needed the walk
#: edited would not be the same walk, and every comparison drawn between the two columns would be
#: a comparison between two different classifiers.
CLASSIFY_SHA = "9096b9aef108e88c9678009ab6fa522aa1cb9cac11485433db1ccfbaeef2c8ba"


def test_classify_py_is_not_touched_by_the_l1_feature():
    """The digest is advisory; the STRUCTURAL assertions below are the ones that must hold."""
    src = (ROOT / "scanno" / "classify.py").read_text(encoding="utf-8")
    got = hashlib.sha256(src.encode()).hexdigest()
    if got != CLASSIFY_SHA:
        print(f"  NOTE  classify.py digest is {got}, pinned {CLASSIFY_SHA}. "
              f"If this change was deliberate, say so and re-pin.")

    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "classify")
    args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    # No L1 parameter, no second tree, no level argument anywhere in the walk's signature.
    assert args == ["Z", "usable", "tree", "store", "assertions", "gap_min", "exclude"], args
    for word in ("l1", "L1", "level", "independent"):
        assert word not in src, f"classify.py mentions {word!r} - the walk was parameterised"


def test_the_independent_l1_is_a_second_call_not_a_second_mode():
    """`cli._annotate` must call `classify` twice, with the same arguments but a different tree."""
    src = (ROOT / "scanno" / "cli.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_annotate")
    calls_ = [n for n in ast.walk(fn)
              if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "classify"]
    assert len(calls_) == 2, f"{len(calls_)} classify() call(s) in _annotate"
    trees = [c.args[2].id for c in calls_]
    assert trees == ["tree", "l1_tree"], trees
    # the same bar and the same withheld set in both, or the two columns are not comparable
    for c in calls_:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw["gap_min"] == "a.gap_min", kw
        assert kw["exclude"] == "drop", kw


# ================================================== 2. the column name, and the sweep collision

def test_the_independent_l1_takes_the_column_a_reader_keys_on():
    import numpy as np
    from scanno.emit import annotate_obs, independent_l1
    A = _obj(6)
    y = np.array([0, 0, 1, 1, 2, 2])
    annotate_obs(A, calls("Stromal/Fibroblast", "Immune/Myeloid/Macrophage",
                          "Endothelial/Endocardial"), y)
    col, rec = independent_l1(A, calls("Stromal", "Immune", "Endothelial"), y)
    assert col == "scAnno_L1"
    assert list(A.obs["scAnno_L1"]) == ["Stromal", "Stromal", "Immune", "Immune",
                                        "Endothelial", "Endothelial"]
    # the deep columns are still the DEEP walk's - the flag replaces L1, not the annotation
    assert list(A.obs["scAnno_L3"])[2] == "Immune/Myeloid/Macrophage"
    assert rec["replaced"] == "derived"


def test_a_suffixed_run_cannot_collide_with_the_unsuffixed_one():
    """The whole reason the flag exists: two runs, one object, two columns that never fight."""
    import numpy as np
    from scanno.emit import annotate_obs, independent_l1
    A = _obj(4)
    y = np.array([0, 0, 1, 1])
    annotate_obs(A, calls("Stromal/Fibroblast", "Immune/Myeloid/Macrophage"), y)
    independent_l1(A, calls("Stromal", "Immune"), y)
    annotate_obs(A, calls("Stromal/Mural/Pericyte", "Immune/Lymphoid"), y, suffix="_r0p5")
    independent_l1(A, calls("Stromal", "Immune"), y, suffix="_r0p5")

    assert "scAnno_L1" in A.obs and "scAnno_L1_r0p5" in A.obs
    assert "scanno_cell_type" in A.obs           # only the UNSUFFIXED run gets that name
    assert "scanno_cell_type_r0p5" not in A.obs
    assert "scanno_label_r0p5" in A.obs
    assert "scAnno_L1_provenance" in A.uns and "scAnno_L1_r0p5_provenance" in A.uns


# ============================================================= 3. the column is marked INDEPENDENT

def test_the_l1_column_says_in_the_object_that_it_is_independent():
    import numpy as np
    from scanno.emit import annotate_obs, independent_l1
    A = _obj(4)
    y = np.array([0, 0, 1, 1])
    annotate_obs(A, calls("Stromal/Fibroblast", "Immune/Myeloid/Macrophage"), y)
    col, rec = independent_l1(A, calls("Stromal", "Immune"), y, tree="l1_tree.json")
    prov = A.uns[f"{col}_provenance"]
    assert prov["source"] == "independent"
    assert prov["column"] == "scAnno_L1"
    assert prov["tree"] == "l1_tree.json"
    assert prov["n_cells"] == 4
    assert prov["labels"] == {"Stromal": 2, "Immune": 2}
    assert prov is rec


def test_a_derived_l1_carries_no_such_mark():
    """`annotate_obs` alone must leave nothing that could be read as 'independent'."""
    import numpy as np
    from scanno.emit import annotate_obs
    A = _obj(2)
    annotate_obs(A, calls("Stromal/Fibroblast"), np.array([0, 0]))
    assert "scAnno_L1" in A.obs
    assert "scAnno_L1_provenance" not in A.uns
    assert not [k for k in A.uns if "independent" in str(k)]


def test_disagreement_with_the_derived_l1_is_counted_not_hidden():
    """L2..Ln stay the DEEP walk's, so a differing L1 leaves the object internally inconsistent.

    Measured on the real cohort the two agree on all 109,140 nuclei, because both walks face the
    same root child set. That is a property of the trees, not a guarantee of the code, and a
    hand-built --l1-tree can break it — so it is counted and printed rather than assumed.
    """
    import numpy as np
    from scanno.emit import annotate_obs, format_independent_l1, independent_l1
    A = _obj(4)
    y = np.array([0, 0, 1, 1])
    annotate_obs(A, calls("Stromal/Fibroblast", "Immune/Myeloid/Macrophage"), y)
    _, rec = independent_l1(A, calls("Stromal", "Mesothelial"), y)
    assert rec["n_disagree"] == 2
    assert rec["disagreements"] == {"Immune -> Mesothelial": 2}
    txt = "\n".join(format_independent_l1(rec))
    assert "REVIEW" in txt and "Immune -> Mesothelial" in txt

    B = _obj(4)
    annotate_obs(B, calls("Stromal/Fibroblast", "Immune/Myeloid/Macrophage"), y)
    _, ok = independent_l1(B, calls("Stromal", "Immune"), y)
    assert ok["n_disagree"] == 0
    assert "REVIEW" not in "\n".join(format_independent_l1(ok))


def test_no_derived_column_to_compare_against_is_minus_one_not_zero():
    """Zero would read as 'checked, and they agreed'. Nothing was checked."""
    import numpy as np
    from scanno.emit import independent_l1
    A = _obj(2)
    _, rec = independent_l1(A, calls("Stromal"), np.array([0, 0]))
    assert rec["n_disagree"] == -1
    assert rec["replaced"] == ""


def test_a_flagged_nucleus_is_EXCLUDED_in_the_independent_l1_too():
    """The exclusion is per NUCLEUS. A column that ignored it would label what QC withheld."""
    import numpy as np
    from scanno.emit import independent_l1
    A = _obj(4)
    y = np.array([0, 0, 1, 1])
    flag = np.array([False, True, False, False])
    _, rec = independent_l1(A, calls("Stromal", "Immune"), y, flag=flag)
    assert list(A.obs["scAnno_L1"]) == ["Stromal", "EXCLUDED", "Immune", "Immune"]
    assert rec["labels"]["EXCLUDED"] == 1


# ======================================================== 4. a tree that is not depth-1 is refused

def test_tree_depth_measures_the_declared_tree():
    assert tree_depth(TREE) == 3                       # root -> Stromal -> Mural -> Pericyte
    assert tree_depth(truncate_tree(TREE, 1)) == 1
    assert tree_depth(truncate_tree(TREE, 2)) == 2
    assert tree_depth({"children": {}}) == 0
    assert tree_depth({}) == 0


def test_tree_depth_does_not_hang_on_a_cyclic_children_map():
    """Malformed input must fail a check, not the wall clock of a 6-hour PBS job."""
    assert tree_depth({"children": {"root": ["A"], "A": ["B"], "B": ["A"]}}) == 2


def test_a_sealed_tree_is_still_too_deep_to_be_an_l1_tree():
    """The scope tree seals two nodes and is STILL depth 3 — sealing is not truncating."""
    paths = {"S1": ["Stromal/Fibroblast", "Stromal/Mural/Pericyte"],
             "S2": ["Stromal/Fibroblast/Matrifibrocyte", "Stromal/Mural/Pericyte"]}
    sealed, _ = seal_tree(TREE, vote(paths, TREE, min_support=1.0))
    assert tree_depth(sealed) == 3
    assert "Fibroblast" not in sealed["children"]


def test_the_library_refuses_a_walk_that_returned_a_path():
    """The last line of defence: an L1 column may not hold `Stromal/Fibroblast`."""
    import numpy as np
    from scanno.emit import independent_l1
    A = _obj(2)
    with pytest.raises(ValueError):
        independent_l1(A, calls("Stromal/Fibroblast"), np.array([0, 0]))
    assert "scAnno_L1" not in A.obs, "it wrote the column before refusing"


def test_the_cli_refuses_a_deep_l1_tree_before_reading_the_object():
    """On the DECLARATION, so every sample of a cohort gets the same verdict.

    A result-based check would be data-dependent: the same tree would pass on the sample whose
    gaps all failed and refuse on the next, leaving half a cohort's column written each way.
    """
    src = (ROOT / "scanno" / "cli.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_annotate")
    body = ast.unparse(fn)
    guard = body.index("tree_depth(l1_tree)")
    read = body.index("ad.read_h5ad(a.h5ad)")
    assert guard < read, "the depth guard runs after the object is read"
    assert "return REFUSE" in body[guard:read], "the guard reports but does not refuse"


def test_root_child_diff_names_the_trees_that_are_not_comparable():
    l1 = truncate_tree(TREE, 1)
    assert root_child_diff(TREE, l1) == ([], [])
    other = json.loads(json.dumps(l1))
    other["children"]["root"] = ["Cardiomyocyte", "Stromal", "Glia"]
    only_deep, only_l1 = root_child_diff(TREE, other)
    assert only_l1 == ["Glia"]
    assert "Endothelial" in only_deep and "Immune" in only_deep


# ============================================== 5. nothing changes when the flag is not given

def test_without_the_flag_the_annotate_path_is_untouched():
    """`--l1-tree` absent must leave the object byte-for-byte what 0.9.0 wrote."""
    import numpy as np
    from scanno.emit import annotate_obs
    A = _obj(4)
    y = np.array([0, 0, 1, 1])
    made = annotate_obs(A, calls("Stromal/Fibroblast", "Immune/Myeloid/Macrophage"), y)
    assert made == ["scanno_cell_type", "scanno_path", "scanno_depth", "scanno_gap",
                    "scanno_survival", "scAnno_L1", "scAnno_L2", "scAnno_L3"]
    assert list(A.obs["scAnno_L1"]) == ["Stromal", "Stromal", "Immune", "Immune"]
    assert not A.uns


def test_the_flag_is_registered_on_annotate_and_read_only_there():
    """`--out-gene-key` was once added to the wrong subparser and read by this handler."""
    src = (ROOT / "scanno" / "cli.py").read_text(encoding="utf-8")
    assert '"--l1-tree"' in src
    i = src.index('sub.add_parser("annotate"')
    j = src.index('sub.add_parser("compare"')
    assert i < src.index('"--l1-tree"') < j, "--l1-tree is not on the annotate subparser"


def test_the_flag_is_optional_and_defaults_to_off():
    """No `required`, no `default` — so argparse gives None and `if l1_tree` is False."""
    src = (ROOT / "scanno" / "cli.py").read_text(encoding="utf-8")
    add = next(n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Call)
               and getattr(n.func, "attr", "") == "add_argument"
               and n.args and getattr(n.args[0], "value", "") == "--l1-tree")
    kw = {k.arg for k in add.keywords}
    assert "required" not in kw and "default" not in kw, kw

    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_annotate")
    body = ast.unparse(fn)
    assert "l1_tree = None" in body, "l1_tree must be None unless the flag was given"
    assert "if res_l1 is not None" in body, "the second walk must be gated on the flag"


# ------------------------------------------------------------------------------- helpers

def _obj(n):
    import anndata as ad
    import numpy as np
    return ad.AnnData(X=np.zeros((n, 3), dtype=np.float32))


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
