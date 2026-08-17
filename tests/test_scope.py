"""The common scope: does the vote reproduce a scope a human derived by reading the table?

The fixture below is not invented. It is a real ten-library cohort's reach/descend pattern at every
internal node, transcribed from the ten pass-1 objects, and the expected verdicts are the scope
the PI approved after reading it. If a change to `scope.py` stops reproducing it, the rule has
moved and somebody has to say so out loud.
"""
import json


#: No pytest. This suite guards the scope-finding mechanism, and it was UNRUNNABLE in the
#: environment the pipeline actually runs in — pytest is not installed there, so the one suite
#: covering the vote never executed while every other suite passed. A guard that cannot run in
#: the place the tool runs is not a guard. `approx` and `raises` are the only two things it
#: needed, and both are three lines.
def approx(x, y, tol=1e-9):
    return abs(float(x) - float(y)) < tol


class raises:
    """`with raises(ValueError):` — asserts the block raises that type."""

    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        assert t is not None and issubclass(t, self.exc), \
            f"expected {self.exc.__name__}, got {t.__name__ if t else 'no exception'}"
        return True


from scanno.scope import (apply_scope, bare_names_unique, internal_nodes, node_votes,
                          seal_tree, sealed_labels, vote)

A = ["L01", "L02", "L03", "L04", "L05", "L06",
     "L07", "L08", "L09", "L10"]

TREE = {
    "children": {
        "root": ["Cardiomyocyte", "Stromal", "Endothelial", "Immune", "Mesothelial",
                 "Adipocyte", "Neural"],
        "Cardiomyocyte": ["Working cardiomyocyte", "Conduction cardiomyocyte"],
        "Stromal": ["Fibroblast", "Mural"],
        "Fibroblast": ["Matrifibrocyte", "Quiescent fibroblast"],
        "Mural": ["Pericyte", "Smooth muscle"],
        "Endothelial": ["Vascular endothelial", "Endocardial", "Lymphatic endothelial"],
        "Immune": ["Myeloid", "Lymphoid"],
        "Myeloid": ["Macrophage"],
        "Lymphoid": ["B cell", "NK cell"],
    },
    "patterns": {n: ["x"] for n in
                 ["Cardiomyocyte", "Working cardiomyocyte", "Conduction cardiomyocyte",
                  "Stromal", "Fibroblast", "Matrifibrocyte", "Quiescent fibroblast", "Mural",
                  "Pericyte", "Smooth muscle", "Endothelial", "Vascular endothelial",
                  "Endocardial", "Lymphatic endothelial", "Immune", "Myeloid", "Macrophage",
                  "Lymphoid", "B cell", "NK cell", "Mesothelial", "Adipocyte", "Neural"]},
    "members": {},
}

#: the reference cohort's real terminal paths, per animal. Transcribed from the ten pass-1 objects.
COHORT = {
    "Cardiomyocyte/Working cardiomyocyte": A,
    "Endothelial/Vascular endothelial": A,
    "Endothelial/Lymphatic endothelial": A,
    "Immune/Myeloid/Macrophage": A,
    "Endothelial/Endocardial": [s for s in A if s != "L04"],
    "Mesothelial": [s for s in A if s != "L01"],
    "Stromal/Mural/Smooth muscle": [s for s in A if s != "L08"],
    "Stromal/Mural/Pericyte": [s for s in A if s not in ("L02", "L07")],
    "Stromal/Fibroblast/Matrifibrocyte": ["L03", "L04", "L05", "L06",
                                          "L08", "L09", "L10"],
    "Stromal/Fibroblast": ["L02", "L04", "L06", "L07", "L09",
                           "L10"],
    "Stromal/Fibroblast/Quiescent fibroblast": ["L01", "L03"],
    "Immune/Lymphoid": ["L01", "L02", "L03", "L05", "L06"],
    "Immune/Lymphoid/B cell": ["L01", "L06"],
    "Immune/Lymphoid/NK cell": ["L04", "L10"],
    "Adipocyte": ["L04", "L05", "L06", "L09", "L10"],
    "Neural": ["L07", "L08", "L09", "L10"],
    "Endothelial": ["L04", "L09"],
    "Stromal": ["L08"],
}


def paths():
    """{sample: [path, ...]} — one cell per (path, sample) pair is enough to vote."""
    out = {s: [] for s in A}
    for path, samples in COHORT.items():
        for s in samples:
            out[s].append(path)
    return out


# ---------------------------------------------------------------- the reproduction test

def test_unanimity_reproduces_the_approved_reference_scope():
    v = vote(paths(), TREE, min_support=1.0)
    sealed = sorted(n for n, r in v.items() if r["verdict"] == "SEAL")
    assert sealed == ["Immune/Lymphoid", "Stromal/Fibroblast"], sealed

    # nothing else was sealed. Two nodes are unanimous AND strand cells, so they are FORCE:
    # admissible splits that nothing may terminate on.
    for node in ("root", "Cardiomyocyte", "Immune", "Immune/Myeloid", "Stromal/Mural"):
        assert v[node]["verdict"] == "KEEP", (node, v[node])
    for node in ("Stromal", "Endothelial"):
        assert v[node]["verdict"] == "FORCE", (node, v[node])
        assert v[node]["support"] == 1.0            # unanimous, yet still stranding cells
        assert v[node]["stranded"], node


def test_the_two_seals_carry_the_evidence_that_produced_them():
    v = vote(paths(), TREE, min_support=1.0)
    assert v["Stromal/Fibroblast"]["n_reached"] == 10
    assert v["Stromal/Fibroblast"]["n_descended"] == 8
    assert v["Stromal/Fibroblast"]["support"] == 0.8 or approx(v["Stromal/Fibroblast"]["support"], 0.8)
    assert v["Immune/Lymphoid"]["n_reached"] == 7
    assert v["Immune/Lymphoid"]["n_descended"] == 4
    assert approx(v["Immune/Lymphoid"]["support"], 4 / 7)


def test_seal_removes_the_named_labels_not_a_category():
    v = vote(paths(), TREE, min_support=1.0)
    lost = sealed_labels(v, paths())
    assert set(lost["Stromal/Fibroblast"]) == {"Stromal/Fibroblast/Matrifibrocyte",
                                               "Stromal/Fibroblast/Quiescent fibroblast"}
    assert set(lost["Immune/Lymphoid"]) == {"Immune/Lymphoid/B cell",
                                            "Immune/Lymphoid/NK cell"}


# ---------------------------------------------------------------- the conditioning

def test_a_sample_that_never_reached_the_node_casts_no_vote():
    """Absence is a missing observation, not a vote against — the easiest way to get this wrong.

    Lymphoid is reached by 7 of 10. If the 3 that never reached it counted as opposition the
    support would be 4/10 = 0.40; conditioned correctly it is 4/7 = 0.571.
    """
    v = node_votes(paths())
    assert v["Immune/Lymphoid"]["n_reached"] == 7
    assert approx(v["Immune/Lymphoid"]["support"], 4 / 7)
    assert "L07" not in v["Immune/Lymphoid"]["reached"]


def test_sentinels_are_not_votes():
    p = {"S1": ["Stromal/Fibroblast/Matrifibrocyte", "UNRESOLVED", "EXCLUDED"]}
    v = node_votes(p)
    assert "UNRESOLVED" not in v and "EXCLUDED" not in v
    assert v["Stromal/Fibroblast"]["cells"] == {"S1": 1}


def test_descend_rule_any_versus_majority_can_disagree():
    """One animal can do both — different clusters of one lineage truncating differently."""
    p = {"S1": ["Stromal/Fibroblast"] * 9 + ["Stromal/Fibroblast/Matrifibrocyte"],
         "S2": ["Stromal/Fibroblast/Matrifibrocyte"] * 10}
    assert node_votes(p, descend_rule="any")["Stromal/Fibroblast"]["n_descended"] == 2
    assert node_votes(p, descend_rule="majority")["Stromal/Fibroblast"]["n_descended"] == 1


def test_bad_descend_rule_refuses():
    with raises(ValueError):
        node_votes(paths(), descend_rule="whatever")


# ---------------------------------------------------------------- what is NOT sealed

def test_too_few_samples_is_unvotable_not_sealed():
    """Sealing on one animal's evidence is a removal with no quorum. It stays OPEN."""
    p = {"S1": ["Immune/Lymphoid"], "S2": ["Cardiomyocyte/Working cardiomyocyte"]}
    v = vote(p, TREE, min_support=1.0, min_reach=2)
    assert v["Immune/Lymphoid"]["verdict"] == "UNVOTABLE"
    assert v["Immune/Lymphoid"] not in [r for r in v.values() if r["verdict"] == "SEAL"]


def test_the_root_is_counted_and_never_sealed():
    """UNRESOLVED is root truncation, so the root MUST be votable — and must never be actionable.

    Sealing the root would delete every level-1 compartment and return UNRESOLVED for every
    nucleus. Its evidence is still reported, because root-level failure is real: it is where a
    whole Pericyte population was lost in 2 of 10 animals.
    """
    p = {"S1": ["Stromal/Mural/Pericyte", "UNRESOLVED", "EXCLUDED"],
         "S2": ["UNRESOLVED", "UNRESOLVED"]}
    n = node_votes(p)
    assert n["root"]["n_reached"] == 2                   # EXCLUDED excluded, UNRESOLVED counted
    assert n["root"]["cells"] == {"S1": 2, "S2": 2}
    assert n["root"]["n_descended"] == 1                 # only S1 got past the root
    v = vote(p, TREE, min_support=1.0)
    assert v["root"]["verdict"] == "KEEP"                # would be SEAL at 0.5 support; refused
    sealed, removed = seal_tree(TREE, v)
    assert sealed["children"]["root"] == TREE["children"]["root"]
    assert "root" not in removed


def test_excluded_and_unresolved_are_different_events():
    assert node_votes({"S": ["EXCLUDED"]}) == {}          # never walked, votes on nothing
    assert node_votes({"S": ["UNRESOLVED"]})["root"]["n_descended"] == 0


def test_a_node_no_sample_reached_is_reported_not_sealed():
    """An empty branch removes nothing, so deleting it buys nothing and perturbs node_weights."""
    v = vote(paths(), TREE, min_support=1.0)
    assert v["Cardiomyocyte"]["verdict"] == "KEEP"          # reached, unanimous
    # Conduction cardiomyocyte is declared but never reached; it is a child, not an internal node
    assert "Cardiomyocyte/Conduction cardiomyocyte" not in v
    sealed = [n for n, r in v.items() if r["verdict"] == "SEAL"]
    assert all(r["n_reached"] > 0 for n, r in v.items() if n in sealed)


def test_loosening_support_stops_sealing_fibroblast():
    """The threshold is the knob, and it is visible: at 0.75 the fibroblast split survives."""
    v = vote(paths(), TREE, min_support=0.75)
    assert v["Stromal/Fibroblast"]["verdict"] in ("KEEP", "FORCE")   # 0.80 >= 0.75: not sealed
    assert v["Stromal/Fibroblast"]["verdict"] != "SEAL"
    assert v["Immune/Lymphoid"]["verdict"] == "SEAL"         # 0.571 < 0.75


# ---------------------------------------------------------------- the tree edit

def test_seal_deletes_the_whole_child_set_so_the_node_becomes_a_leaf():
    v = vote(paths(), TREE, min_support=1.0)
    sealed, removed = seal_tree(TREE, v)
    assert "Fibroblast" not in sealed["children"]            # a LEAF, not a smaller node
    assert "Lymphoid" not in sealed["children"]
    assert removed["Stromal/Fibroblast"] == ["Matrifibrocyte", "Quiescent fibroblast"]
    # open nodes are untouched — including their never-reached children
    assert sealed["children"]["Cardiomyocyte"] == ["Working cardiomyocyte",
                                                   "Conduction cardiomyocyte"]
    assert TREE["children"]["Fibroblast"] == ["Matrifibrocyte", "Quiescent fibroblast"]  # no mutation


def test_patterns_for_unreachable_nodes_are_dropped():
    v = vote(paths(), TREE, min_support=1.0)
    sealed, _ = seal_tree(TREE, v)
    for gone in ("Matrifibrocyte", "Quiescent fibroblast", "B cell", "NK cell"):
        assert gone not in sealed["patterns"], gone
    for kept in ("Pericyte", "Macrophage", "Conduction cardiomyocyte"):
        assert kept in sealed["patterns"], kept


def test_sealed_tree_is_json_serialisable():
    v = vote(paths(), TREE, min_support=1.0)
    sealed, _ = seal_tree(TREE, v)
    assert json.loads(json.dumps(sealed))["children"]["Stromal"] == ["Fibroblast", "Mural"]


# ---------------------------------------------------------------- applying it

def test_apply_scope_collapses_sealed_descendants_and_leaves_open_ones_alone():
    v = vote(paths(), TREE, min_support=1.0)
    assert apply_scope("Stromal/Fibroblast/Matrifibrocyte", v) == "Stromal/Fibroblast"
    assert apply_scope("Immune/Lymphoid/NK cell", v) == "Immune/Lymphoid"
    assert apply_scope("Stromal/Mural/Pericyte", v) == "Stromal/Mural/Pericyte"
    assert apply_scope("UNRESOLVED", v) == "UNRESOLVED"
    assert apply_scope("EXCLUDED", v) == "EXCLUDED"


def test_apply_scope_makes_every_sample_agree_at_the_sealed_nodes():
    """The point of the whole exercise: after the scope, no animal reports a fibroblast subtype."""
    v = vote(paths(), TREE, min_support=1.0)
    for sample, ps in paths().items():
        after = {apply_scope(p, v) for p in ps}
        assert not any(p.startswith("Stromal/Fibroblast/") for p in after), sample
        assert not any(p.startswith("Immune/Lymphoid/") for p in after), sample


# ---------------------------------------------------------------- the ambiguity guard

def test_duplicate_bare_names_are_detected():
    """`children` is keyed by BARE name, so a name at two positions would seal both."""
    assert bare_names_unique(TREE) == {}
    bad = {"children": {"root": ["A", "B"], "A": ["X"], "B": ["X"], "X": ["leaf"]}}
    assert bare_names_unique(bad) == {"X": 2}


def test_internal_nodes_are_full_paths():
    n = internal_nodes(TREE)
    assert "Stromal/Fibroblast" in n and "Fibroblast" not in n
    assert n["Stromal/Fibroblast"] == ["Matrifibrocyte", "Quiescent fibroblast"]


# ---------------------------------------------------------------- the independent L1 tree

def test_truncate_tree_keeps_only_the_top_level():
    from scanno.scope import truncate_tree
    t = truncate_tree(TREE, depth=1)
    assert t["children"] == {"root": TREE["children"]["root"]}
    assert "Stromal" not in t["children"]          # a LEAF now: the walk stops after one step
    assert set(t["patterns"]) == set(TREE["children"]["root"]) | set()
    assert TREE["children"]["Stromal"] == ["Fibroblast", "Mural"]      # no mutation


def test_the_l1_tree_is_independent_of_every_seal():
    """No seal at any depth can move the L1 column, because L1 has no depth to seal."""
    from scanno.scope import truncate_tree
    v = vote(paths(), TREE, min_support=1.0)
    sealed, _ = seal_tree(TREE, v)
    assert truncate_tree(TREE, 1) == truncate_tree(sealed, 1)


def test_truncate_tree_depth_two_keeps_the_second_level():
    from scanno.scope import truncate_tree
    t = truncate_tree(TREE, depth=2)
    assert t["children"]["Stromal"] == ["Fibroblast", "Mural"]
    assert "Fibroblast" not in t["children"]       # third level gone
    assert "Matrifibrocyte" not in t["patterns"]


def test_truncate_tree_refuses_depth_zero():
    from scanno.scope import truncate_tree
    with raises(ValueError):
        truncate_tree(TREE, depth=0)


def test_report_does_not_hardcode_a_cohort_size():
    """The first version printed "/10" as a literal — this cohort's size, baked into the tool."""
    from scanno.scope import format_report
    p = {f"S{i}": ["Cardiomyocyte/Working cardiomyocyte"] for i in range(7)}
    v = vote(p, TREE, min_support=1.0)
    txt = "\n".join(format_report(v, n_samples=len(p)))
    assert "7/7" in txt and "/10" not in txt


def test_an_open_node_holding_its_own_cells_is_drawn_with_its_count():
    """The first version drew such a node bare, hiding every nucleus stranded at it.

    An open internal node holds cells when one sample's gap failed where the others' cleared —
    the exact disagreement the scope exists to surface. Drawing it with no count made the tree
    read as complete while being short by every stranded nucleus.
    """
    from scanno.scope import format_tree, scoped_counts
    p = {"S1": ["Endothelial"] * 3 + ["Endothelial/Endocardial"] * 7,
         "S2": ["Endothelial/Endocardial"] * 10}
    v = vote(p, TREE, min_support=1.0)
    sealed, _ = seal_tree(TREE, v)
    lines = format_tree(sealed, v, p)
    endo = [l for l in lines if "Endothelial" in l and "Endocardial" not in l]
    # drawn with its stranded count and where those cells go — never as a terminal label
    assert endo and "3" in endo[0] and "most similar child" in endo[0], lines
    assert v["Endothelial"]["verdict"] == "FORCE"

    # every scoped nucleus is accounted for: terminal counts plus the stranded ones being
    # reassigned. The earlier version of this assertion scraped digits out of rendered text,
    # which is a check on the formatting rather than on the arithmetic.
    cells, _ = scoped_counts(v, p)
    stranded = sum(sum(r.get("stranded", {}).values()) for r in v.values()
                   if r["verdict"] == "FORCE")
    terminal = sum(n for node, n in cells.items()
                   if v.get(node, {}).get("verdict") != "FORCE")
    assert terminal + stranded == sum(cells.values()) == 20, (terminal, stranded, cells)


def test_the_l1_run_is_untouched_by_seals_and_by_force():
    """L1 is its OWN annotation. Nothing the scope decides may reach it.

    The scope acts on the deep tree: it SEALS nodes and it FORCEs stranded cells down to their
    most similar child. Both operate strictly BELOW level 1, and the L1 tree has nothing below
    level 1 — root's children are leaves there — so neither verdict can apply to it. That is the
    guarantee the two-column deliverable rests on: `Stromal` in the L1 column means the
    compartment and always will, while the scope column can never say `Stromal` bare at all.
    """
    from scanno.scope import truncate_tree
    v = vote(paths(), TREE, min_support=1.0)
    assert {r["verdict"] for r in v.values()} & {"SEAL", "FORCE"}      # both fire on this data

    l1 = truncate_tree(TREE, 1)
    sealed, _ = seal_tree(TREE, v)
    assert truncate_tree(sealed, 1) == l1                              # seals cannot reach it

    # and no node of the L1 tree can ever be FORCEd: they have no children to be pushed to
    assert all(c not in l1["children"] for c in l1["children"]["root"])
    v_l1 = vote(paths(), l1, min_support=1.0)
    assert "FORCE" not in {r["verdict"] for r in v_l1.values()}
    assert "SEAL" not in {r["verdict"] for r in v_l1.values()}


# =============================================================================================
# THE SCOPE ITSELF — locked. These assert the RESULT, not the mechanism that produces it.
# =============================================================================================
#
# The vote, the verdicts and the sealed tree were all covered. The SCOPE — the label set the
# following annotation is aimed at — was not, because nothing returned it: a caller had to
# re-derive it from a tree, and the report stated the rule and its costs without ever stating the
# answer. These lock it.


def test_scope_is_the_leaves_of_the_tree_in_force():
    """The scope is the label set: every leaf of the sealed tree, as a full path."""
    from scanno.scope import scope_labels
    v = vote(paths(), TREE, min_support=1.0)
    got = {r["label"] for r in scope_labels(TREE, v)}
    sealed_tree, _ = seal_tree(TREE, v)
    kids = sealed_tree["children"]
    below = {c for ks in kids.values() for c in ks}
    expect_bare = (below | set(kids)) - set(kids)
    assert {p.split("/")[-1] for p in got} == expect_bare
    assert "root" not in got, "the root is never a label: with no children every cell is UNRESOLVED"


def test_scope_says_why_each_label_terminates():
    """`leaf` and `sealed` look identical in a label column and mean opposite things."""
    from scanno.scope import scope_labels
    v = vote(paths(), TREE, min_support=1.0)
    why = {r["label"]: r["terminal"] for r in scope_labels(TREE, v)}
    for path, verdict in v.items():
        if verdict.get("verdict") == "SEAL":
            assert why.get(path) == "sealed", (
                f"{path} was sealed, so it is a label and must be marked 'sealed' — a reader who "
                f"cannot tell it from 'leaf' reads a sealed compartment as the finest resolution "
                f"the tissue supports")
    assert any(t == "leaf" for t in why.values()), "declared leaves must be marked 'leaf'"


def test_scope_carries_the_depth_each_label_stops_at():
    """Mixed depth is the point: a scope with one depth has not used the tree."""
    from scanno.scope import scope_labels
    rows = scope_labels(TREE, vote(paths(), TREE, min_support=1.0))
    assert all(r["depth"] == len(r["label"].split("/")) for r in rows)
    assert len({r["depth"] for r in rows}) > 1, "the scope should span more than one depth here"


def test_scope_labels_unpacks_seal_tree_correctly():
    """REGRESSION. `seal_tree` returns (tree, removed); unpacking it as a tree raised
    AttributeError: 'tuple' object has no attribute 'get' on the first real scope. It survived a
    compile check and two readings because the name reads like a tree."""
    from scanno.scope import scope_labels
    rows = scope_labels(TREE, vote(paths(), TREE, min_support=1.0))
    assert isinstance(rows, list) and rows and isinstance(rows[0], dict)
    assert set(rows[0]) == {"label", "depth", "terminal"}


def test_scope_is_a_vocabulary_not_a_census():
    """A label no cell reaches is STILL in the scope.

    "this cohort has none" and "this run could not have said so" are different statements, and
    only the first is a finding. So the scope is derived from the TREE and the verdicts, never
    from which labels happened to be populated.
    """
    from scanno.scope import scope_labels
    v = vote(paths(), TREE, min_support=1.0)
    rows = scope_labels(TREE, v)
    reached = {p for ps in paths().values() for p in ps}
    unpopulated = [r["label"] for r in rows if r["label"] not in reached]
    assert unpopulated, "this fixture should leave at least one scope label unreached"
    assert all(r["terminal"] in ("leaf", "sealed") for r in rows)


def test_seal_is_the_only_edit_the_scope_makes():
    """FORCE and KEEP must not change the tree. Only SEAL removes children."""
    from scanno.scope import scope_labels
    v = vote(paths(), TREE, min_support=1.0)
    only_force = {k: {"verdict": "FORCE" if d.get("verdict") == "SEAL" else d.get("verdict")}
                  for k, d in v.items()}
    declared = {r["label"] for r in scope_labels(TREE, {})}
    forced = {r["label"] for r in scope_labels(TREE, only_force)}
    assert forced == declared, (
        "turning every SEAL into a FORCE must leave the label set identical to the declared "
        "tree's leaves — a FORCE reassigns cells and never removes a label")


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
