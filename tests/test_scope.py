"""The common scope: does the vote reproduce a scope a human derived by reading the table?

The SAMBO fixture below is not invented. It is that cohort's real reach/descend pattern at every
internal node, transcribed from the ten pass-1 objects, and the expected verdicts are the scope
the PI approved after reading it. If a change to `scope.py` stops reproducing it, the rule has
moved and somebody has to say so out loud.
"""
import json

import pytest

from scanno.scope import (apply_scope, bare_names_unique, internal_nodes, node_votes,
                          seal_tree, sealed_labels, vote)

A = ["Aging1", "Aging2", "Aging3", "Aging_HFD1", "Aging_HFD2", "Aging_HFD3",
     "Young1", "Young2", "Young_HFD1", "Young_HFD2"]

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

#: SAMBO's real terminal paths, per animal. Transcribed from the ten pass-1 objects.
SAMBO = {
    "Cardiomyocyte/Working cardiomyocyte": A,
    "Endothelial/Vascular endothelial": A,
    "Endothelial/Lymphatic endothelial": A,
    "Immune/Myeloid/Macrophage": A,
    "Endothelial/Endocardial": [s for s in A if s != "Aging_HFD1"],
    "Mesothelial": [s for s in A if s != "Aging1"],
    "Stromal/Mural/Smooth muscle": [s for s in A if s != "Young2"],
    "Stromal/Mural/Pericyte": [s for s in A if s not in ("Aging2", "Young1")],
    "Stromal/Fibroblast/Matrifibrocyte": ["Aging3", "Aging_HFD1", "Aging_HFD2", "Aging_HFD3",
                                          "Young2", "Young_HFD1", "Young_HFD2"],
    "Stromal/Fibroblast": ["Aging2", "Aging_HFD1", "Aging_HFD3", "Young1", "Young_HFD1",
                           "Young_HFD2"],
    "Stromal/Fibroblast/Quiescent fibroblast": ["Aging1", "Aging3"],
    "Immune/Lymphoid": ["Aging1", "Aging2", "Aging3", "Aging_HFD2", "Aging_HFD3"],
    "Immune/Lymphoid/B cell": ["Aging1", "Aging_HFD3"],
    "Immune/Lymphoid/NK cell": ["Aging_HFD1", "Young_HFD2"],
    "Adipocyte": ["Aging_HFD1", "Aging_HFD2", "Aging_HFD3", "Young_HFD1", "Young_HFD2"],
    "Neural": ["Young1", "Young2", "Young_HFD1", "Young_HFD2"],
    "Endothelial": ["Aging_HFD1", "Young_HFD1"],
    "Stromal": ["Young2"],
}


def paths():
    """{sample: [path, ...]} — one cell per (path, sample) pair is enough to vote."""
    out = {s: [] for s in A}
    for path, samples in SAMBO.items():
        for s in samples:
            out[s].append(path)
    return out


# ---------------------------------------------------------------- the reproduction test

def test_unanimity_reproduces_the_approved_sambo_scope():
    v = vote(paths(), TREE, min_support=1.0)
    sealed = sorted(n for n, r in v.items() if r["verdict"] == "SEAL")
    assert sealed == ["Immune/Lymphoid", "Stromal/Fibroblast"], sealed

    # and nothing else moved: every other reached node was unanimous
    for node in ("root", "Cardiomyocyte", "Stromal", "Endothelial", "Immune",
                 "Immune/Myeloid", "Stromal/Mural"):   # root: reached 10/10, never sealable
        assert v[node]["verdict"] == "KEEP", (node, v[node])


def test_the_two_seals_carry_the_evidence_that_produced_them():
    v = vote(paths(), TREE, min_support=1.0)
    assert v["Stromal/Fibroblast"]["n_reached"] == 10
    assert v["Stromal/Fibroblast"]["n_descended"] == 8
    assert v["Stromal/Fibroblast"]["support"] == pytest.approx(0.8)
    assert v["Immune/Lymphoid"]["n_reached"] == 7
    assert v["Immune/Lymphoid"]["n_descended"] == 4
    assert v["Immune/Lymphoid"]["support"] == pytest.approx(4 / 7)


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
    assert v["Immune/Lymphoid"]["support"] == pytest.approx(4 / 7)
    assert "Young1" not in v["Immune/Lymphoid"]["reached"]


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
    with pytest.raises(ValueError):
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
    assert v["Stromal/Fibroblast"]["verdict"] == "KEEP"      # 0.80 >= 0.75
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
    with pytest.raises(ValueError):
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
    assert endo and "3" in endo[0] and "stranded" in endo[0], lines

    # and the drawing must account for every scoped nucleus
    cells, _ = scoped_counts(v, p)
    drawn = sum(int(t.replace(",", "")) for l in lines for t in l.split()
                if t.replace(",", "").isdigit() and "/" not in t)
    assert drawn >= sum(cells.values()), (drawn, sum(cells.values()))
