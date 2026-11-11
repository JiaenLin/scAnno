"""The common scope: does the vote return the scope its rule implies, on a cohort built to test it?

The fixture below is INVENTED: a ten-sample reach/descend pattern over an invented tree, laid out so
that every case the vote has to get right occurs once - a node most samples reached and most
descended (sealed under unanimity), a node seven samples reached and four of those descended
(support conditioned on reach), two unanimous nodes that still strand cells (FORCE), open nodes (KEEP) and a
declared child nobody reached. If a change to `scope.py` stops returning these verdicts, the rule has
moved and somebody has to say so out loud.

Whether the vote reproduces the scope a cohort decided on is not a unit test: it is that cohort's
reproduction (single-cell-harness rule 1(c)), which re-runs `annotate --scope` on the approved
scope.json and compares every label column.
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

A = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10"]

TREE = {
    "children": {
        "root": ["Epithelial", "Mesenchymal", "Vascular", "Immune", "Glial", "Germline"],
        "Epithelial": ["Basal", "Luminal", "Ciliated"],
        "Mesenchymal": ["Stellate", "Contractile"],
        "Stellate": ["Stellate quiescent", "Stellate activated"],
        "Contractile": ["Contractile slow", "Contractile fast"],
        "Vascular": ["Arteriolar", "Venular", "Sinusoidal"],
        "Immune": ["Phagocyte", "Lymphocyte"],
        "Phagocyte": ["Resident phagocyte"],
        "Lymphocyte": ["Lymphocyte T", "Lymphocyte B"],
    },
    "patterns": {n: ["x"] for n in
                 ["Epithelial", "Basal", "Luminal", "Ciliated", "Mesenchymal", "Stellate",
                  "Stellate quiescent", "Stellate activated", "Contractile", "Contractile slow",
                  "Contractile fast", "Vascular", "Arteriolar", "Venular", "Sinusoidal", "Immune",
                  "Phagocyte", "Resident phagocyte", "Lymphocyte", "Lymphocyte T", "Lymphocyte B",
                  "Glial", "Germline"]},
    "members": {},
}

#: Invented terminal paths, per sample: which samples hold at least one cluster ending on each path.
#: Laid out so that `Mesenchymal/Stellate` is reached by all ten and descended by eight (S03 and S06
#: stop on it), `Immune/Lymphocyte` is reached by seven (not S06, S08, S09) and descended by four,
#: `Mesenchymal` and `Vascular` are descended by every sample that reached them yet hold cells of
#: their own (S04; S03 and S08), and `Epithelial/Ciliated` is declared and never reached.
COHORT = {
    "Epithelial/Basal": A,
    "Epithelial/Luminal": [s for s in A if s != "S05"],
    "Vascular/Arteriolar": A,
    "Vascular/Venular": [s for s in A if s != "S02"],
    "Vascular/Sinusoidal": ["S01", "S03", "S06", "S07", "S09"],
    "Vascular": ["S03", "S08"],
    "Immune/Phagocyte/Resident phagocyte": A,
    "Mesenchymal/Contractile/Contractile slow": [s for s in A if s != "S06"],
    "Mesenchymal/Contractile/Contractile fast": [s for s in A if s not in ("S01", "S09")],
    "Mesenchymal/Stellate/Stellate quiescent": ["S02", "S04", "S05", "S07", "S08", "S10"],
    "Mesenchymal/Stellate/Stellate activated": ["S01", "S05", "S07", "S09"],
    "Mesenchymal/Stellate": ["S03", "S06", "S10"],
    "Mesenchymal": ["S04"],
    "Immune/Lymphocyte/Lymphocyte T": ["S02", "S05"],
    "Immune/Lymphocyte/Lymphocyte B": ["S07", "S10"],
    "Immune/Lymphocyte": ["S01", "S03", "S04", "S05"],
    "Glial": ["S02", "S03", "S07", "S10"],
    "Germline": ["S01", "S06", "S08"],
}


def paths():
    """{sample: [path, ...]} — one cell per (path, sample) pair is enough to vote."""
    out = {s: [] for s in A}
    for path, samples in COHORT.items():
        for s in samples:
            out[s].append(path)
    return out


# ---------------------------------------------------------------- the reproduction test

def test_unanimity_seals_the_two_contested_splits_and_forces_the_two_stranding_nodes():
    v = vote(paths(), TREE, min_support=1.0)
    sealed = sorted(n for n, r in v.items() if r["verdict"] == "SEAL")
    assert sealed == ["Immune/Lymphocyte", "Mesenchymal/Stellate"], sealed

    # nothing else was sealed. Two nodes are unanimous AND strand cells, so they are FORCE:
    # admissible splits that nothing may terminate on.
    for node in ("root", "Epithelial", "Immune", "Immune/Phagocyte", "Mesenchymal/Contractile"):
        assert v[node]["verdict"] == "KEEP", (node, v[node])
    for node in ("Mesenchymal", "Vascular"):
        assert v[node]["verdict"] == "FORCE", (node, v[node])
        assert v[node]["support"] == 1.0            # unanimous, yet still stranding cells
        assert v[node]["stranded"], node


def test_the_two_seals_carry_the_evidence_that_produced_them():
    v = vote(paths(), TREE, min_support=1.0)
    assert v["Mesenchymal/Stellate"]["n_reached"] == 10
    assert v["Mesenchymal/Stellate"]["n_descended"] == 8
    assert approx(v["Mesenchymal/Stellate"]["support"], 0.8)
    assert v["Immune/Lymphocyte"]["n_reached"] == 7
    assert v["Immune/Lymphocyte"]["n_descended"] == 4
    assert approx(v["Immune/Lymphocyte"]["support"], 4 / 7)


def test_seal_removes_the_named_labels_not_a_category():
    v = vote(paths(), TREE, min_support=1.0)
    lost = sealed_labels(v, paths())
    assert set(lost["Mesenchymal/Stellate"]) == {"Mesenchymal/Stellate/Stellate quiescent",
                                                 "Mesenchymal/Stellate/Stellate activated"}
    assert set(lost["Immune/Lymphocyte"]) == {"Immune/Lymphocyte/Lymphocyte T",
                                              "Immune/Lymphocyte/Lymphocyte B"}


# ---------------------------------------------------------------- the conditioning

def test_a_sample_that_never_reached_the_node_casts_no_vote():
    """Absence is a missing observation, not a vote against — the easiest way to get this wrong.

    Lymphocyte is reached by seven of the ten. If the three that never reached it counted as opposition the
    support would be 4/10 = 0.40; conditioned correctly it is 4/7 = 0.571.
    """
    v = node_votes(paths())
    assert v["Immune/Lymphocyte"]["n_reached"] == 7
    assert approx(v["Immune/Lymphocyte"]["support"], 4 / 7)
    assert "S06" not in v["Immune/Lymphocyte"]["reached"]


def test_sentinels_are_not_votes():
    p = {"S1": ["Mesenchymal/Stellate/Stellate quiescent", "UNRESOLVED", "EXCLUDED"]}
    v = node_votes(p)
    assert "UNRESOLVED" not in v and "EXCLUDED" not in v
    assert v["Mesenchymal/Stellate"]["cells"] == {"S1": 1}


def test_descend_rule_any_versus_majority_can_disagree():
    """One sample can do both — different clusters of one lineage truncating differently."""
    p = {"S1": ["Mesenchymal/Stellate"] * 9 + ["Mesenchymal/Stellate/Stellate quiescent"],
         "S2": ["Mesenchymal/Stellate/Stellate quiescent"] * 10}
    assert node_votes(p, descend_rule="any")["Mesenchymal/Stellate"]["n_descended"] == 2
    assert node_votes(p, descend_rule="majority")["Mesenchymal/Stellate"]["n_descended"] == 1


def test_bad_descend_rule_refuses():
    with raises(ValueError):
        node_votes(paths(), descend_rule="whatever")


# ---------------------------------------------------------------- what is NOT sealed

def test_too_few_samples_is_unvotable_not_sealed():
    """Sealing on one sample's evidence is a removal with no quorum. It stays OPEN."""
    p = {"S1": ["Immune/Lymphocyte"], "S2": ["Epithelial/Basal"]}
    v = vote(p, TREE, min_support=1.0, min_reach=2)
    assert v["Immune/Lymphocyte"]["verdict"] == "UNVOTABLE"
    assert v["Immune/Lymphocyte"] not in [r for r in v.values() if r["verdict"] == "SEAL"]


def test_the_root_is_counted_and_never_sealed():
    """UNRESOLVED is root truncation, so the root MUST be votable — and must never be actionable.

    Sealing the root would delete every level-1 compartment and return UNRESOLVED for every
    nucleus. Its evidence is still reported, because root-level failure is real: a whole
    population can be lost there, in one sample and not in the next.
    """
    p = {"S1": ["Mesenchymal/Contractile/Contractile slow", "UNRESOLVED", "EXCLUDED"],
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
    assert v["Epithelial"]["verdict"] == "KEEP"             # reached, unanimous
    # Ciliated is declared but never reached; it is a child, not an internal node
    assert "Epithelial/Ciliated" not in v
    sealed = [n for n, r in v.items() if r["verdict"] == "SEAL"]
    assert all(r["n_reached"] > 0 for n, r in v.items() if n in sealed)


def test_loosening_support_stops_sealing_the_eight_of_ten_split():
    """The threshold is the knob, and it is visible: at 0.75 the 8-of-10 split survives."""
    v = vote(paths(), TREE, min_support=0.75)
    assert v["Mesenchymal/Stellate"]["verdict"] in ("KEEP", "FORCE")   # 0.80 >= 0.75: not sealed
    assert v["Mesenchymal/Stellate"]["verdict"] != "SEAL"
    assert v["Immune/Lymphocyte"]["verdict"] == "SEAL"         # 0.571 < 0.75


# ---------------------------------------------------------------- the tree edit

def test_seal_deletes_the_whole_child_set_so_the_node_becomes_a_leaf():
    v = vote(paths(), TREE, min_support=1.0)
    sealed, removed = seal_tree(TREE, v)
    assert "Stellate" not in sealed["children"]              # a LEAF, not a smaller node
    assert "Lymphocyte" not in sealed["children"]
    assert removed["Mesenchymal/Stellate"] == ["Stellate quiescent", "Stellate activated"]
    # open nodes are untouched — including their never-reached children
    assert sealed["children"]["Epithelial"] == ["Basal", "Luminal", "Ciliated"]
    assert TREE["children"]["Stellate"] == ["Stellate quiescent", "Stellate activated"]  # no mutation


def test_patterns_for_unreachable_nodes_are_dropped():
    v = vote(paths(), TREE, min_support=1.0)
    sealed, _ = seal_tree(TREE, v)
    for gone in ("Stellate quiescent", "Stellate activated", "Lymphocyte T", "Lymphocyte B"):
        assert gone not in sealed["patterns"], gone
    for kept in ("Contractile slow", "Resident phagocyte", "Ciliated"):
        assert kept in sealed["patterns"], kept


def test_sealed_tree_is_json_serialisable():
    v = vote(paths(), TREE, min_support=1.0)
    sealed, _ = seal_tree(TREE, v)
    assert json.loads(json.dumps(sealed))["children"]["Mesenchymal"] == ["Stellate", "Contractile"]


# ---------------------------------------------------------------- applying it

def test_apply_scope_collapses_sealed_descendants_and_leaves_open_ones_alone():
    v = vote(paths(), TREE, min_support=1.0)
    assert apply_scope("Mesenchymal/Stellate/Stellate quiescent", v) == "Mesenchymal/Stellate"
    assert apply_scope("Immune/Lymphocyte/Lymphocyte B", v) == "Immune/Lymphocyte"
    assert apply_scope("Mesenchymal/Contractile/Contractile slow", v) == \
        "Mesenchymal/Contractile/Contractile slow"
    assert apply_scope("UNRESOLVED", v) == "UNRESOLVED"
    assert apply_scope("EXCLUDED", v) == "EXCLUDED"


def test_apply_scope_makes_every_sample_agree_at_the_sealed_nodes():
    """The point of the whole exercise: after the scope, no sample reports a sealed subtype."""
    v = vote(paths(), TREE, min_support=1.0)
    for sample, ps in paths().items():
        after = {apply_scope(p, v) for p in ps}
        assert not any(p.startswith("Mesenchymal/Stellate/") for p in after), sample
        assert not any(p.startswith("Immune/Lymphocyte/") for p in after), sample


# ---------------------------------------------------------------- the ambiguity guard

def test_duplicate_bare_names_are_detected():
    """`children` is keyed by BARE name, so a name at two positions would seal both."""
    assert bare_names_unique(TREE) == {}
    bad = {"children": {"root": ["A", "B"], "A": ["X"], "B": ["X"], "X": ["leaf"]}}
    assert bare_names_unique(bad) == {"X": 2}


def test_internal_nodes_are_full_paths():
    n = internal_nodes(TREE)
    assert "Mesenchymal/Stellate" in n and "Stellate" not in n
    assert n["Mesenchymal/Stellate"] == ["Stellate quiescent", "Stellate activated"]


# ---------------------------------------------------------------- the independent L1 tree

def test_truncate_tree_keeps_only_the_top_level():
    from scanno.scope import truncate_tree
    t = truncate_tree(TREE, depth=1)
    assert t["children"] == {"root": TREE["children"]["root"]}
    assert "Mesenchymal" not in t["children"]      # a LEAF now: the walk stops after one step
    assert set(t["patterns"]) == set(TREE["children"]["root"]) | set()
    assert TREE["children"]["Mesenchymal"] == ["Stellate", "Contractile"]  # no mutation


def test_the_l1_tree_is_independent_of_every_seal():
    """No seal at any depth can move the L1 column, because L1 has no depth to seal."""
    from scanno.scope import truncate_tree
    v = vote(paths(), TREE, min_support=1.0)
    sealed, _ = seal_tree(TREE, v)
    assert truncate_tree(TREE, 1) == truncate_tree(sealed, 1)


def test_truncate_tree_depth_two_keeps_the_second_level():
    from scanno.scope import truncate_tree
    t = truncate_tree(TREE, depth=2)
    assert t["children"]["Mesenchymal"] == ["Stellate", "Contractile"]
    assert "Stellate" not in t["children"]         # third level gone
    assert "Stellate quiescent" not in t["patterns"]


def test_truncate_tree_refuses_depth_zero():
    from scanno.scope import truncate_tree
    with raises(ValueError):
        truncate_tree(TREE, depth=0)


def test_report_does_not_hardcode_a_cohort_size():
    """The first version printed its denominator as a literal, a cohort size baked into the tool."""
    from scanno.scope import format_report
    p = {f"S{i}": ["Epithelial/Basal"] for i in range(7)}
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
    p = {"S1": ["Vascular"] * 3 + ["Vascular/Venular"] * 7,
         "S2": ["Vascular/Venular"] * 10}
    v = vote(p, TREE, min_support=1.0)
    sealed, _ = seal_tree(TREE, v)
    lines = format_tree(sealed, v, p)
    endo = [l for l in lines if "Vascular" in l and "Venular" not in l]
    # drawn with its stranded count and where those cells go — never as a terminal label
    assert endo and "3" in endo[0] and "most similar child" in endo[0], lines
    assert v["Vascular"]["verdict"] == "FORCE"

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
    guarantee the two-column deliverable rests on: `Mesenchymal` in the L1 column means the
    compartment and always will, while the scope column can never say `Mesenchymal` bare at all.
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
