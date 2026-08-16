"""The common scope, as a section of the STANDARD report.

WHY THE FIRST TEST IS ABOUT MODULE IDENTITY RATHER THAN RENDERING

A previous attempt put this section in `scanno/report.py`. That module is real and it IS
imported — `scanno annotate --report` calls `report.collect`, and `scanno report --panels auto`
calls `report.panels_by_depth` — so the code compiled, the suite passed, and the section rendered
NOWHERE, because the cohort document is assembled by `scanno/document.py:write_all` and
`document.py` does not import `report.py`. A section that compiles and does not appear is worse
than one that raises: nothing anywhere says it is missing.

So `test_the_section_is_on_the_live_path` asserts the WIRING, and every other test renders the
section and looks for a value that could only have come from the scope JSON.

The fixture is SAMBO's real reach/descend pattern, shared with `test_scope.py`. The last group
of tests re-runs the section on a THREE-sample cohort with unrelated node names, because the one
failure a human will not notice is the section quoting a number that belongs to the cohort it
was written for.
"""
import html as _html
import json
import re
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scanno.scope import format_tree, internal_nodes, seal_tree, sealed_labels, vote  # noqa: E402

from test_scope import A, TREE, paths                                        # noqa: E402


def scope_json(paths_by_sample, tree, *, min_support=1.0, min_reach=2, descend_rule="any",
               path_key="scanno_path_r1p0", tree_path="mouse_heart_tree.json"):
    """The payload `scanno scope --out` writes, built by the calls `cli._scope` makes.

    Constructed rather than transcribed so a change to the vote reaches this file: a hand-written
    fixture keeps passing after `vote` starts producing something the section cannot render.
    """
    verdicts = vote(paths_by_sample, tree, min_support=min_support, min_reach=min_reach,
                    descend_rule=descend_rule)
    lost = sealed_labels(verdicts, paths_by_sample)
    sealed_tree, removed = seal_tree(tree, verdicts)
    payload = {"rule": {"min_support": min_support, "min_reach": min_reach,
                        "descend_rule": descend_rule, "path_key": path_key},
               "samples": sorted(paths_by_sample),
               "tree": tree_path,
               "nodes": verdicts,
               "sealed": {k: list(v) for k, v in removed.items()},
               "removed_labels": lost,
               "declared_internal_nodes": internal_nodes(tree),
               "tree_lines": format_tree(sealed_tree, verdicts, paths_by_sample),
               "n_samples": len(paths_by_sample)}
    # Round-tripped through JSON because that is how the report receives it. A NaN support
    # survives in memory and comes back as a float that formats as "nan" on the page.
    return json.loads(json.dumps(payload, default=str))


def render(scope, **kw):
    from scanno.document import scope_section
    return "".join(scope_section(scope, **kw))


def rows_of(h, needle):
    return [r for r in re.findall(r"<tr>.*?</tr>", h, re.S) if needle in r]


_SAMBO = {}


def sambo():
    if "s" not in _SAMBO:
        _SAMBO["s"] = scope_json(paths(), TREE)
    return _SAMBO["s"]


# ---------------------------------------------------------------- the wiring, not the rendering

def test_the_section_is_on_the_live_path():
    """`document.py` assembles the cohort document. `report.py` does not, and never did."""
    doc = (ROOT_DIR / "scanno" / "document.py").read_text(encoding="utf-8")
    rpt = (ROOT_DIR / "scanno" / "report.py").read_text(encoding="utf-8")
    cli = (ROOT_DIR / "scanno" / "cli.py").read_text(encoding="utf-8")

    assert "def scope_section(" in doc, "the section must live in the module write_all() calls"
    # Defined is not enough: a defined-but-uncalled section is the same invisible failure with
    # the same clean suite.
    cohort = doc.split("def write_cohort(", 1)[1].split("\ndef ", 1)[0]
    assert "scope_section(" in cohort, "write_cohort() must call it"
    assert "def scope_section(" not in rpt, (
        "report.py is not imported by document.py; a section there renders nowhere")
    # the invariant that made the earlier attempt invisible
    assert "from .report" not in doc and "import report" not in doc
    assert "from .document import write_all" in cli


def test_the_cli_threads_scope_from_flag_to_context():
    cli = (ROOT_DIR / "scanno" / "cli.py").read_text(encoding="utf-8")
    ctxsrc = (ROOT_DIR / "scanno" / "context.py").read_text(encoding="utf-8")
    doc = (ROOT_DIR / "scanno" / "document.py").read_text(encoding="utf-8")
    assert '"--scope"' in cli, "the report subcommand must accept --scope"
    assert "scope=scope" in cli, "and pass it into the Context"
    assert "scope=None" in ctxsrc and "self.scope = scope" in ctxsrc
    assert '"scope": getattr(ctx, "scope", None)' in doc, "report.json must carry it too"


def test_the_section_reads_only_keys_the_writer_writes():
    """Every key the section consumes must be one `cli._scope` actually emits."""
    cli = (ROOT_DIR / "scanno" / "cli.py").read_text(encoding="utf-8")
    fn = cli.split("def _scope(", 1)[1].split("\ndef ", 1)[0]
    payload = fn.split("payload = {", 1)[1].split("Path(a.out).write_text", 1)[0]
    for key in ("rule", "nodes", "sealed", "removed_labels", "tree_lines", "n_samples",
                "samples", "tree"):
        assert f'"{key}"' in payload, f"scope.json is not written with {key!r}"


# ---------------------------------------------------------------- what the section must show

def test_a_missing_scope_is_a_NAMED_absence_not_a_missing_section():
    h = render(None)
    assert "The common scope" in h
    assert "scanno scope" in h and "--scope" in h
    # it must say what the reader LOSES, not merely that something is absent
    assert "the scope removed the label everywhere" in h


def test_the_rule_is_stated_in_words_with_the_values_actually_used():
    h = render(sambo())
    assert "min-support 1.0" in h
    assert "min-reach 2" in h
    assert "descend-rule <code>any</code>" in h
    assert "one cluster descending is enough" in h          # what 'any' MEANS
    assert "casts NO vote" in h                             # the conditioning
    assert "never the cohort size" in h
    assert "The root is never sealed" in h
    assert "scanno_path_r1p0" in h                          # the column it was voted on


def test_the_descend_rule_words_follow_the_rule_that_was_used():
    h = render(scope_json(paths(), TREE, descend_rule="majority"))
    assert "descend-rule <code>majority</code>" in h
    assert "more than half" in h
    assert "one cluster descending is enough" not in h


def test_the_drawn_tree_is_rendered_preformatted_and_escaped():
    s = sambo()
    h = render(s)
    assert "<pre class='tree'>" in h
    assert "Working cardiomyocyte" in h and "SEALED" in h
    body = h.split("<pre class='tree'>", 1)[1].split("</pre>", 1)[0]
    for line in s["tree_lines"]:
        if str(line).strip():
            assert _html.escape(str(line), quote=True) in body, line
    assert "<" not in body.replace("&lt;", ""), "tree text must be escaped, never injected"


def test_the_per_node_table_carries_reach_descend_support_and_verdict():
    h = render(sambo())
    fib = rows_of(h, "Stromal/Fibroblast</span>")
    assert fib, "no row for the sealed node"
    assert "10/10" in fib[0] and "0.800" in fib[0] and "SEAL" in fib[0], fib[0]
    lym = [r for r in rows_of(h, "Immune/Lymphoid</span>") if "0.571" in r]
    assert lym and "7/10" in lym[0], lym
    assert "KEEP" in h
    # the glossary: every verdict the vote can return is explained on the page
    for v in ("KEEP", "SEAL", "UNVOTABLE", "UNREACHED"):
        assert f"<b>{v}</b>" in h


def test_the_denominator_is_the_cohort_not_the_reach():
    """4/7 and 4/10 are different statements. `reached` must show which one is on the page."""
    h = render(sambo())
    lym = [r for r in rows_of(h, "Immune/Lymphoid</span>") if "0.571" in r]
    assert lym, "no vote row for Immune/Lymphoid"
    assert "7/10" in lym[0], lym[0]              # reached by 7, of a cohort of 10
    assert "4/10" not in lym[0], "support is conditioned on the 7 that reached it"


def test_each_seal_is_reported_BY_LABEL_with_counts_not_as_a_category():
    """The project rule: a removal is stated as its members and read, never described."""
    p = paths()
    p["Aging3"] += ["Stromal/Fibroblast/Matrifibrocyte"] * 6      # so counts are not all 1
    s = scope_json(p, TREE)
    h = render(s)
    for label in ("Matrifibrocyte", "Quiescent fibroblast", "B cell", "NK cell"):
        assert label in h, f"{label} is removed by a seal and must be NAMED"
    n = s["removed_labels"]["Stromal/Fibroblast"]["Stromal/Fibroblast/Matrifibrocyte"]
    assert n == 13, n
    row = rows_of(h, "Matrifibrocyte")
    assert row and f"{n:,}" in row[-1], row
    assert "nuclei in pass 1" in h


def test_a_removed_child_that_held_no_nuclei_is_still_named():
    tree = json.loads(json.dumps(TREE))
    tree["children"]["Lymphoid"] = ["B cell", "NK cell", "T cell"]
    tree["patterns"]["T cell"] = ["x"]
    h = render(scope_json(paths(), tree))
    assert "T cell" in h, "a seal removes a child whether or not any sample reached it"
    row = rows_of(h, "T cell")
    assert row and ">0</td>" in row[-1], row


def test_nothing_sealed_says_so_rather_than_rendering_an_empty_table():
    p = {s: ["Cardiomyocyte/Working cardiomyocyte", "Endothelial/Endocardial"] for s in A}
    h = render(scope_json(p, TREE))
    assert "Nothing was sealed" in h
    assert "no label was removed" in h


def test_the_cannot_show_note_is_present_and_says_the_three_things():
    h = render(sambo()).lower()
    assert "never an observation" in h
    assert "reversible" in h
    assert "cannot distinguish a split the data cannot make" in h
    assert "corpus" in h
    assert "the vote counts samples" in h


def test_stranded_nuclei_at_an_open_node_are_cross_checked_from_the_vote():
    """The drawing has been short by exactly these before, so the section restates the total.

    `Endothelial` and `Stromal` are KEPT and still hold nuclei of their own in this fixture —
    the residual disagreement a unanimous vote cannot see. Whatever the drawing says, the
    section computes the number from the vote and puts it on the page.
    """
    h = render(sambo())
    m = re.search(r"Cross-check[^<]*</b>\s*([\d,]+) nuclei are stranded at (\d+)", h)
    assert m, h[max(0, h.find("Cross-check") - 200):h.find("Cross-check") + 400]
    s = sambo()
    want = sum(sum(v["cells"].values()) - sum(v["cells_below"].values())
               for n, v in s["nodes"].items()
               if n != "root" and v["verdict"] != "SEAL")
    assert int(m.group(1).replace(",", "")) == want > 0
    assert int(m.group(2)) == 2, "Endothelial and Stromal"


def test_root_truncation_is_reported_separately_from_stranding():
    p = paths()
    p["Young1"] += ["UNRESOLVED"] * 5
    h = render(scope_json(p, TREE))
    assert "truncated at the <b>root</b>" in h and "UNRESOLVED" in h
    assert "<b>5</b> nuclei truncated" in h
    assert "no seal can repair them" in h


def test_a_scope_json_without_tree_lines_names_the_absence():
    s = json.loads(json.dumps(sambo()))
    s.pop("tree_lines")
    h = render(s)
    assert "tree_lines" in h and "predates" in h
    assert "the node table below is unaffected" in h
    assert "Stromal/Fibroblast" in h              # the rest of the section still renders


def test_it_writes_the_tables_as_csv_when_given_an_output_directory(tmp_path):
    h = render(sambo(), out_dir=tmp_path)
    assert (tmp_path / "tables" / "scope_nodes.csv").exists()
    assert (tmp_path / "tables" / "scope_removed_labels.csv").exists()
    assert "tables/scope_nodes.csv" in h and "tables/scope_removed_labels.csv" in h
    lost = (tmp_path / "tables" / "scope_removed_labels.csv").read_text(encoding="utf-8")
    assert "Matrifibrocyte" in lost and "Stromal/Fibroblast" in lost
    nodes = (tmp_path / "tables" / "scope_nodes.csv").read_text(encoding="utf-8")
    assert "nuclei_stopped_here_pass1" in nodes and "children_declared" in nodes


# ---------------------------------------------------------------- nothing belongs to one cohort

OTHER_TREE = {
    "children": {"root": ["Alpha", "Beta"],
                 "Alpha": ["Alpha1", "Alpha2"],
                 "Beta": ["Beta1", "Beta2"]},
    "patterns": {n: ["x"] for n in ["Alpha", "Alpha1", "Alpha2", "Beta", "Beta1", "Beta2"]},
    "members": {},
}


def test_a_three_sample_cohort_never_prints_a_ten():
    """The denominator is data, not a literal. This is how "/10" got into the tool once."""
    p = {"S1": ["Alpha/Alpha1", "Beta/Beta1"],
         "S2": ["Alpha/Alpha1", "Beta"],
         "S3": ["Alpha/Alpha2", "Beta"]}
    h = render(scope_json(p, OTHER_TREE, path_key="p", tree_path="other.json"))
    assert "/10" not in h
    assert "3/3" in h                                  # root, reached by all three
    assert "Beta1" in h
    for leaked in ("Fibroblast", "Cardiomyocyte", "Matrifibrocyte", "Lymphoid", "SAMBO",
                   "Aging", "Young"):
        assert leaked not in h, f"{leaked} belongs to another cohort"


def test_an_unreached_node_shows_n_a_rather_than_the_word_nan():
    """`support` is float('nan') where nothing reached the node; JSON gives it back as a float."""
    p = {"S1": ["Alpha/Alpha1"], "S2": ["Alpha/Alpha1"], "S3": ["Alpha/Alpha2"]}
    s = scope_json(p, OTHER_TREE, path_key="p")
    assert s["nodes"]["Beta"]["verdict"] == "UNREACHED"
    h = render(s)
    assert "<td>nan</td>" not in h.lower()
    beta = rows_of(h, ">Beta</span>")
    assert beta and "n/a" in beta[0], beta


def test_the_module_hardcodes_no_node_name_and_no_cohort_size():
    doc = (ROOT_DIR / "scanno" / "document.py").read_text(encoding="utf-8")
    section = doc.split("def scope_section(", 1)[1].split("\n# ====", 1)[0]
    code = re.sub(r'""".*?"""', "", section, flags=re.S)
    for banned in ("Fibroblast", "Matrifibrocyte", "Lymphoid", "Cardiomyocyte", "Endothelial",
                   "B cell", "NK cell", "Macrophage", "Pericyte"):
        assert banned not in code, f"{banned} is one cohort's node name"
    for m in re.finditer(r"/10\b|\b10 samples\b|109,?140|3,?799", code):
        raise AssertionError(f"cohort literal in the section: {m.group(0)}")
    # the only node name the section may know is the root, which is structural
    assert code.count('"root"') <= 2


def test_a_child_the_seal_removed_but_nobody_reached_is_still_named():
    p = {"S1": ["Alpha/Alpha1"], "S2": ["Alpha"], "S3": ["Alpha"]}
    s = scope_json(p, OTHER_TREE, path_key="p")
    assert s["nodes"]["Alpha"]["verdict"] == "SEAL"
    h = render(s)
    assert "Alpha1" in h and "Alpha2" in h              # both, one of them at 0
    a2 = rows_of(h, "Alpha2")
    assert a2 and ">0</td>" in a2[-1], a2
    assert "Beta" in h                                   # UNREACHED, reported not sealed


def test_an_unvotable_node_is_reported_as_such_and_not_sealed():
    p = {"S1": ["Alpha/Alpha1"], "S2": ["Beta/Beta1"], "S3": ["Beta/Beta1"]}
    s = scope_json(p, OTHER_TREE, min_reach=2, path_key="p")
    assert s["nodes"]["Alpha"]["verdict"] == "UNVOTABLE", s["nodes"]["Alpha"]
    h = render(s)
    alpha = rows_of(h, ">Alpha</span>")
    assert alpha and "UNVOTABLE" in alpha[0], alpha
    assert "too few samples reached it to vote" in h
    assert "reported by name and left OPEN, never sealed" in h
    assert "Alpha1" not in h.split("what each seal removes", 1)[-1]


if __name__ == "__main__":                                            # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
