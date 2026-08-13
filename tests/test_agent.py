"""The agentic route — every way a language model can put a wrong label in a results table.

Runs entirely OFFLINE. A suite that needs a key and a network is a suite nobody runs, and this
is the component where an unrun test matters most: the failure is a confident label that looks
exactly like a correct one.

The design being tested is deliberately NOT a closed vocabulary. An earlier version accepted
only nodes of the declared tree and refused everything else, which defeated the purpose - the
reason to ask a model is that the tree and the corpus may not contain the answer. The model now
answers freely and the answer is RESOLVED in tiers, so free text cannot fragment and a genuine
off-tree cell type is not thrown away.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scanno.agent import (NONE_LABEL, StubProvider, agreement, annotate_agentic, ask_cluster,
                          cluster_markers, make_prompt, parse_reply, resolve_label, tree_nodes)

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


TREE = {"children": {"root": ["Cardiomyocyte", "Immune"],
                     "Cardiomyocyte": ["Working cardiomyocyte"],
                     "Immune": ["Macrophage", "T cell"]},
        "patterns": {"Cardiomyocyte": ["cardiomyocyte", "cardiac muscle"],
                     "Working cardiomyocyte": ["working cardiomyocyte", "ventricular"],
                     "Immune": ["immune", "leukocyte"],
                     "Macrophage": ["macrophage"], "T cell": ["t cell"]}}
ONTOLOGY = [{"cell_name": "Cardiac neuron", "cl": "CL_0010022"},
            {"cell_name": "Adipocyte", "cl": "CL_0000136"},
            {"cell_name": "Macrophage", "cl": "CL_0000235"}]


def reply(label, conf=0.9, reason="because"):
    return json.dumps({"label": label, "confidence": conf, "reason": reason})


print("\n1 · free text is RESOLVED onto the taxonomy, not refused")
for text, want in (("cardiomyocyte", "Cardiomyocyte"),
                   ("Cardiac muscle cell", "Cardiomyocyte"),
                   ("ventricular cardiomyocytes", "Working cardiomyocyte"),
                   ("Macrophage", "Macrophage"),
                   ("tissue-resident macrophages", "Macrophage")):
    tier, lab, _ = resolve_label(text, TREE, ONTOLOGY)
    check(f"{text!r} -> {want}", tier == "tree" and lab == want, f"got {tier}:{lab}")
check("the longest matching pattern wins",
      resolve_label("ventricular cardiomyocyte", TREE, ONTOLOGY)[1] == "Working cardiomyocyte",
      "'cardiomyocyte' also matches the parent; the more specific pattern must win")

print("\n2 · a real cell type the TREE lacks is a finding, not an error")
tier, lab, cl = resolve_label("cardiac neuron", TREE, ONTOLOGY)
check("it resolves to the standard name", tier == "ontology" and lab == "Cardiac neuron")
check("...with its Cell Ontology id", cl == "CL_0010022", cl)
tier, lab, _ = resolve_label("some unnamed stressed state", TREE, ONTOLOGY)
check("an unknown answer is kept verbatim as a proposal",
      tier == "proposal" and lab == "some unnamed stressed state",
      "deleting it would delete exactly what this route exists to find")
check("NONE stays NONE", resolve_label(NONE_LABEL, TREE, ONTOLOGY)[0] == "none")

print("\n3 · an unusable reply is an error, never a guess")
for text, why in (("", "empty"), ("I think these are macrophages.", "prose, no JSON"),
                  ('{"label": ', "truncated JSON"),
                  ('{"reason": "no label key"}', "JSON without a label")):
    lab, _, _, err = parse_reply(text)
    check(f"{why} is an error", lab is None and bool(err), err[:56])

print("\n4 · the prompt carries the evidence but never the classifier's call")
Z = np.zeros((2, 4))
Z[0] = [5.0, 1.0, 0.0, 0.0]
Z[1] = [0.0, 0.0, 4.0, 2.0]
usable = np.ones(4, bool)
genes = np.array(["TTN", "MYH6", "PTPRC", "CD68"])
corpus = [{"path": "Cardiomyocyte/Working cardiomyocyte"}, {"path": "Immune/Macrophage"}]
ev = [{"cell_name": "Cardiomyocyte", "cl": "CL_0000746", "tier": 1, "n_hit": 2,
       "hits": ["TTN", "MYH6"]}]
p = make_prompt(["TTN", "MYH6"], TREE, ev, ONTOLOGY, species="Mouse", tissue="Heart")
check("the corpus evidence is shown", "tier 1" in p and "TTN, MYH6" in p)
check("the standard names are offered", "Cardiac neuron" in p and "CL_0010022" in p)
check("the taxonomy is offered as PREFERRED, not as the only option",
      "PREFERRED TAXONOMY" in p and "standard cell type name" in p)
check("the model is told the corpus is incomplete", "incomplete" in p and "disagree" in p,
      "an agent that defers to a thin corpus adds nothing")

prov = StubProvider([reply("Working cardiomyocyte"), reply("Macrophage")])
rows = annotate_agentic(prov, Z, usable, genes, TREE, corpus_calls=corpus)
leak = [q for q in prov.calls if "Working cardiomyocyte" in q.split("PREFERRED TAXONOMY")[0]]
check("no prompt reveals the classifier's answer", not leak,
      "an agent shown the corpus call agrees with it and the comparison measures nothing")

print("\n5 · off-tree answers are not scored as disagreements")
prov = StubProvider([reply("cardiac neuron"), reply("Macrophage")])
rows = annotate_agentic(prov, Z, usable, genes, TREE, corpus_calls=corpus, ontology=ONTOLOGY)
ag = agreement(rows)
check("only tree-resolved answers are comparable", ag["n_comparable"] == 1,
      f"n_comparable={ag['n_comparable']}")
check("...and agreement is computed over those", ag["agree"] == 1.0)
check("the off-tree answer is reported separately",
      len(ag["off_tree"]) == 1 and ag["off_tree"][0][2] == "Cardiac neuron",
      f"{ag['off_tree']}")

print("\n6 · non-determinism is measured, not hidden")
prov = StubProvider([reply("Macrophage"), reply("T cell"), reply("Macrophage")])
r = ask_cluster(prov, ["CD68"], TREE, votes=3)
check("the majority answer is returned", r["resolved"] == "Macrophage")
check("...with the disagreement rate attached", abs(r["consensus"] - 2 / 3) < 1e-9,
      f"consensus {r['consensus']:.2f} of {r['votes']}")
check("synonyms are counted as ONE answer, not as disagreement",
      ask_cluster(StubProvider([reply("cardiomyocyte"), reply("cardiac muscle cell")]),
                  ["TTN"], TREE, votes=2)["consensus"] == 1.0,
      "resolution happens before the vote or wording noise reads as instability")

print("\n7 · every call is recorded well enough to be re-read")
r = ask_cluster(StubProvider([reply("T cell", 0.4, "CD3 positive")]), ["CD3D"], TREE,
                species="Mouse", tissue="Heart")
for k in ("prompt_sha", "raw", "confidence", "reason", "votes", "tier"):
    check(f"{k} is recorded", k in r and r[k] not in (None, ""), str(r[k])[:44])

print("\n8 · a provider that fails does not stop the run")


class Broken:
    name, model = "broken", "none"

    def complete(self, prompt):
        raise RuntimeError("no key configured")


rows = annotate_agentic(Broken(), Z, usable, genes, TREE)
check("every cluster still returns a row carrying the reason",
      len(rows) == 2 and all(r["resolved"] is None and "no key" in r["error"] for r in rows))

print("\n9 · web search is refused for a style with no tool block, not silently dropped")
from scanno.agent import WEB_TOOL, HTTPProvider  # noqa: E402
check("known styles have a tool block", set(WEB_TOOL) == {"openai", "anthropic"})
try:
    HTTPProvider(preset="openai", url="http://x", style="mystery", env="NOPE", web=True)
    check("an unknown style refuses --web", False, "it accepted the flag")
except ValueError as e:
    check("an unknown style refuses --web", "no tool block" in str(e))
except RuntimeError:
    check("an unknown style refuses --web", False, "it checked the key before the tool")

print("\n" + "=" * 62)
if FAILED:
    print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
    raise SystemExit(1)
print("agentic route OK - offline, resolved not constrained, nothing coerced")
