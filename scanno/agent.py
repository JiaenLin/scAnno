"""An optional agentic second opinion: bring your own key, or your own coding agent.

WHAT THIS IS FOR

The classifier scores clusters against a curated corpus. A corpus is finite, and for most
tissues it is thin - mouse heart carries 618 tier-1/2 assertions where human lung carries 60,501.
A language model has read papers the corpus has not, and may have web search. On a thin panel it
is worth asking.

WHAT IT IS NOT

It is not the classifier and never replaces it. `classify()` stays a pure function of (query,
store, tree): same inputs, same answer, reproducible from a digest. Nothing here can change that
call. This produces a SECOND column, and where the two disagree the disagreement IS the output.

THE ANSWER IS FREE TEXT, AND THEN IT IS RESOLVED - IT IS NOT CONSTRAINED

An earlier version of this module accepted only nodes of the declared tree and refused anything
else. That was too strict, and it defeated the point: the reason to ask a model is precisely that
the tree and the corpus may not contain the answer, and a closed vocabulary makes discovering
that impossible. A cluster that is a cell type nobody put in the tree came back as a refusal.

So the model answers in its own words, and the answer is then RESOLVED, in three tiers, with the
tier recorded:

  tree        the reply matches a node's declared synonym patterns -> that node. Directly
              comparable with the classifier.
  ontology    it does not match the tree but does match a CellMarker cell type for this tissue
              -> that name plus its Cell Ontology id. A real cell type the taxonomy lacks, which
              is a finding about the TREE rather than an error by the model.
  unresolved  neither. Recorded verbatim as a proposal, never discarded and never coerced.

Resolution is what stops free text fragmenting - "cardiac muscle cell", "cardiomyocyte", "CM"
and "ventricular CM" all land on one node - without the closed list that stopped discovery.

WHAT THE MODEL IS GIVEN

  * the cluster's most distinctive genes, from the profile the classifier already built
  * WHAT THE CORPUS KNOWS about those genes in this tissue: which cell types cite them, at what
    evidence tier, with what specificity. The same evidence the classifier used, in words
  * the declared taxonomy, as the preferred vocabulary rather than the only one
  * the standard cell type names for this tissue, with their Cell Ontology ids
  * web search, where the provider supports it and it is switched on

WHAT IT IS NEVER GIVEN, AND WHY

The classifier's call. An agent told what the corpus concluded agrees with it, and two routes
that agree because one was shown the other measure nothing.

REPRODUCIBILITY

An LLM call is not reproducible by re-running it, so it is made reproducible by RECORD:
provider, model, temperature, the prompt, its hash, and the raw reply, written beside the call.
`votes > 1` asks n times and reports the agreement rate; a label that changes between identical
calls is a finding the single answer would have concealed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import urllib.error
import urllib.request

import numpy as np

TOP_N = 30                 # genes shown per cluster
EVIDENCE_TYPES = 12        # candidate cell types quoted from the corpus per cluster
TIMEOUT = 120
NONE_LABEL = "NONE"

#: Web-search tool blocks, per provider style. Absent for a style means the flag is refused
#: rather than silently ignored - a run that believed it had search and did not is a run whose
#: answers cannot be interpreted.
WEB_TOOL = {
    "openai": [{"type": "web_search"}],
    "anthropic": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
}


# ---------------------------------------------------------------------------- providers


class StubProvider:
    """Offline provider for tests and dry runs. The suite must not need a key or a network."""

    name = "stub"

    def __init__(self, replies=None, model="stub-1"):
        self.replies, self.model, self.calls = list(replies or []), model, []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.replies.pop(0) if self.replies else json.dumps(
            {"label": NONE_LABEL, "confidence": 0.0, "reason": "stub"})


class CommandProvider:
    """Any command that reads a prompt on stdin and writes a reply on stdout.

    THE BRING-YOUR-OWN-AGENT HOOK. A coding agent, a local model, a wrapper script - if it runs
    from a shell it can annotate here, and scAnno needs to know nothing about it. No vendor is
    privileged and no key passes through this package. An agent with its own web search or its
    own tools brings them; scAnno neither provides nor prevents that.

        CommandProvider(["claude", "-p"])
        CommandProvider(["ollama", "run", "llama3"])
    """

    name = "command"

    def __init__(self, argv, model=None, timeout=TIMEOUT):
        if isinstance(argv, str):
            argv = argv.split()
        if not argv:
            raise ValueError("CommandProvider needs a command to run")
        self.argv, self.timeout = list(argv), timeout
        self.model = model or " ".join(self.argv)
        self.web = "the agent's own, if it has any"

    def complete(self, prompt: str) -> str:
        try:
            r = subprocess.run(self.argv, input=prompt, capture_output=True, text=True,
                               timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"agent command failed: {type(e).__name__}: {e}") from e
        if r.returncode != 0:
            raise RuntimeError(f"agent command exited {r.returncode}: "
                               f"{(r.stderr or '').strip()[:400]}")
        return r.stdout


class HTTPProvider:
    """OpenAI-compatible or Anthropic HTTP APIs, over the standard library.

    Deliberately not the vendor SDKs: a classifier that is a pure function should not drag two
    API clients into every install. The key comes from the environment, never from an argument
    and never from a file this package reads.
    """

    name = "http"
    PRESETS = {
        "openai": {"url": "https://api.openai.com/v1/chat/completions",
                   "env": "OPENAI_API_KEY", "style": "openai"},
        "anthropic": {"url": "https://api.anthropic.com/v1/messages",
                      "env": "ANTHROPIC_API_KEY", "style": "anthropic"},
    }

    def __init__(self, preset="openai", model=None, url=None, env=None, style=None,
                 temperature=0.0, timeout=TIMEOUT, max_tokens=1024, web=False):
        p = dict(self.PRESETS.get(preset, {}))
        self.url = url or p.get("url")
        self.style = style or p.get("style", "openai")
        self.env = env or p.get("env", "")
        self.model = model or ("gpt-4o-mini" if self.style == "openai"
                               else "claude-sonnet-4-5")
        self.temperature, self.timeout, self.max_tokens = temperature, timeout, max_tokens
        self.name = f"http:{preset}"
        if not self.url:
            raise ValueError(f"unknown preset {preset!r} and no --url given; "
                             f"known: {sorted(self.PRESETS)}")
        if web and self.style not in WEB_TOOL:
            raise ValueError(
                f"web search was requested but this package has no tool block for style "
                f"{self.style!r}. Refused rather than ignored: a run that believed it searched "
                f"and did not is a run whose answers cannot be read.")
        self.web = bool(web)
        self.key = os.environ.get(self.env, "")
        if not self.key:
            raise RuntimeError(
                f"no API key: set {self.env}. scAnno never stores a key and never reads one "
                f"from a config file.")

    def complete(self, prompt: str) -> str:
        if self.style == "anthropic":
            body = {"model": self.model, "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [{"role": "user", "content": prompt}]}
            hdr = {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        else:
            body = {"model": self.model, "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
            hdr = {"Authorization": f"Bearer {self.key}", "content-type": "application/json"}
        if self.web:
            body["tools"] = WEB_TOOL[self.style]
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{self.name} HTTP {e.code}: "
                               f"{e.read().decode()[:400]}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(f"{self.name} unreachable: {e}") from None
        if self.style == "anthropic":
            return "".join(b.get("text", "") for b in out.get("content", [])
                           if b.get("type") == "text" or "text" in b)
        return out["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- what the corpus knows


def marker_evidence(db, species, tissue, genes, limit=EVIDENCE_TYPES):
    """Which cell types this tissue's corpus associates with these genes, and how strongly.

    This is the classifier's own evidence rendered in words, so the model reasons over the same
    material rather than from memory alone. Ordered by evidence tier then by how many of the
    cluster's genes each type claims - a type citing eight of the top thirty at tier 1 is a
    different proposition from one citing a single gene at tier 5, and the model is shown which.
    """
    up = [str(g).upper() for g in genes]
    q = ",".join("?" * len(up))
    sql = (f"SELECT cell_name, cellontology_id, MIN(evidence_tier) AS tier, "
           f"COUNT(DISTINCT symbol_norm) AS n_hit, "
           f"GROUP_CONCAT(DISTINCT symbol_display) AS hits "
           f"FROM assertion WHERE species=? AND tissue_class=? AND symbol_norm IN ({q}) "
           f"GROUP BY cell_name, cellontology_id ORDER BY tier ASC, n_hit DESC LIMIT ?")
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
            rows = c.execute(sql, [species, tissue, *up, limit]).fetchall()
    except sqlite3.Error as e:
        return [], f"corpus lookup failed: {e}"
    return [{"cell_name": r[0], "cl": r[1], "tier": r[2], "n_hit": r[3],
             "hits": (r[4] or "").split(",")[:8]} for r in rows], ""


def standard_names(db, species, tissue, limit=400):
    """The standard vocabulary for this tissue: cell names with their Cell Ontology ids.

    Offered to the model as available names, not as a closed list. Its purpose is to make an
    off-tree answer LANDABLE - a model that says "cardiac neuron" against a tree with no such
    node should produce a resolvable standard name rather than a string nobody can index.
    """
    sql = ("SELECT DISTINCT cell_name, cellontology_id FROM assertion "
           "WHERE species=? AND tissue_class=? AND cell_name IS NOT NULL "
           "ORDER BY cell_name LIMIT ?")
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
            return [{"cell_name": a, "cl": b} for a, b in c.execute(sql, (species, tissue, limit))]
    except sqlite3.Error:
        return []


# ---------------------------------------------------------------------- resolving a reply


def tree_nodes(tree):
    ch = tree.get("children", {})
    return sorted(({n for kids in ch.values() for n in kids} | set(ch)) - {"root"})


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _depths(tree):
    """Depth of every node below root, for preferring the more specific of two matches."""
    ch, out, stack = tree.get("children", {}), {}, [("root", 0)]
    while stack:
        n, d = stack.pop()
        for k in ch.get(n, []):
            if k not in out or d + 1 > out[k]:
                out[k] = d + 1
                stack.append((k, d + 1))
    return out


def resolve_label(text, tree, ontology=None):
    """Free text -> (tier, label, cl_id). Never coerced, never discarded.

    tree      matches a node's declared synonym patterns, or the node name itself
    ontology  matches a CellMarker cell type for this tissue -> its name and Cell Ontology id
    proposal  neither; kept verbatim, because a cell type this project has no word for is
              exactly the thing worth reading, and the alternative is deleting it
    """
    t = _norm(text)
    if not t or t == _norm(NONE_LABEL):
        return "none", NONE_LABEL, ""
    # exact node name first, so a tree node never loses to a looser pattern elsewhere
    for n in tree_nodes(tree):
        if _norm(n) == t:
            return "tree", n, ""
    # Prefer the DEEPEST matching node, then the longest pattern. Ranking by pattern length
    # alone sent "ventricular cardiomyocytes" to Cardiomyocyte, because the parent's pattern
    # "cardiomyocyte" (13 chars) outscored the child's "ventricular" (11) - the more specific
    # answer losing to the less specific one on a spelling accident.
    depth = _depths(tree)
    best = None
    for node, pats in (tree.get("patterns") or {}).items():
        for p in pats:
            np_ = _norm(p)
            if np_ and np_ in t:
                rank = (depth.get(node, 0), len(np_))
                if best is None or rank > best[0]:
                    best = (rank, node)
    if best:
        return "tree", best[1], ""
    for e in ontology or []:
        if _norm(e["cell_name"]) == t:
            return "ontology", e["cell_name"], e.get("cl") or ""
    for e in ontology or []:
        if _norm(e["cell_name"]) and _norm(e["cell_name"]) in t:
            return "ontology", e["cell_name"], e.get("cl") or ""
    return "proposal", str(text).strip(), ""


# ---------------------------------------------------------------------------- the ask


def cluster_markers(Z, usable, genes, top_n=TOP_N):
    """Each cluster's most distinctive genes, from the profile the classifier already built.

    No DE test is run. `Z` is the cluster profile standardised against the gene background, so
    its largest entries are the genes unusual here relative to what those genes do across cell
    types - what a DE test approximates, without the composition dependence of using the rest of
    the object as the comparison group. The agent and the classifier then see the same evidence.
    """
    g = np.asarray(genes)[usable]
    return [[str(x) for x in g[np.argsort(-Z[c][usable])[:top_n]]] for c in range(Z.shape[0])]


def make_prompt(markers, tree, evidence=None, ontology=None, species="", tissue="", assay="",
                cluster=None, web=False):
    """Genes, what the corpus knows about them, the taxonomy, the standard names. Not the call."""
    nodes = tree_nodes(tree)
    who = " ".join(x for x in (species, tissue) if x)
    p = [f"Annotate one single-{'nucleus' if assay in ('sn', 'snrna') else 'cell'} RNA-seq "
         f"cluster{'' if cluster is None else f' (id {cluster})'}"
         f"{' from ' + who if who else ''}.",
         "",
         f"MOST DISTINCTIVE GENES (most distinctive first):\n{', '.join(markers)}", ""]
    if evidence:
        p.append("WHAT A CURATED MARKER DATABASE SAYS about these genes in this tissue "
                 "(tier 1 is strongest evidence):")
        for e in evidence:
            p.append(f"  - {e['cell_name']} [{e.get('cl') or 'no CL id'}] tier {e['tier']}, "
                     f"{e['n_hit']} of the genes above: {', '.join(e['hits'])}")
        p.append("")
    p.append("PREFERRED TAXONOMY - use one of these names if the cluster fits one:")
    p += [f"  - {n}" for n in nodes]
    p.append("")
    if ontology:
        p.append(f"If it does NOT fit the taxonomy, use a standard cell type name. Names known "
                 f"for this tissue ({len(ontology)}), with Cell Ontology ids:")
        p += [f"  - {e['cell_name']} [{e.get('cl') or ''}]" for e in ontology[:120]]
        if len(ontology) > 120:
            p.append(f"  ... and {len(ontology) - 120} more; any standard name is acceptable")
        p.append("")
    if web:
        p.append("You may search the web for recent marker evidence. Cite what you used in "
                 "`reason`.")
        p.append("")
    p += [
        "Judge the evidence yourself. The database above is incomplete - it is a curated corpus, "
        "not the literature - so disagree with it where the genes warrant, and say why.",
        f"If nothing fits, answer {NONE_LABEL} rather than forcing a choice.",
        "",
        "Reply with JSON only, no other text:",
        '{"label": "<taxonomy name, standard cell type name, or ' + NONE_LABEL + '>", '
        '"confidence": <0.0-1.0>, "reason": "<one or two sentences>"}',
    ]
    return "\n".join(p)


def parse_reply(text):
    """Reply -> (raw label, confidence, reason, error). Resolution happens separately."""
    if not text or not str(text).strip():
        return None, float("nan"), "", "empty reply"
    m = re.search(r"\{.*\}", str(text), re.S)
    if not m:
        return None, float("nan"), "", f"no JSON object in reply: {str(text)[:120]!r}"
    try:
        d = json.loads(m.group(0))
    except ValueError as e:
        return None, float("nan"), "", f"unparseable JSON: {e}"
    lab = str(d.get("label", "")).strip()
    if not lab:
        return None, float("nan"), str(d.get("reason", "")), "reply carries no label"
    try:
        conf = float(d.get("confidence"))
    except (TypeError, ValueError):
        conf = float("nan")
    return lab, conf, str(d.get("reason", "")), ""


def ask_cluster(provider, markers, tree, evidence=None, ontology=None, votes=1, **kw):
    """Ask about one cluster `votes` times; report the answer and how often it recurred."""
    prompt = make_prompt(markers, tree, evidence, ontology, web=getattr(provider, "web", False),
                         **kw)
    key = hashlib.sha256((str(getattr(provider, "model", "?")) + "\n"
                          + prompt).encode()).hexdigest()
    replies = [provider.complete(prompt) for _ in range(max(1, votes))]
    parsed = [parse_reply(r) for r in replies]
    good = [p for p in parsed if p[0] is not None]
    base = {"votes": len(replies), "prompt_sha": key[:16],
            "raw": replies[0][:2000] if replies else ""}
    if not good:
        return {**base, "label": None, "resolved": None, "tier": "error", "cl": "",
                "confidence": float("nan"), "reason": "", "error": parsed[0][3],
                "consensus": 0.0}
    res = [resolve_label(p[0], tree, ontology) for p in good]
    keys = [f"{t}:{lab}" for t, lab, _ in res]
    vals, counts = np.unique(keys, return_counts=True)
    top = str(vals[int(np.argmax(counts))])
    i = keys.index(top)
    return {**base, "label": good[i][0], "resolved": res[i][1], "tier": res[i][0],
            "cl": res[i][2], "confidence": good[i][1], "reason": good[i][2], "error": "",
            "consensus": float(counts.max() / len(replies)), "raw": replies[i][:2000]}


def annotate_agentic(provider, Z, usable, genes, tree, db=None, species="", tissue="",
                     assay="", top_n=TOP_N, votes=1, corpus_calls=None, log=None,
                     ontology=None):
    """Every cluster, asked. Returns one row per cluster; writes nothing.

    `ontology` is the standard-name vocabulary. Given explicitly it is used as given; otherwise
    it is read from `db`. Both routes exist because a caller may hold a curated list without a
    CellMarker database, and because a test must be able to supply one without a 624 MB file.

    `corpus_calls` is used ONLY to report agreement afterwards and is never put in a prompt.
    """
    marks = cluster_markers(Z, usable, genes, top_n)
    if ontology is None:
        ontology = standard_names(db, species, tissue) if db else []
    rows = []
    for c, mk in enumerate(marks):
        ev, _ = marker_evidence(db, species, tissue, mk) if db else ([], "")
        try:
            r = ask_cluster(provider, mk, tree, ev, ontology, votes=votes, species=species,
                            tissue=tissue, assay=assay, cluster=c)
        except RuntimeError as e:
            r = {"label": None, "resolved": None, "tier": "error", "cl": "",
                 "confidence": float("nan"), "reason": "", "error": str(e)[:300],
                 "consensus": 0.0, "votes": 0, "prompt_sha": "", "raw": ""}
        r.update({"cluster": c, "provider": provider.name,
                  "model": getattr(provider, "model", ""), "top_genes": ";".join(mk[:10])})
        if corpus_calls is not None and c < len(corpus_calls):
            call = corpus_calls[c]
            path = (call.get("path") if isinstance(call, dict) else str(call)) or ""
            r["corpus_path"] = path
            # Comparable only where the agent landed on the SAME taxonomy. An ontology or
            # proposal answer is not a disagreement with the tree - it is a statement that the
            # tree has no word for this - and scoring it as wrong would punish the discovery
            # this route exists to permit.
            r["comparable"] = r["tier"] == "tree"
            r["agrees"] = bool(r["comparable"] and r["resolved"] in path.split("/"))
        rows.append(r)
        if log:
            log(f"  cluster {c:>4}  [{r['tier']:<9}] {str(r['resolved'] or 'ERROR'):<32} "
                f"consensus {r['consensus']:>4.0%}"
                + (f"   corpus {r.get('corpus_path', '')}" if corpus_calls is not None else "")
                + (f"   {r['error']}" if r["error"] else ""))
    return rows


def agreement(rows):
    """Agreement on the taxonomy, plus what the agent proposed that the taxonomy lacks."""
    comp = [r for r in rows if r.get("comparable") and r.get("corpus_path")]
    off = [r for r in rows if r.get("tier") in ("ontology", "proposal")]
    return {
        "n_comparable": len(comp),
        "agree": (sum(r["agrees"] for r in comp) / len(comp)) if comp else float("nan"),
        "disagreements": [(r["cluster"], r["corpus_path"], r["resolved"])
                          for r in comp if not r["agrees"]],
        "off_tree": [(r["cluster"], r["tier"], r["resolved"], r["cl"],
                      r.get("corpus_path", "")) for r in off],
    }
