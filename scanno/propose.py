"""`scanno tree propose`: a DRAFT tree from the corpus, for the user to approve or edit.

WHY (single-cell-harness B1-098). `annotate` needs a declared taxonomy, and "the tree is yours and it
is the biggest lever" left every new tissue starting from a blank page. The tree stays the user's -
which lineages to split and how deep is a decision about the study - so this writes a draft and
nothing more: rooted, lineage -> type, built from what the corpus itself says about the context.

HOW. The corpus's own grouping is `cell_name_class` (CellMarker files each cell name under a class:
"CD8+ T cell" under "T cell", "Microglial cell" under "Glial cell"). The draft is

    root -> the tissue's own lineages -> each lineage's cell names

A tissue's studies also report the tumours that grow in it, the immune cells that infiltrate them,
and the blood and vessels every tissue carries (GAP-C85: a draft that took every class those
studies mention drafted lineages the sample cannot hold, and cells walked to them). So a lineage is
the tissue's OWN only when two tests on the corpus's own fields hold - disease, tissue_class and
cell_name_class, with nothing about any tissue or lineage written here, so they hold for any tissue:

  (a) HEALTH: at least half its claims in the context come from healthy samples (a claim is one
      assertion - tissue, name, marker - and it is healthy when the corpus records it from a sample
      whose disease field is Normal). A lineage reported mostly from disease samples is the disease's.
  (b) SPECIFICITY: the context's healthy claims file it, as a share of all their claims, at least as
      often as the healthy claims of every tissue of the species do (enrichment >= 1). A lineage the
      tissue reports less often than the corpus at large is shared - more common elsewhere.

A name is placed under the kept lineage where it has the most markers, keeping only names with at
least `min_markers` distinct markers at tier <= `min_tier` (the tiers `annotate` reads) and with at
least half their claims there healthy; at most `max_lineages` lineages and `max_types` names per
lineage, ranked by markers. A lineage left with one name is a leaf. A name the lineage shares is
folded into the lineage's panel. Bare names are unique, as the walk and the scope require. Patterns
are the corpus's own names, lower-cased, so each node's panel is exactly the claims of the names
under it. Every lineage set aside and every name that cleared the marker floor but was not placed is
written into the draft's `proposed` block with its numbers and why, for the person who approves it.
A corpus without `cell_name_class` gets a flat draft, root -> names, and says so; a corpus without
disease counts skips (a) and says so; a context that reads every tissue passes (b) by definition.

WHAT IT IS NOT. Not a taxonomy anybody validated: the classes are the corpus curators' filing, the
ranking is by how STUDIED a type is, and "the tissue's own" is the tissue's own in the corpus - a
sample that is not the tissue (a culture, an organoid) may hold fewer. The draft carries a
`proposed` block naming its rule and corpus, with `approved_by` empty; `annotate` prints REVIEW and
records "tree: proposed draft" in STATUS for as long as nobody has set it.
"""
from __future__ import annotations

import sqlite3

RULE = ("root -> the tissue's own lineages (cell_name_class) -> their cell names. A lineage is the "
        "tissue's own when at least half its claims in the context come from healthy samples "
        "(disease Normal) and the context's healthy claims file it at least as often, as a share, "
        "as the healthy claims of every tissue of the species (enrichment >= 1). A name is placed "
        "under the kept lineage where it has the most markers, with at least min_markers distinct "
        "markers at tier <= min_tier and at least half its claims there healthy; at most "
        "max_lineages lineages and max_types names per lineage, ranked by markers; a lineage with "
        "one name is a leaf; what cleared the floor and was not placed is listed with why")


def _groups(con, where, params, sel, health, min_tier):
    """(lineage, name, markers, healthy markers, claims, healthy claims) per corpus grouping."""
    return con.execute(
        f"SELECT {sel}, cell_name, COUNT(DISTINCT symbol_norm), "
        f"COUNT(DISTINCT CASE WHEN {health} THEN symbol_norm END), COUNT(*), "
        f"SUM(CASE WHEN {health} THEN 1 ELSE 0 END) FROM assertion WHERE {where} "
        f"AND evidence_tier<=? AND symbol_norm!='' AND cell_name IS NOT NULL "
        f"GROUP BY 1, 2", (*params, min_tier)).fetchall()


def propose(db, ctx, *, min_tier=4, min_markers=3, max_lineages=10, max_types=6, version=""):
    """(tree, summary) for the corpus context `ctx` (corpus.resolve's). Raises ValueError when the
    corpus has too little for a tree: fewer than two names clear the marker floor under lineages
    that are the tissue's own, or they all sit under one lineage."""
    where, params = ctx.where()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(assertion)")}
        has_class = "cell_name_class" in cols
        has_health = "n_normal" in cols
        sel = "COALESCE(cell_name_class, '')" if has_class else "''"
        health = "n_normal>0" if has_health else "1=1"
        rows = _groups(con, where, params, sel, health, min_tier)
        every_tissue = ctx.tissues is None
        wide = rows if every_tissue else _groups(
            con, f"species IN ({','.join('?' * len(ctx.species))})", list(ctx.species), sel,
            health, min_tier)
        sha = None
        if "corpus_info" in {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}:
            sha = dict(con.execute("SELECT key, value FROM corpus_info")).get("source_sha256")
    finally:
        con.close()

    def key(cls, name):
        return cls if has_class and str(cls).strip() else name

    # THE LINEAGE TESTS, on every claim the context files under each lineage (all names).
    claims, healthy, wide_healthy = {}, {}, {}
    for cls, name, _n, _nh, c, h in rows:
        k = key(cls, name)
        claims[k] = claims.get(k, 0) + c
        healthy[k] = healthy.get(k, 0) + (h or 0)
    for cls, name, _n, _nh, _c, h in wide:
        k = key(cls, name)
        wide_healthy[k] = wide_healthy.get(k, 0) + (h or 0)
    h_ctx, h_wide = sum(healthy.values()), sum(wide_healthy.values())
    stats, aside = {}, {}
    for k in claims:
        e = (1.0 if every_tissue else
             ((healthy[k] / h_ctx) / (wide_healthy[k] / h_wide)
              if h_ctx and h_wide and wide_healthy.get(k) else 0.0))
        stats[k] = {"lineage": k, "claims": claims[k], "healthy": healthy[k],
                    "enrichment": round(e, 3)}
        if has_health and 2 * healthy[k] < claims[k]:
            aside[k] = (f"disease: {claims[k] - healthy[k]} of its {claims[k]} claims here come only "
                        f"from disease samples")
        elif e < 1:
            aside[k] = (f"shared: enrichment {e:.2f} - this context's healthy samples report it less "
                        f"often than healthy samples across the species' tissues do")

    # THE NAMES. One home per name: the kept lineage where it has the most markers (then the
    # lineage's name, for a stable tie). The rest are recorded, with the most-markered filing.
    best, unplaced = {}, {}

    def miss(name, k, n, nh, why):
        cur = unplaced.get(name)
        if cur is None or (n, str(k)) > (cur["markers"], str(cur["lineage"])):
            unplaced[name] = {"name": name, "lineage": k, "markers": n, "healthy_markers": nh,
                              "reason": why}

    for cls, name, n, nh, _c, _h in rows:
        if n < min_markers or not str(name).strip() or str(name).lower() == "root":
            continue
        k = key(cls, name)
        if k in aside:
            miss(name, k, n, nh, f"its lineage is set aside ({aside[k]})")
        elif has_health and 2 * nh < n:
            miss(name, k, n, nh, f"disease: {n - nh} of its {n} markers here come only from "
                                 f"disease samples")
        else:
            cur = best.get(name)
            if cur is None or (n, str(k)) > (cur[1], str(cur[0])):
                best[name] = (k, n, nh)
    for name in best:
        unplaced.pop(name, None)
    if len(best) < 2:
        held = (f"; {len(aside)} lineage(s) the context mentions are set aside as not the tissue's "
                f"own (health and specificity)" if aside else "")
        raise ValueError(f"only {len(best)} cell name(s) in this context have {min_markers} or more "
                         f"markers at tier <= {min_tier} under the tissue's own lineages{held}; a "
                         f"tree needs two at the root")

    by_class = {}
    for name, (k, n, _nh) in best.items():
        by_class.setdefault(k, []).append((n, name))
    ranked = sorted(by_class.items(), key=lambda kv: (-sum(n for n, _ in kv[1]), str(kv[0])))
    for k, members in ranked[max_lineages:]:
        aside[k] = f"beyond max_lineages={max_lineages}"
        for n, name in members:
            miss(name, k, n, best[name][2], f"its lineage is beyond max_lineages={max_lineages}")
    ranked = ranked[:max_lineages]
    lineages = [c for c, _ in ranked]
    taken = {c.lower() for c in lineages}

    children, patterns = {"root": []}, {}
    for cls, members in ranked:
        ordered = [nm for _n, nm in sorted(members, key=lambda t: (-t[0], t[1]))]
        names = ordered[:max_types]
        for nm in ordered[max_types:]:
            if nm.lower() not in taken:
                miss(nm, cls, best[nm][1], best[nm][2], f"beyond max_types={max_types} under {cls}")
        kids = [nm for nm in names if nm.lower() != cls.lower() and nm.lower() not in taken]
        children["root"].append(cls)
        # the class's own name and the names filed under it - never another lineage's name,
        # which would pull that lineage's claims into this panel
        patterns[cls] = sorted({cls.lower()} | {nm.lower() for nm in names
                                                if nm.lower() == cls.lower() or nm in kids})
        if len(kids) >= 2:
            children[cls] = kids
            for nm in kids:
                patterns[nm] = [nm.lower()]
                taken.add(nm.lower())
    if len(children["root"]) < 2:
        raise ValueError("the corpus files every kept name under one of the tissue's own lineages; a "
                         "rooted draft needs two lineages - edit one by hand from `scanno panel`'s list"
                         + (f" (set aside: {', '.join(sorted(map(str, aside)))})" if aside else ""))
    placed_somewhere = {nm for nms in children.values() for nm in nms} | set(lineages)
    named_aside = {r["lineage"] for r in unplaced.values()}
    set_aside = [dict(stats[k], reason=aside[k]) for k in sorted(
        aside, key=lambda k: (-stats[k]["claims"], str(k))) if k in named_aside]
    not_placed = sorted((r for nm, r in unplaced.items() if nm not in placed_somewhere),
                        key=lambda r: (str(r["lineage"]), -r["markers"], str(r["name"])))
    lineage_tests = {
        "health": "applied" if has_health else
                  "not applied: this corpus records no disease counts (n_normal)",
        "specificity": "not applied: the context reads every tissue" if every_tissue else "applied"}
    summary = {"lineages": len(children["root"]),
               "types": sum(len(v) for k, v in children.items() if k != "root"),
               "lineage_from": "cell_name_class" if has_class else "none: a flat draft, root -> names",
               "set_aside": set_aside, "not_placed": len(not_placed),
               "lineage_tests": lineage_tests}
    tree = {
        "children": children, "patterns": patterns, "members": {},
        "proposed": {
            "by": "scanno tree propose", "version": version, "rule": RULE,
            "corpus": str(db), "corpus_sha256": sha,
            "species": list(ctx.species),
            "tissues": list(ctx.tissues) if ctx.tissues is not None else "every tissue",
            "min_tier": min_tier, "min_markers": min_markers, "max_lineages": max_lineages,
            "max_types": max_types, "lineage_from": summary["lineage_from"],
            "lineage_tests": lineage_tests,
            "kept": [stats[k] for k in lineages],
            "set_aside": set_aside,
            "not_placed": not_placed,
            "approved_by": None,
            "note": ("A DRAFT for you to approve or edit. Set approved_by to your name once you "
                     "have read it; until then every annotate run that uses it prints REVIEW and "
                     "records 'tree: proposed draft' in STATUS. 'set_aside' and 'not_placed' list "
                     "what the corpus mentions here and the draft left out, with why: add back "
                     "what your sample holds."),
        },
    }
    return tree, summary


def provenance(tree):
    """What STATUS records for a tree: None for a tree nobody proposed, else whether it was
    approved and by whom. Read by `annotate`."""
    prop = tree.get("proposed") if isinstance(tree, dict) else None
    if not isinstance(prop, dict):
        return None
    who = prop.get("approved_by")
    return {"source": "proposed draft" + (f", approved by {who}" if who else ""),
            "approved_by": who or None, "rule": prop.get("rule"),
            "corpus_sha256": prop.get("corpus_sha256")}
