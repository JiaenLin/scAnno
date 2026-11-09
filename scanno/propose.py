"""`scanno tree propose`: a DRAFT tree from the corpus, for the user to approve or edit.

WHY (single-cell-harness B1-098). `annotate` needs a declared taxonomy, and "the tree is yours and it
is the biggest lever" left every new tissue starting from a blank page. The tree stays the user's -
which lineages to split and how deep is a decision about the study - so this writes a draft and
nothing more: rooted, lineage -> type, built from what the corpus itself says about the context.

HOW. The corpus's own grouping is `cell_name_class` (CellMarker files each cell name under a class:
"CD8+ T cell" under "T cell", "Microglial cell" under "Glial cell"). The draft is

    root -> the classes with the most curated markers -> each class's cell names

keeping only cell names with at least `min_markers` distinct markers at tier <= `min_tier` (the
tiers `annotate` reads), at most `max_lineages` classes and `max_types` names per class, ranked by
markers. A class left with one name is a leaf. A name the class shares is folded into the class's
panel. Bare names are unique, as the walk and the scope require. Patterns are the corpus's own
names, lower-cased, so each node's panel is exactly the claims of the names under it. A corpus
without `cell_name_class` gets a flat draft, root -> names, and says so.

WHAT IT IS NOT. Not a taxonomy anybody validated: the classes are the corpus curators' filing, the
ranking is by how STUDIED a type is, not by whether it is in this tissue sample. The draft carries a
`proposed` block naming its rule and corpus, with `approved_by` empty; `annotate` prints REVIEW and
records "tree: proposed draft" in STATUS for as long as nobody has set it.
"""
from __future__ import annotations

import sqlite3

RULE = ("root -> the corpus's cell_name_class values -> their cell names, each name with at least "
        "min_markers distinct markers at tier <= min_tier, ranked by markers, at most max_lineages "
        "classes and max_types names per class; a class with one name is a leaf")


def propose(db, ctx, *, min_tier=4, min_markers=3, max_lineages=10, max_types=6, version=""):
    """(tree, summary) for the corpus context `ctx` (corpus.resolve's). Raises ValueError when the
    corpus has too little for a tree: fewer than two names clear the marker floor."""
    where, params = ctx.where()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(assertion)")}
        has_class = "cell_name_class" in cols
        sel = "COALESCE(cell_name_class, ''), cell_name" if has_class else "'', cell_name"
        rows = con.execute(
            f"SELECT {sel}, COUNT(DISTINCT symbol_norm) FROM assertion WHERE {where} "
            f"AND evidence_tier<=? AND symbol_norm!='' AND cell_name IS NOT NULL "
            f"GROUP BY 1, 2", (*params, min_tier)).fetchall()
        sha = None
        if "corpus_info" in {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}:
            sha = dict(con.execute("SELECT key, value FROM corpus_info")).get("source_sha256")
    finally:
        con.close()

    # One home per name: where it has the most markers (then the class name, for a stable tie).
    best = {}
    for cls, name, n in rows:
        if n < min_markers or not str(name).strip() or str(name).lower() == "root":
            continue
        cur = best.get(name)
        if cur is None or (n, str(cls)) > (cur[1], str(cur[0])):
            best[name] = (cls, n)
    if len(best) < 2:
        raise ValueError(f"only {len(best)} cell name(s) in this context have {min_markers} or more "
                         f"markers at tier <= {min_tier}; a tree needs two at the root")

    by_class = {}
    for name, (cls, n) in best.items():
        by_class.setdefault(cls if has_class and str(cls).strip() else name, []).append((n, name))
    ranked = sorted(by_class.items(), key=lambda kv: (-sum(n for n, _ in kv[1]), str(kv[0])))
    ranked = ranked[:max_lineages]
    lineages = [c for c, _ in ranked]
    taken = {c.lower() for c in lineages}

    children, patterns = {"root": []}, {}
    for cls, members in ranked:
        names = [nm for _n, nm in sorted(members, key=lambda t: (-t[0], t[1]))][:max_types]
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
        raise ValueError("the corpus files every kept name under one class; a rooted draft needs "
                         "two lineages - edit one by hand from `scanno panel`'s list")
    summary = {"lineages": len(children["root"]),
               "types": sum(len(v) for k, v in children.items() if k != "root"),
               "lineage_from": "cell_name_class" if has_class else "none: a flat draft, root -> names"}
    tree = {
        "children": children, "patterns": patterns, "members": {},
        "proposed": {
            "by": "scanno tree propose", "version": version, "rule": RULE,
            "corpus": str(db), "corpus_sha256": sha,
            "species": list(ctx.species),
            "tissues": list(ctx.tissues) if ctx.tissues is not None else "every tissue",
            "min_tier": min_tier, "min_markers": min_markers, "max_lineages": max_lineages,
            "max_types": max_types, "lineage_from": summary["lineage_from"],
            "approved_by": None,
            "note": ("A DRAFT for you to approve or edit. Set approved_by to your name once you "
                     "have read it; until then every annotate run that uses it prints REVIEW and "
                     "records 'tree: proposed draft' in STATUS."),
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
