"""`scanno tree propose`: a DRAFT tree from the corpus, for the user to approve or edit.

WHY (single-cell-harness B1-098). `annotate` needs a declared taxonomy, and "the tree is yours and it
is the biggest lever" left every new tissue starting from a blank page. The tree stays the user's -
which lineages to split and how deep is a decision about the study - so this writes a draft and
nothing more: rooted, lineage -> type, built from what the corpus itself says about the context.

HOW. The corpus's own grouping is `cell_name_class` (CellMarker files each cell name under a class:
"CD8+ T cell" under "T cell", "Microglial cell" under "Glial cell"). The draft is

    root -> the tissue's own lineages -> each lineage's cell names

A tissue's studies also report the tumours that grow in it and the cells that infiltrate them
(GAP-C85: a draft that took every class those studies mention drafted lineages the sample cannot
hold, and cells walked to them). So a lineage is drafted only when the corpus's own fields - each
row's study, disease, tissue_class and cell_name_class - say it is the tissue's own.

THE UNIT IS THE STUDY (a PMID), so one large panel cannot outvote the studies beside it: a study
counts once however many markers it lists. A REPORT is one study's report of a lineage in one of the
context's tissues, counted from the rows the studies themselves reported (the corpus's `curated`
sources) at any marker tier: the question is whether studies report the lineage, not how strong its
markers are (the marker floor below is the tier's job). A report is HEALTHY when one of its rows
comes from a sample the corpus records as healthy (the release's word Normal, which build-markers
reads in any case, trimmed), DISEASE when one comes from any other sample; one study can be both. A
study's report is HEALTHY-ONLY when every row it reported in that tissue is healthy: the study
sampled no disease there. With HT healthy studies in the context, H the lineage's and Hp its
healthy-only ones, four tests, each threshold a named parameter (DEFAULTS; the CLI's flags; recorded
as values in the draft's `proposed.parameters`, and named on stdout when not the default):

  (1) EVIDENCE     H >= min_studies (2): no lineage rests on one study.
  (2) SHARE        H / HT >= min_share (0.02): at least one healthy study of the context in fifty
                   reports it.
  (3) SPECIFICITY  e = (H / HT) / (W_L / W) >= min_enrichment (1.0), W the healthy reports of every
                   tissue of the species (the context's included), W_L the lineage's: the context's
                   healthy studies report it at least as often as healthy studies across the
                   species do. Not applied when the context reads every tissue.
  (4) DISEASE      Hp >= min_studies: at least min_studies of its healthy studies are healthy-only.
                   A lineage whose healthy reports all come from studies that also sampled the
                   disease (tumour-adjacent tissue) is the disease's. The test never asks how often
                   disease studies report a lineage, so a disease that expands a tissue's own cells
                   (fat cells in obesity, fibroblasts in fibrosis) cannot remove them while
                   min_studies healthy-only studies report them. It adds no threshold. The disease
                   share, r = (D / DT) / (H / HT), is reported in every entry and tested nowhere.

Nothing about any tissue, lineage or name is written here, so the tests hold for any tissue. WHAT
THEY CANNOT DO: tell a visitor that the context's healthy studies report as often as healthy studies
elsewhere (cells in transit are reported by the healthy studies of most tissues) from a resident
lineage, or a tumour lineage that healthy-only studies of the tissue report from one that is the
tissue's own. Those are the person's edit, and every entry carries the numbers that decided it.

A name is placed under the kept lineage where it has the most markers, keeping only names with at
least `min_markers` distinct markers at tier <= `min_tier` there (the tiers `annotate` reads), at
least min_studies healthy studies reporting the name in the context, and not the disease's by (4);
lineages are ranked by healthy studies, names by healthy studies then markers, at most
`max_lineages` lineages and `max_types` names per lineage. A lineage left with one name is a leaf. A
name the lineage shares is folded into the lineage's panel. Bare names are unique, as the walk and
the scope require. Patterns are the corpus's own names, lower-cased, so each node's panel is exactly
the claims of the names under it. THE VERDICT IS WRITTEN WHOLE into the draft's `proposed` block,
every entry with its studies, healthy, healthy-only and disease studies, claims and healthy claims:
`kept` (in the root), `capped` (passed every test, cut by max_lineages, with its rank), `no_panel`
(passed every test, no name placed under it), `set_aside` (failed a test, with why; those that held
a name clearing the marker floor) and `not_placed` (names clearing the floor, with why).

WHEN THE CORPUS CANNOT TELL, it refuses, saying why. A corpus with no per-study rows (no `v_marker`:
not built by `scanno build-markers`), a release in which no row of the species reads as healthy (a
blank disease field, or only other words such as "Healthy"; the refusal names the field and what it
holds), and a context with fewer healthy studies than min_studies are refused. `--lineage-tests off`
is the one way to draft without the tests: every lineage the context mentions is drafted unchecked,
and it is an ESCAPE, recorded in STATUS `escapes`, in the draft's `proposed.lineage_tests`, and by
`annotate` for every run on that draft. A context with fewer than 1/min_share healthy studies is
drafted with a `cannot_tell` line: there one study is more than the share. A corpus without
`cell_name_class` gets a flat draft, root -> names, and says so.

WHAT IT IS NOT. Not a taxonomy anybody validated: the classes are the corpus curators' filing, the
ranking is by how often the tissue's healthy studies report a type, and "the tissue's own" is the
tissue's own in the corpus - a sample that is not the tissue (a culture, an organoid) may hold fewer.
The draft carries a `proposed` block naming its rule, its parameters and its corpus, with
`approved_by` empty; `annotate` prints REVIEW and records "tree: proposed draft" in STATUS for as
long as nobody has set it.
"""
from __future__ import annotations

import math
import sqlite3

# THE DECLARED THRESHOLDS: the CLI's defaults, recorded as values in every draft.
DEFAULTS = {"min_studies": 2, "min_share": 0.02, "min_enrichment": 1.0}
# the draft's shape: the CLI's defaults too, so a value that is not one of them is named
SHAPE = {"min_tier": 4, "min_markers": 3, "max_lineages": 10, "max_types": 6}
PARAMETERS = {
    "min_studies": "a lineage, and a name, needs at least this many distinct healthy studies "
                   "reporting it in the context, and at least this many of them healthy-only "
                   "(studies that sampled no disease there) (2: none rests on one study)",
    "min_share": "a lineage needs at least this share of the context's healthy studies (0.02: one "
                 "in fifty)",
    "min_enrichment": "a lineage's share of the context's healthy studies over its share of the "
                      "healthy reports across the species' tissues must be at least this (1.0: as "
                      "often as elsewhere)",
}

RULE = ("root -> the tissue's own lineages (cell_name_class) -> their cell names, counted in studies "
        "(a study counts once however many markers it lists). A lineage is the tissue's own when at "
        "least min_studies distinct healthy studies (disease Normal) report it in the context, they "
        "are at least min_share of the context's healthy studies, its share of them is at least "
        "min_enrichment times its share of the healthy reports across the species' tissues, and at "
        "least min_studies of its healthy studies are healthy-only (they sampled no disease in the "
        "context). A name is placed under the kept lineage where it has the most markers, with "
        "at least min_markers distinct markers at tier <= min_tier and at least min_studies healthy "
        "and min_studies healthy-only studies; lineages are ranked by healthy studies, "
        "names by healthy studies then markers, at most max_lineages and max_types; a lineage with "
        "one name is a leaf; what cleared the marker floor and was not placed is listed with why")


class CannotTell(ValueError):
    """The corpus cannot tell this context's own lineages; the message says why, `code` which
    case: the context is too thin, the release reads no row as healthy, or it has no study rows."""

    def __init__(self, msg, code="tree_cannot_tell"):
        super().__init__(msg)
        self.code = code


def _words(db, species, tissues, top=4):
    """What the disease field holds among the species' study rows (the context's, when `tissues`
    is given), most rows first, for a refusal to name: "'Healthy' (24 rows), ...". '' when the
    corpus has no such rows."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        q = ("SELECT COALESCE(disease, ''), MAX(is_normal), COUNT(*) FROM v_marker WHERE species IN "
             f"({','.join('?' * len(species))}) AND curated = 1")
        args = list(species)
        if tissues is not None:
            q += f" AND tissue_class IN ({','.join('?' * len(tissues))})"
            args += list(tissues)
        rows = con.execute(q + " GROUP BY 1 ORDER BY 3 DESC, 1", args).fetchall()
    except sqlite3.Error:
        return ""
    finally:
        con.close()
    said = [f"{repr(w) if str(w).strip() else '(blank)'} ({n:,} row{'' if n == 1 else 's'}"
            f"{', read as healthy' if h else ''})" for w, h, n in rows[:top]]
    more = len(rows) - top
    return ", ".join(said) + (f" and {more} more value{'' if more == 1 else 's'}" if more > 0 else "")


def _groups(con, where, params, sel, min_tier):
    """(lineage, name, markers, claims, healthy claims) per corpus grouping, at tier <= min_tier."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(assertion)")}
    health = "n_normal>0" if "n_normal" in cols else "0"
    return con.execute(
        f"SELECT {sel}, cell_name, COUNT(DISTINCT symbol_norm), COUNT(*), "
        f"SUM(CASE WHEN {health} THEN 1 ELSE 0 END) FROM assertion WHERE {where} "
        f"AND evidence_tier<=? AND symbol_norm!='' AND cell_name IS NOT NULL "
        f"GROUP BY 1, 2", (*params, min_tier)).fetchall()


def _reports(con, species):
    """The studies' own reports for the species: a list of (tissue_class, class, name, healthy,
    study), one per distinct combination; None when the corpus has no per-study rows."""
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('view','table')")}
    if "v_marker" not in names:
        return None
    cols = {r[1] for r in con.execute("PRAGMA table_info(v_marker)")}
    if not {"curated", "pmid", "is_normal", "cell_name_class", "tissue_class"} <= cols:
        return None
    return con.execute(
        "SELECT DISTINCT tissue_class, COALESCE(cell_name_class, ''), cell_name, is_normal, "
        "COALESCE(NULLIF(pmid, ''), NULLIF(title, ''), '?') FROM v_marker "
        f"WHERE species IN ({','.join('?' * len(species))}) AND curated = 1 "
        "AND symbol_norm != '' AND cell_name IS NOT NULL", list(species)).fetchall()


def _ratio(num, den):
    return num / den if den else 0.0


def propose(db, ctx, *, min_tier=4, min_markers=3, max_lineages=10, max_types=6,
            min_studies=DEFAULTS["min_studies"], min_share=DEFAULTS["min_share"],
            min_enrichment=DEFAULTS["min_enrichment"], lineage_tests=True, version=""):
    """(tree, summary) for the corpus context `ctx` (corpus.resolve's). Raises CannotTell (its
    `code` says which) when the corpus has no per-study rows, when no row of the species reads as
    healthy, or when the context has fewer healthy studies than a lineage needs (unless
    `lineage_tests` is off), and ValueError when the corpus has too little for a tree: fewer than
    two names clear the floors, or they all sit under one lineage."""
    where, params = ctx.where()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(assertion)")}
        has_class = "cell_name_class" in cols
        sel = "COALESCE(cell_name_class, '')" if has_class else "''"
        rows = _groups(con, where, params, sel, min_tier)
        reports = _reports(con, ctx.species)
        sha = None
        if "corpus_info" in {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}:
            sha = dict(con.execute("SELECT key, value FROM corpus_info")).get("source_sha256")
    finally:
        con.close()
    every_tissue = ctx.tissues is None
    in_ctx = (lambda ti: True) if every_tissue else (lambda ti, s=set(ctx.tissues): ti in s)
    parameters = {"min_tier": min_tier, "min_markers": min_markers, "max_lineages": max_lineages,
                  "max_types": max_types, "min_studies": min_studies, "min_share": min_share,
                  "min_enrichment": min_enrichment, "lineage_tests": "on" if lineage_tests else "off"}

    def key(cls, name):
        return cls if has_class and str(cls).strip() else name

    # THE REPORTS, in studies: per lineage and per name in the context; healthy per lineage across
    # the species. A report is (study, tissue), so a study reporting in two tissues counts in each.
    lin = {}            # lineage -> {"h": set, "d": set, "a": set}
    nam = {}            # name -> the same, any filing, in the context
    wide_h, ctx_h, ctx_d, all_h = {}, set(), set(), set()
    for ti, cls, name, normal, study in reports or []:
        k, rep = key(cls, name), (study, ti)
        if normal:
            all_h.add(rep)
            wide_h.setdefault(k, set()).add(rep)
        if not in_ctx(ti):
            continue
        for d in (lin.setdefault(k, {"h": set(), "d": set(), "a": set()}),
                  nam.setdefault(name, {"h": set(), "d": set(), "a": set()})):
            d["a"].add(rep)
            d["h" if normal else "d"].add(rep)
        (ctx_h if normal else ctx_d).add(rep)
    HT, DT, W = len(ctx_h), len(ctx_d), len(all_h)
    # a HEALTHY-ONLY report: a study's report in one of the context's tissues where every row it
    # reported there comes from a sample recorded as Normal (it sampled no disease there)
    pure = ctx_h - ctx_d
    off_by = ("`--lineage-tests off` drafts every lineage the context mentions, unchecked, and "
              "records that as an escape")

    cannot_tell = []
    if not lineage_tests:
        cannot_tell.append("--lineage-tests off: the lineage tests are not applied, by request (an "
                           "escape, recorded in STATUS); every lineage the context mentions is "
                           "drafted unchecked")
    elif reports is None:
        raise CannotTell(
            "this corpus has no per-study rows (no v_marker view with each row's study, disease, "
            "tissue and class: it was not built by `scanno build-markers`), so it cannot count "
            "studies or read its disease field, and cannot tell this tissue's own lineages; build "
            f"the corpus with `scanno build-markers`, or {off_by}", code="tree_no_study_rows")
    elif not W:
        raise CannotTell(
            f"this release cannot tell health from disease: no {'/'.join(ctx.species)} row's disease "
            "field reads as healthy (build-markers reads the release's word Normal, in any case and "
            f"trimmed, as a healthy sample); the field holds {_words(db, ctx.species, None) or 'no value'}"
            ". The lineage tests count healthy studies; write Normal for the healthy samples, or "
            f"{off_by}", code="tree_no_healthy_rows")
    tested = not cannot_tell
    if tested and HT < min_studies:
        raise CannotTell(
            f"this context holds {HT} healthy stud{'y' if HT == 1 else 'ies'} (and {DT} disease "
            f"stud{'y' if DT == 1 else 'ies'}) in this corpus, fewer than the {min_studies} a lineage "
            "needs (min_studies), so the corpus cannot tell this tissue's own lineages from its "
            "healthy samples (its disease field holds "
            f"{_words(db, ctx.species, None if every_tissue else list(ctx.tissues)) or 'no value'}); "
            + off_by)
    if tested and min_share > 0 and HT < 1 / min_share:
        cannot_tell.append(f"this context holds only {HT} healthy studies: one study is "
                           f"{100 / HT:.1f}% of them, more than min_share={min_share:g}, so the share "
                           "test cannot tell a rare lineage from a common one; read each lineage's "
                           "numbers")

    def evidence(d):
        return {"studies": len(d["a"]), "healthy_studies": len(d["h"]),
                "healthy_only_studies": len(d["h"] & pure),
                "disease_studies": len(d["d"])} if d else \
               {"studies": 0, "healthy_studies": 0, "healthy_only_studies": 0, "disease_studies": 0}

    def disease_r(h, dd):
        if not h:
            return None if not dd else math.inf
        return _ratio(dd, DT) / (h / HT)

    def sick(d):
        """THE DISEASE TEST: its reason, or None. A lineage (or a name) is the disease's when fewer
        than min_studies of the healthy studies reporting it here are healthy-only - every other
        healthy report of it comes from a study that also sampled the disease here. It never asks
        how often disease studies report it, so a disease that expands a tissue's own cells cannot
        remove them while min_studies healthy-only studies report them."""
        d = d or {"h": set(), "d": set(), "a": set()}
        h, hp = len(d["h"]), len(d["h"] & pure)
        if hp >= min_studies:
            return None
        return (f"disease: {hp} of the {h} healthy stud{'y' if h == 1 else 'ies'} reporting it here "
                f"sampled no disease in this context, fewer than min_studies={min_studies}: its other "
                "healthy reports come from studies that also sampled disease here")

    # THE LINEAGE TESTS, on every report the context files under each lineage (all names).
    claims, hclaims = {}, {}
    for cls, name, _n, c, h in rows:
        k = key(cls, name)
        claims[k] = claims.get(k, 0) + c
        hclaims[k] = hclaims.get(k, 0) + (h or 0)
    stats, aside = {}, {}
    for k in sorted(set(claims) | set(lin), key=str):
        ev = evidence(lin.get(k))
        h, dd = ev["healthy_studies"], ev["disease_studies"]
        share = _ratio(h, HT)
        e = None if every_tissue else (
            share / (len(wide_h[k]) / W) if W and wide_h.get(k) and HT else 0.0)
        r = disease_r(h, dd)
        stats[k] = {"lineage": k, **ev, "claims": claims.get(k, 0),
                    "healthy_claims": hclaims.get(k, 0), "share": round(share, 4),
                    "enrichment": None if e is None else round(e, 3),
                    "disease_ratio": None if r is None or math.isinf(r) else round(r, 3),
                    "single_study": (h if tested else ev["studies"]) == 1}
        if not tested:
            continue
        why = []
        if h < min_studies:
            why.append(f"evidence: {h} healthy stud{'y' if h == 1 else 'ies'} report{'s' if h == 1 else ''} "
                       f"it here, fewer than min_studies={min_studies} ({dd} disease "
                       f"stud{'y' if dd == 1 else 'ies'})")
        if share < min_share:
            why.append(f"rare: {h} of the context's {HT} healthy studies ({100 * share:.1f}%) "
                       f"report it, below min_share={min_share:g}")
        if e is not None and e < min_enrichment:
            why.append(f"shared: enrichment {e:.2f}, below min_enrichment={min_enrichment:g} - the "
                       "context's healthy studies report it less often than healthy studies across "
                       "the species' tissues do")
        sw = sick(lin.get(k)) if h >= min_studies else None   # else the evidence test decides
        if sw:
            why.append(sw)
        if why:
            aside[k] = "; ".join(why)

    def name_ev(name):
        return evidence(nam.get(name))

    # THE NAMES. One home per name: the kept lineage where it has the most markers (then the
    # lineage's name, for a stable tie). The rest are recorded, with the most-markered filing.
    best, unplaced, held = {}, {}, set()

    def miss(name, k, n, why):
        cur = unplaced.get(name)
        if cur is None or (n, str(k)) > (cur["markers"], str(cur["lineage"])):
            ev = name_ev(name)
            r = disease_r(ev["healthy_studies"], ev["disease_studies"]) if tested else None
            unplaced[name] = {"name": name, "lineage": k, "markers": n, **ev,
                              "disease_ratio": None if r is None or math.isinf(r) else round(r, 3),
                              "reason": why}

    for cls, name, n, _c, _h in rows:
        if n < min_markers or not str(name).strip() or str(name).lower() == "root":
            continue
        k = key(cls, name)
        held.add(k)
        ev = name_ev(name)
        h, dd = ev["healthy_studies"], ev["disease_studies"]
        if k in aside:
            miss(name, k, n, f"its lineage is set aside ({aside[k]})")
        elif tested and h < min_studies:
            miss(name, k, n, f"evidence: {h} healthy stud{'y' if h == 1 else 'ies'} report"
                             f"{'s' if h == 1 else ''} it here, fewer than min_studies={min_studies} "
                             f"({dd} disease stud{'y' if dd == 1 else 'ies'})")
        elif tested and sick(nam.get(name)):
            miss(name, k, n, sick(nam.get(name)))
        else:
            cur = best.get(name)
            if cur is None or (n, str(k)) > (cur[1], str(cur[0])):
                best[name] = (k, n)
    for name in best:
        unplaced.pop(name, None)
    if len(best) < 2:
        failed = (f"; {len(aside)} lineage(s) the context mentions failed the lineage tests, the "
                  "reasons per lineage: " + "; ".join(f"{k} ({aside[k]})" for k in sorted(
                      aside, key=lambda k: -stats[k]["studies"])[:5]) if aside else "")
        said = (f"only {len(best)} cell name(s) in this context have {min_markers} or more markers "
                f"at tier <= {min_tier} under lineages that pass the tests{failed}; a tree needs two "
                "at the root")
        if tested and cannot_tell:      # a thin context: say first that the corpus cannot tell
            raise CannotTell(f"{cannot_tell[0]} (and {DT} disease studies). With so few, {said}; "
                             "`--lineage-tests off` drafts every lineage it mentions, unchecked")
        raise ValueError(f"this context holds {HT} healthy and {DT} disease studies; {said}"
                         if tested else said)

    def hs(name):
        return name_ev(name)["healthy_studies"] if tested else name_ev(name)["studies"]

    by_class = {}
    for name, (k, n) in best.items():
        by_class.setdefault(k, []).append((hs(name), n, name))
    lin_rank = (lambda k: stats[k]["healthy_studies"]) if tested else (lambda k: stats[k]["studies"])
    ranked = sorted(by_class.items(), key=lambda kv: (-lin_rank(kv[0]), -sum(n for _s, n, _ in kv[1]),
                                                      str(kv[0])))
    # THE CAP never hides a verdict: a lineage that passed every test and is cut by max_lineages is
    # listed as capped, with its numbers and rank, never filed with the lineages that failed one
    capped = []
    for i, (k, members) in enumerate(ranked[max_lineages:], start=max_lineages + 1):
        capped.append(dict(stats[k], rank=i, reason=(
            f"{'passed every lineage test' if tested else 'drafted unchecked'}; ranked {i} by "
            f"{'healthy studies' if tested else 'studies'}, beyond max_lineages={max_lineages}: "
            "raise --max-lineages to draft it")))
        for _s, n, name in members:
            miss(name, k, n, f"its lineage is beyond max_lineages={max_lineages}")
    ranked = ranked[:max_lineages]
    lineages = [c for c, _ in ranked]
    taken = {c.lower() for c in lineages}

    children, patterns = {"root": []}, {}
    for cls, members in ranked:
        ordered = [nm for _s, _n, nm in sorted(members, key=lambda t: (-t[0], -t[1], t[2]))]
        names = ordered[:max_types]
        for nm in ordered[max_types:]:
            if nm.lower() not in taken:
                miss(nm, cls, best[nm][1], f"beyond max_types={max_types} under {cls}")
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
    # a lineage that passed every test but holds no placed name cannot be drafted (annotate cannot
    # score a node without markers); it is listed, so the verdict is read whole
    no_panel = [dict(stats[k], reason="passed every lineage test; " + (
        "its names that clear the marker floor are placed under another lineage or fail the name "
        "tests" if k in held else f"no name filed under it has {min_markers} or more distinct "
        f"markers at tier <= {min_tier}")) for k in sorted(
            stats, key=lambda k: (-stats[k]["healthy_studies"], str(k)))
        if tested and k not in aside and k not in by_class and str(k).strip()]
    shape = {**SHAPE, **DEFAULTS, "lineage_tests": "on"}
    not_default = {k: {"value": parameters[k], "default": v} for k, v in shape.items()
                   if parameters[k] != v}
    # every lineage set aside that held a name clearing the floor, even when each of its names is
    # placed under a kept lineage or recorded under another filing
    set_aside = [dict(stats[k], reason=aside[k]) for k in sorted(
        aside, key=lambda k: (-stats[k]["studies"], str(k))) if k in held]
    not_placed = sorted((r for nm, r in unplaced.items() if nm not in placed_somewhere),
                        key=lambda r: (str(r["lineage"]), -r["markers"], str(r["name"])))
    off = "not applied: " + (cannot_tell[0] if not tested else "")
    tests_applied = {
        "evidence": "applied" if tested else off, "share": "applied" if tested else off,
        "specificity": off if not tested else (
            "not applied: the context reads every tissue" if every_tissue else "applied"),
        "disease": "applied" if tested else off}
    context = {"healthy_studies": HT, "disease_studies": DT,
               "species_healthy_reports": W, "studies_counted_from": (
                   "the rows the studies reported (curated sources), any marker tier"
                   if reports is not None else "none: the corpus has no per-study rows")}
    summary = {"lineages": len(children["root"]),
               "types": sum(len(v) for k, v in children.items() if k != "root"),
               "lineage_from": "cell_name_class" if has_class else "none: a flat draft, root -> names",
               "kept": [stats[k] for k in lineages],
               "set_aside": set_aside, "not_placed": len(not_placed), "capped": capped,
               "no_panel": no_panel, "not_default": not_default,
               "lineage_tests": tests_applied, "cannot_tell": cannot_tell,
               "parameters": parameters, "context": context}
    tree = {
        "children": children, "patterns": patterns, "members": {},
        "proposed": {
            "by": "scanno tree propose", "version": version, "rule": RULE,
            "corpus": str(db), "corpus_sha256": sha,
            "species": list(ctx.species),
            "tissues": list(ctx.tissues) if ctx.tissues is not None else "every tissue",
            "parameters": parameters, "parameters_mean": PARAMETERS,
            "min_tier": min_tier, "min_markers": min_markers, "max_lineages": max_lineages,
            "max_types": max_types, "lineage_from": summary["lineage_from"],
            "lineage_tests": tests_applied, "cannot_tell": cannot_tell, "context": context,
            "not_default": not_default,
            "kept": summary["kept"], "capped": capped, "no_panel": no_panel,
            "set_aside": set_aside,
            "not_placed": not_placed,
            "approved_by": None,
            "note": ("A DRAFT for you to approve or edit. Set approved_by to your name once you "
                     "have read it; until then every annotate run that uses it prints REVIEW and "
                     "records 'tree: proposed draft' in STATUS. 'kept', 'set_aside' and "
                     "'not_placed' carry the studies and claims that decided each entry; "
                     "'cannot_tell' says where the corpus could not decide. Add back what your "
                     "sample holds; remove what it cannot."),
        },
    }
    return tree, summary


#: the tests a draft's `proposed.lineage_tests` records, and the parameters that set them
TESTS = ("evidence", "share", "specificity", "disease")


def provenance(tree):
    """What STATUS records for a tree: None for a tree nobody proposed, else whether it was
    approved and by whom, and which lineage tests were NOT applied when it was drafted (all of them
    for `--lineage-tests off`, or for a draft that records none: one drafted before the tests
    existed) and which test parameters were not the defaults - so a reader of the labels can tell
    an unchecked draft from a checked one. Read by `annotate`."""
    prop = tree.get("proposed") if isinstance(tree, dict) else None
    if not isinstance(prop, dict):
        return None
    who = prop.get("approved_by")
    tests = prop.get("lineage_tests")
    par = prop.get("parameters") if isinstance(prop.get("parameters"), dict) else {}
    if isinstance(tests, dict) and tests and par.get("lineage_tests", "on") != "off":
        unchecked = [t for t in TESTS if not str(tests.get(t, "")).startswith("applied")]
    else:       # no record of the tests, or the draft's parameters say they were off: the same fact
        unchecked = list(TESTS)
    nd = prop.get("not_default") if isinstance(prop.get("not_default"), dict) else {}
    lifted = {k: v for k, v in nd.items() if k in DEFAULTS or k == "lineage_tests"}
    return {"source": "proposed draft"
            + (f", unchecked: lineage tests not applied ({', '.join(unchecked)})" if unchecked else "")
            + (f", approved by {who}" if who else ""),
            "approved_by": who or None, "rule": prop.get("rule"),
            "corpus_sha256": prop.get("corpus_sha256"),
            "lineage_tests": tests if isinstance(tests, dict) and tests else
            "not recorded: the draft records no lineage tests",
            "unchecked": unchecked, "not_default": lifted}
