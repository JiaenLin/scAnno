"""Write the annotation back into the object, per CELL, so something else can open it.

`classify()` returns one row per CLUSTER. Every consumer of an annotation wants one label per
CELL - a viewer colouring an embedding, a composition table, a pseudobulk grouping - and until
0.3.1 `scanno annotate` printed the cluster table, optionally wrote it as a TSV, and stopped.
The join back onto the object was left to the caller, so every caller wrote it again, and the
object scAnno had just annotated still carried no annotation.

That is what this module does, and the reason it is a module rather than four lines in the CLI
is that the join has three edges worth testing:

  - a cluster index that appears in `y` and not in `res` must RAISE, not silently label a
    population `UNRESOLVED`. classify() promises one row per cluster in order, and a caller
    that reindexes it "silently mislabels everything after the first gap" (its own docstring).
  - the per-cell flag OVERRIDES the cluster's call. A flagged nucleus is `EXCLUDED` even when
    the cluster around it was walked and labelled, because the exclusion is per nucleus and
    never per cluster - the distinction the whole of `scanno/exclude.py` exists to hold.
  - statistics of a call that was not made are NaN, never 0. A gap of 0.0 sorts and averages
    beside real ones; a NaN does not.

WHAT THE COLUMNS ARE FOR

`<prefix>_cell_type` is the label. It is a pandas Categorical because AnnData writes one as a
`categories` + `codes` group, which is the encoding every reader understands; a plain object
column is written as a string dataset, which is also readable but larger and less canonical.

The evidence columns travel beside it deliberately. A gap says how far a call beat its siblings
ON THIS DATA and says nothing about how much evidence stood behind the panel, so `support` and
`survival` ride along - the same reason `classify()` returns them.

They are float32 with NaN for unknown rather than a nullable pandas integer, and that is not a
taste: AnnData writes a nullable integer as an HDF5 GROUP, and a reader that expects
`categories`/`codes` in a group skips it. Checked against scrnaseq-lab's reader, which notes
"obs/<key> is a group this reader does not understand; skipped". A float that is read beats an
integer that is dropped.
"""
from __future__ import annotations

import numpy as np

from .exclude import EXCLUDED

#: The default column-name stem. `scanno_cell_type` is not arbitrary: a downstream reader has to
#: GUESS which column holds the annotation, and every convention for that guess keys on the
#: substring `cell_type`. Naming it `scanno_label` would make the tool's own output something a
#: viewer has to be told about by hand.
DEFAULT_PREFIX = "scanno"

UNRESOLVED = "UNRESOLVED"


def per_cell(res, y, flag=None):
    """Per-CLUSTER calls -> per-CELL arrays. `y` is the cluster index of each cell.

    `flag` is the per-cell boolean handed to `--exclude-flag`, or None. Where it is True the
    cell is `EXCLUDED` whatever its cluster was called, because that is what per-nucleus means.
    """
    y = np.asarray(y)
    by_cluster = {int(r["cluster"]): r for r in res}
    seen = set(int(c) for c in np.unique(y))
    missing = sorted(seen - set(by_cluster))
    if missing:
        # Not a warning. A cell whose cluster has no call would silently receive some other
        # cluster's label, or a plausible-looking UNRESOLVED that is really "we lost it".
        raise KeyError(
            f"{len(missing)} cluster(s) present in the cell assignment have no call from "
            f"classify(): {missing[:8]}{' ...' if len(missing) > 8 else ''}. "
            f"classify() returns one row per cluster in order - the caller has reindexed or "
            f"filtered `res`.")

    n = y.shape[0]
    out = {
        "cell_type": np.array([by_cluster[int(c)]["label"] for c in y], dtype=object),
        "path": np.array([by_cluster[int(c)]["path"] for c in y], dtype=object),
        "depth": np.array([by_cluster[int(c)]["depth"] for c in y], dtype=np.int16),
        "gap": np.array([by_cluster[int(c)]["gap"] for c in y], dtype=np.float32),
        "survival": np.array([by_cluster[int(c)].get("survival", np.nan) for c in y],
                             dtype=np.float32),
    }
    if flag is not None:
        flag = np.asarray(flag, dtype=bool)
        if flag.shape[0] != n:
            raise ValueError(f"flag has {flag.shape[0]} values for {n} cells")
        out["cell_type"][flag] = EXCLUDED
        out["path"][flag] = EXCLUDED
        out["depth"][flag] = 0
        out["gap"][flag] = np.nan
        out["survival"][flag] = np.nan
    return out


def support_per_cell(cell_type, support):
    """Curated tier<=2 assertions behind each cell's label. NaN where unknown.

    Unknown covers three different things - no corpus was consulted, the label is a sentinel
    (`EXCLUDED`, `UNRESOLVED`), or the corpus has no entry for that node - and none of them is
    zero. Zero would mean "counted, and there were none".
    """
    return np.array([float(support.get(str(v), np.nan)) for v in cell_type], dtype=np.float32)


def annotate_obs(adata, res, y, flag=None, prefix=DEFAULT_PREFIX, support=None, suffix=""):
    """Write the per-cell annotation into `adata.obs`. Returns the column names written.

    `suffix` goes on the END of each column name, which is where a sweep needs it:
    `scanno resolution` reads a family of columns sharing a prefix and differing by the
    resolution, so annotating the same object at eight resolutions wants
    `scanno_path_r0p25 … scanno_path_r2p0` rather than the resolution buried in the middle.
    """
    import pandas as pd

    cols = per_cell(res, y, flag=flag)
    written = []

    def put(name, values, categorical=False):
        key = f"{prefix}_{name}{suffix}"
        adata.obs[key] = (pd.Categorical([str(v) for v in values]) if categorical
                          else values)
        written.append(key)

    put("cell_type", cols["cell_type"], categorical=True)
    put("path", cols["path"], categorical=True)
    put("depth", cols["depth"])
    put("gap", cols["gap"])
    put("survival", cols["survival"])
    if support:
        s = support_per_cell(cols["cell_type"], support)
        if flag is not None:
            s[np.asarray(flag, dtype=bool)] = np.nan
        put("support", s)
    return written


def plain_string_labels(adata):
    """Re-back every label and string column with plain object arrays before writing.

    pandas gives string labels a `StringDtype` backing by default, and anndata writes that as a
    NULLABLE STRING - an HDF5 group of `values` + `mask` rather than a dataset of strings. The
    file is valid AnnData and round-trips through anndata perfectly. It is also unreadable by
    anything that expects `obs/_index` to be a dataset, which is most readers, and the failure is
    not a nice one: the index comes back undefined and the first `.map` over it throws. A viewer
    reports "that file did not open" and names a property access, which points nowhere near the
    cause.

    Older anndata refuses to write StringDtype at all, so the same object is unwritable on one
    version and unreadable-by-others on the next. Neither is a good place to leave a deliverable.

    Object-backed is lossless - the same `str` objects, a different array behind them - and is
    preferred over `anndata.settings.allow_write_nullable_strings`, which is global: flipping it
    changes how every object written anywhere in the process is stored.

    Returns what it converted, so a run can say so rather than doing it invisibly.
    """
    import numpy as np
    import pandas as pd

    changed = {"obs_index": False, "var_index": False, "obs_columns": [], "var_columns": []}

    for axis, frame in (("obs", adata.obs), ("var", adata.var)):
        idx = frame.index
        if str(idx.dtype) != "object":
            new = pd.Index(np.array([str(v) for v in idx], dtype=object), name=idx.name)
            if axis == "obs":
                adata.obs.index = new
                changed["obs_index"] = True
            else:
                adata.var.index = new
                changed["var_index"] = True
        for col in list(frame.columns):
            if str(frame[col].dtype) in ("string", "str"):
                frame[col] = np.array([None if pd.isna(v) else str(v) for v in frame[col]],
                                      dtype=object)
                changed[f"{axis}_columns"].append(str(col))
    return changed


def format_plain_labels(changed) -> list:
    """One line, and only when something was converted."""
    bits = []
    if changed["obs_index"] or changed["var_index"]:
        which = " and ".join(x for x, on in (("obs", changed["obs_index"]),
                                             ("var", changed["var_index"])) if on)
        bits.append(f"{which} index")
    for axis in ("obs", "var"):
        if changed[f"{axis}_columns"]:
            bits.append(f"{len(changed[f'{axis}_columns'])} {axis} column(s)")
    if not bits:
        return []
    return [f"  re-backed as plain strings before writing: {', '.join(bits)}",
            "    pandas' StringDtype is written as a nullable string - a group, not a dataset - "
            "and most readers cannot read it"]


def reindex_by_symbol(adata, key="gene_symbol", keep_as="gene_id"):
    """Re-index `var` by the symbol column, without ever merging two genes into one.

    An object is usually keyed by accession because SYMBOLS ARE NOT UNIQUE - this reference has
    43 symbols shared by more than one accession - and a reader that wants to look up `Myh6`
    should not have to know that. So the written object is keyed by symbol.

    THE ONE THING THIS MUST NOT DO is collapse the duplicates. Two accessions sharing a symbol
    are two genes; summing them puts one gene's counts under another's name and nothing
    downstream can tell. So every row survives: duplicated symbols are disambiguated the way
    anndata does it (`Myh6`, `Myh6-1`), the accession is preserved in `var[keep_as]` so the
    mapping stays reversible, and a gene with no symbol keeps its accession rather than being
    given a blank name.

    Returns a report: what was renamed, what was disambiguated, what kept its accession.
    """
    import numpy as np
    import pandas as pd

    if key not in adata.var:
        return {"applied": False, "reason": f"no var column {key!r}"}

    original = [str(v) for v in adata.var_names]
    sym = [("" if v is None else str(v)).strip() for v in adata.var[key]]
    # A gene with no symbol keeps its accession. A blank name is not a name, and `nan` as a row
    # label is worse than the accession it replaced.
    no_symbol = sum(1 for s in sym if not s or s.lower() in ("nan", "none", "na"))
    proposed = [s if (s and s.lower() not in ("nan", "none", "na")) else o
                for s, o in zip(sym, original)]

    if keep_as and keep_as not in adata.var:
        adata.var[keep_as] = pd.Categorical(original)

    counts = pd.Series(proposed).value_counts()
    dup_names = set(counts[counts > 1].index)
    n_dup_rows = int(sum(1 for p in proposed if p in dup_names))

    adata.var_names = pd.Index(proposed, dtype=object)
    if dup_names:
        adata.var_names_make_unique()          # Myh6, Myh6-1 - both rows kept
    n_changed = sum(1 for a, b in zip(original, [str(v) for v in adata.var_names]) if a != b)
    return {
        "applied": True, "key": key, "kept_as": keep_as,
        "n_genes": len(original), "n_renamed": int(n_changed),
        "n_without_symbol": int(no_symbol),
        "n_duplicate_symbols": len(dup_names), "n_rows_sharing_a_symbol": n_dup_rows,
        "unique": bool(pd.Index([str(v) for v in adata.var_names]).is_unique),
    }


def format_reindex(rep) -> list:
    """The re-indexing as lines, saying what happened to the awkward cases rather than hiding it."""
    if not rep.get("applied"):
        return [f"  var_names unchanged: {rep.get('reason', 'not applied')}"]
    L = [f"  var re-indexed by {rep['key']!r}: {rep['n_renamed']:,} of {rep['n_genes']:,} rows "
         f"renamed, accessions kept in var[{rep['kept_as']!r}]"]
    if rep["n_duplicate_symbols"]:
        L.append(f"    {rep['n_duplicate_symbols']:,} symbols are shared by more than one "
                 f"accession ({rep['n_rows_sharing_a_symbol']:,} rows). EVERY row is kept and "
                 f"the names disambiguated")
        L.append("    - merging them would sum two genes under one name, which nothing "
                 "downstream could detect")
    if rep["n_without_symbol"]:
        L.append(f"    {rep['n_without_symbol']:,} genes have no symbol and keep their "
                 f"accession")
    if not rep["unique"]:
        L.append("    WARNING var_names are still not unique")
    return L


# --------------------------------------------------------------------------- readiness
#
# These name lists are a HINT, copied from the conventions a downstream reader uses to guess
# which column is which. They are deliberately not authoritative and nothing here refuses on
# them: the reader does its own content check and has the final say, and two copies of a rule
# are two rules the moment one is edited. They exist so that `annotate` can say "the object you
# just wrote is missing the thing a viewer will ask you for" while the object is still cheap to
# rebuild - rather than the user discovering it after the conversion.
EMBED_HINTS = ("umap", "tsne", "draw_graph", "pca")
SYMBOL_COLUMNS = ("feature_name", "gene_symbol", "gene_symbols", "gene_name", "gene_names",
                  "symbol", "hgnc_symbol", "mgi_symbol", "gene", "genes", "name")
SAMPLE_HINTS = ("sample", "sample_id", "donor", "patient", "subject", "animal", "mouse",
                "library", "batch", "replicate", "orig.ident")
CONDITION_HINTS = ("condition", "treatment", "group", "genotype", "status", "disease",
                   "timepoint", "time", "day", "age", "sex", "region")

#: An accession looks like one. Anything else is treated as a symbol, which is the safe
#: direction: a false "these are symbols" costs a warning nobody needed, a false "these are
#: accessions" would send someone hunting for a mapping column they do not need.
_ACCESSION_PREFIXES = ("ENS", "FBGN", "WBGENE", "ZDB-", "MGI:")


def _looks_like_accession(names, sample=500):
    hits = 0
    seen = 0
    for v in list(names)[:sample]:
        s = str(v).upper()
        seen += 1
        if s.startswith(_ACCESSION_PREFIXES):
            hits += 1
    return seen > 0 and hits / seen > 0.5


def lab_readiness(adata, label_key):
    """What a downstream viewer needs from this object. Returns [(level, message), ...].

    `level` is "ok", "warn" or "missing". Nothing here raises: scAnno's job is to annotate, and
    an object with no embedding is not a bad annotation - it is an object somebody still has to
    run UMAP on. Saying so at write time is the whole value; refusing would be scAnno deciding
    what the object is for.
    """
    out = []

    # 1. the annotation itself
    if label_key in adata.obs:
        levels = [str(v) for v in dict.fromkeys(adata.obs[label_key].astype(str))]
        k, n = len(levels), adata.n_obs
        named = [v for v in levels if not v.lstrip("-").isdigit()]
        if k < 2:
            out.append(("warn", f"{label_key} has one level ({levels[0] if levels else 'none'})"
                                f" - a viewer needs at least two to group by"))
        elif n >= 4 and k >= n * 0.9:
            out.append(("warn", f"{label_key} has {k:,} levels across {n:,} cells - that reads "
                                f"as an identifier rather than an annotation"))
        elif k > 300:
            out.append(("warn", f"{label_key} has {k:,} levels; readers commonly treat more "
                                f"than 300 as an identifier"))
        elif not named:
            out.append(("warn", f"{label_key} is entirely numeric - a viewer will prefer a "
                                f"named column if the object has one"))
        else:
            out.append(("ok", f"cell annotation: {label_key}, {k} levels"))
    else:
        out.append(("missing", f"no {label_key} column was written"))

    # 2. an embedding - required by any viewer that draws cells
    keys = list(getattr(adata, "obsm", {}) or {})
    two_d = [k for k in keys
             if getattr(adata.obsm[k], "ndim", 0) == 2 and adata.obsm[k].shape[1] >= 2]
    preferred = [k for k in two_d if any(h in k.lower() for h in EMBED_HINTS)]
    if preferred:
        out.append(("ok", f"embedding: {', '.join(preferred)}"))
    elif two_d:
        out.append(("warn", f"obsm has {', '.join(two_d)} but nothing named like a UMAP, "
                            f"t-SNE or PCA; a viewer may not recognise it"))
    else:
        out.append(("missing", "no 2-D embedding in obsm - a viewer needs one to draw cells. "
                               "scAnno does not compute embeddings; run UMAP upstream"))

    # 3. expression a viewer will accept
    X = adata.X
    try:
        head = X[:200]
        head = head.toarray() if hasattr(head, "toarray") else np.asarray(head)
        if head.size and float(np.nanmin(head)) < 0:
            out.append(("warn", ".X holds negative values, so it is scaled rather than "
                                "log-normalised. Viewers plot expression, not z-scores, and "
                                "commonly refuse it - keep counts or log-norm in .X"))
        else:
            out.append(("ok", ".X has no negative values"))
    except Exception:                                                     # noqa: BLE001
        out.append(("warn", ".X could not be sampled; check what it holds before converting"))

    # 4. gene symbols beside accession-named rows
    if _looks_like_accession(adata.var_names):
        have = [c for c in adata.var.columns if c.lower() in SYMBOL_COLUMNS]
        if have:
            out.append(("ok", f"gene symbols: var[{have[0]!r}] beside accession row names"))
        else:
            out.append(("warn", "var_names look like accessions and no symbol column is "
                                f"present. Gene sets written as symbols will not match. "
                                f"Conventional names: {', '.join(SYMBOL_COLUMNS[:4])}"))

    # 5. optional groupings - reported, never required
    obs_lower = {c.lower(): c for c in adata.obs.columns}
    for role, hints in (("sample", SAMPLE_HINTS), ("condition", CONDITION_HINTS)):
        hit = next((obs_lower[h] for h in hints if h in obs_lower), None)
        out.append(("ok", f"{role} column: {hit}") if hit else
                   ("warn", f"no obvious {role} column - optional, but without one a viewer "
                            f"cannot group by it"))
    return out


def format_readiness(checks):
    """The readiness report as lines. `missing` first, because that is what blocks a viewer."""
    mark = {"ok": "  ok     ", "warn": "  REVIEW ", "missing": "  MISSING"}
    order = {"missing": 0, "warn": 1, "ok": 2}
    return [f"{mark[lvl]} {msg}" for lvl, msg in sorted(checks, key=lambda c: order[c[0]])]
