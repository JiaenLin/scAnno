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

    # THE LABEL COLUMN IS ONLY CALLED `cell_type` WHEN IT IS THE ANSWER.
    #
    # A viewer has to GUESS which column holds the annotation, and every convention for that
    # guess keys on the substring `cell_type`. A sweep writes one label column per resolution, so
    # naming them all `cell_type` gives the guesser eight equally good candidates and it picks by
    # tie-break - on a real cohort it chose the finest resolution, not the one the study chose.
    # Suffixed runs are a sweep; the unsuffixed run is the answer, and only it gets the name a
    # reader looks for.
    put("cell_type" if not suffix else "label", cols["cell_type"], categorical=True)
    put("path", cols["path"], categorical=True)
    put("depth", cols["depth"])
    put("gap", cols["gap"])
    put("survival", cols["survival"])
    if support:
        s = support_per_cell(cols["cell_type"], support)
        if flag is not None:
            s[np.asarray(flag, dtype=bool)] = np.nan
        put("support", s)

    # THE LEVEL COLUMNS ARE WRITTEN HERE, NOT ONLY IN THE VIEWER REWRITE.
    #
    # `scAnno_L1` is the only label that is comparable across samples BY CONSTRUCTION. Every
    # other column depends on where that sample's walk happened to stop: two animals of the same
    # arm, same batch and same chemistry have been measured returning 11.25% `Fibroblast` / 0.00%
    # `Matrifibrocyte` and 0.00% / 17.71% on the same cells, because the gap landed either side of
    # the bar. A cross-sample claim needs a column that cannot do that, and L1 is it.
    #
    # Until 0.8.2 these existed only inside `rewrite_for_viewer`, so the objects every downstream
    # stage actually reads carried the path and no levels, and each consumer re-derived the
    # truncation itself. `level_columns` is CALLED rather than copied for exactly that reason -
    # this project has already been bitten by a second copy of a rule drifting from the first.
    made, _ = level_columns(adata, f"{prefix}_path{suffix}", suffix=suffix)
    written.extend(made)
    return written


class classic_string_encoding:
    """Write string columns as HDF5 string DATASETS, not as nullable-string groups.

    On pandas >= 3 there is no way to hold a string column as `object` - the new `str` dtype is
    the default and pandas coerces back to it - and anndata >= 0.11 writes that dtype as a
    NULLABLE STRING: a group of `values` + `mask`. The result is valid AnnData that round-trips
    through anndata perfectly and is unreadable by anything else, because `obs/_index` is no
    longer a dataset. A viewer reports "cannot read properties of undefined (reading 'map')",
    which points nowhere near the cause.

    Re-backing the labels as object arrays does NOT fix it on that stack; it is the writer, not
    the dtype. `allow_write_nullable_strings = False` is the only route, and it is the
    COMPATIBLE direction: the classic encoding is what every older reader expects, so the file
    is readable by more things, not fewer.

    Scoped rather than set once at import, because it is a global on the anndata module and this
    library has no business changing how objects written elsewhere in the caller's process are
    stored. Restored even if the write raises.
    """

    def __init__(self):
        self._prev = None
        self._had = False

    def __enter__(self):
        try:
            import anndata as ad
            self._prev = ad.settings.allow_write_nullable_strings
            ad.settings.allow_write_nullable_strings = False
            self._had = True
        except Exception:                                                 # noqa: BLE001
            self._had = False        # older anndata has no such setting and does not need one
        return self

    def __exit__(self, *exc):
        if self._had:
            try:
                import anndata as ad
                ad.settings.allow_write_nullable_strings = self._prev
            except Exception:                                             # noqa: BLE001
                pass
        return False


def write_h5ad(adata, path, **kw):
    """Write an object every reader can open: plain labels, classic string encoding."""
    from pathlib import Path

    changed = plain_string_labels(adata)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with classic_string_encoding():
        adata.write_h5ad(str(path), **kw)
    return changed


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


# ===================================================================== the viewer audit

#: `uns` keys a viewer has no use for and that a writer left behind. Scratch entries from a
#: clustering sweep are the common case: they are invisible in Python, they multiply the file's
#: metadata, and a viewer that walks `uns` shows them to the user as if they were results.
SCRATCH_UNS_PREFIXES = ("_tmp", "_scratch", "_temp")


def _index_encoding(h5, group):
    """How this group's index is stored on disk: 'dataset' (classic) or 'group' (nullable)."""
    g = h5[group]
    name = g.attrs.get("_index", "_index")
    name = name.decode() if isinstance(name, bytes) else str(name)
    if name not in g:
        return "absent", name
    obj = g[name]
    import h5py
    return ("group" if isinstance(obj, h5py.Group) else "dataset"), name


def audit_file(path):
    """Everything a browser-based viewer needs from an .h5ad ON DISK, checked on disk.

    WHY ON DISK AND NOT THROUGH ANNDATA

    The failure this exists to catch is invisible in Python. `anndata` reads a nullable-string
    index and a classic one into the same pandas object, so an object that a viewer cannot open
    is indistinguishable in a notebook from one it can. The error the user sees is
    `Cannot read properties of undefined (reading 'map')`, which names nothing.

    The cause is a default that changed underneath us: with pandas 3 a string column round-trips
    as `str` dtype, and anndata then writes it as a `nullable-string-array` GROUP rather than a
    variable-length string DATASET. Viewers written against the older layout read the group as
    undefined. Re-backing the column as `object` does not help - pandas coerces it straight back.
    The only fix is to ask anndata for the classic encoding at write time.

    Returns [(level, code, message), ...] with level in "ok" | "warn" | "missing".
    """
    import h5py
    out = []
    with h5py.File(str(path), "r") as f:
        for grp in ("obs", "var"):
            if grp not in f:
                out.append(("missing", f"{grp}-absent", f"no /{grp} group"))
                continue
            kind, name = _index_encoding(f, grp)
            if kind == "group":
                out.append(("missing", f"{grp}-index-nullable",
                            f"/{grp}/{name} is a nullable-string GROUP. Browser viewers read it "
                            f"as undefined and fail with 'Cannot read properties of undefined'. "
                            f"Rewrite with scanno lab --fix"))
            elif kind == "dataset":
                out.append(("ok", f"{grp}-index", f"/{grp}/{name} is a classic string dataset"))
            else:
                out.append(("warn", f"{grp}-index-absent", f"/{grp} has no index dataset"))
            nullable = [k for k in f[grp]
                        if isinstance(f[grp][k], h5py.Group)
                        and str(f[grp][k].attrs.get("encoding-type", "")) == "nullable-string-array"]
            if nullable:
                out.append(("warn", f"{grp}-nullable-columns",
                            f"{len(nullable)} {grp} column(s) use the nullable-string encoding: "
                            f"{', '.join(nullable[:5])}"))

        emb = [k for k in f.get("obsm", {}) or {}]
        drawable = [k for k in emb if any(h in k.lower() for h in EMBED_HINTS)]
        if drawable:
            out.append(("ok", "embedding", f"obsm has {', '.join(drawable)}"))
        elif emb:
            out.append(("warn", "embedding-unnamed",
                        f"obsm has {', '.join(emb)} but nothing named like a UMAP or t-SNE; "
                        f"a viewer may not recognise it"))
        else:
            out.append(("missing", "embedding-absent",
                        "no embedding in obsm - a viewer needs one to draw cells"))

        junk = [k for k in (f.get("uns", {}) or {})
                if str(k).startswith(SCRATCH_UNS_PREFIXES)]
        if junk:
            out.append(("warn", "uns-scratch",
                        f"{len(junk)} scratch key(s) left in uns ({', '.join(sorted(junk)[:4])}"
                        f"...). A viewer that walks uns shows them as results"))
        else:
            out.append(("ok", "uns-clean", "no scratch keys in uns"))

        vcols = list(f.get("var", {}) or {})
        sym = [c for c in vcols if c.lower() in SYMBOL_COLUMNS]
        # Only a concern when the ROW NAMES are accessions. An object already indexed by symbol
        # needs no symbol column, and warning about one anyway is a gate firing on correct
        # behaviour - which is how a gate comes to be ignored on the occasion it is right.
        vkind, vname = _index_encoding(f, "var")
        names = []
        if vkind == "dataset":
            try:
                names = [n.decode() if isinstance(n, bytes) else str(n)
                         for n in f["var"][vname][:200]]
            except Exception:                                             # noqa: BLE001
                names = []
        accessions = bool(names) and _looks_like_accession(names)
        if sym:
            out.append(("ok", "gene-symbols", f"var has {sym[0]!r}"))
        elif accessions:
            out.append(("warn", "gene-symbols-absent",
                        "var_names look like accessions and no gene-symbol column is present; "
                        "gene sets written as symbols will not match. Conventional names: "
                        f"{', '.join(SYMBOL_COLUMNS[:4])}"))
        else:
            out.append(("ok", "gene-symbols", "var_names are already symbols"))
    return out


#: obs columns a viewer always wants, whatever else is dropped. Matched exactly or by prefix.
VIEWER_KEEP = ("sample", "group", "batch", "chemistry", "age", "diet", "sex", "condition",
               "donor", "patient", "timepoint", "treatment", "replicate",
               "total_counts", "n_genes", "n_genes_by_counts", "pct_counts_mt",
               "pct_counts_ribo", "doublet_score", "nn_agreement")


def level_columns(adata, path_key, prefix="scAnno_L", sep="/", sentinels=("EXCLUDED",
                                                                          "UNRESOLVED"),
                  suffix=""):
    """Write one column per level of the taxonomy: L1, L2, ... down to the deepest path.

    WHY A COLUMN PER LEVEL RATHER THAN THE PATH

    A viewer groups by a categorical. Handed `Immune/Myeloid/Macrophage` it offers one category
    per full path - dozens of them, most tiny - and there is no way to ask it for "the level-1
    picture". The truncations are what a reader actually switches between, so they are written
    out as their own columns and the full path is kept beside them.

    A path SHORTER than the level it is being truncated to is a call the walk stopped on, and it
    keeps its own value rather than being blanked: the annotator resolved it that far and no
    further, which is a partial identity, not a missing one.

    THAT RULE IS WHAT MAKES THE DEEPEST COLUMN THE BALANCED ONE. `scAnno_L3` holds `Macrophage`
    and `Pericyte` where the walk reached depth 3 and `Stromal/Fibroblast` where it stopped at 2,
    because a shallower path keeps its own value. It is the walk's own answer, not a flat level:
    padding it down would invent a subtype the walk explicitly refused to choose, and reading L2
    instead would discard the four subtypes that cleared their gap by 2.5-3.5x.

    `suffix` mirrors `annotate_obs`: a sweep annotating one object at eight resolutions needs
    `scAnno_L1_r0p25 ... scAnno_L1_r2p0` rather than eight columns fighting over one name.
    """
    import pandas as pd
    paths = [str(v) for v in adata.obs[path_key]]
    depth = max((len(p.split(sep)) for p in paths if p not in sentinels), default=1)
    made = []
    for d in range(1, depth + 1):
        col = f"{prefix}{d}{suffix}"
        adata.obs[col] = pd.Categorical(
            [p if p in sentinels else sep.join(p.split(sep)[:d]) for p in paths])
        made.append(col)
    return made, depth


def rewrite_for_viewer(src, dst, *, drop_scratch_uns=True, path_key=None,
                       level_prefix="scAnno_L", slim=False, keep=(), log=None):
    """Rewrite an object so a browser viewer can open it, and can be navigated once open.

    Three things happen, in increasing order of how much they remove:

      1. ENCODING. Every string column and both indices are written in the CLASSIC
         variable-length encoding, which is what viewers read. Nothing is removed.
      2. SCRATCH. `uns` keys a sweep left behind are dropped. They are not data, they are
         working state, and a viewer that walks `uns` shows them to the user as results.
      3. SLIM (only with `slim=True`). The per-resolution sweep columns are dropped. An object
         swept over eight resolutions carries eight of everything and a viewer's column list
         becomes unusable - but this IS a removal, so every dropped column is NAMED, the
         chosen-resolution columns are kept, and nothing matching a design, QC or identity name
         is touched. It is reversible in the strongest sense: the source object is not modified,
         so the sweep is still on disk in full.

    Returns (adata, report) where `report` names exactly what changed.
    """
    import anndata as ad
    import pandas as pd
    log = log or (lambda *_a, **_k: None)
    A = ad.read_h5ad(str(src))
    rep = {"uns_dropped": [], "levels": [], "obs_dropped": [], "obs_before": len(A.obs.columns)}

    if drop_scratch_uns:
        for k in [k for k in list(A.uns) if str(k).startswith(SCRATCH_UNS_PREFIXES)]:
            rep["uns_dropped"].append(k)
            del A.uns[k]

    if path_key and path_key in A.obs:
        made, depth = level_columns(A, path_key, prefix=level_prefix)
        rep["levels"], rep["depth"] = made, depth

    if slim:
        keepset = set(keep) | set(VIEWER_KEEP) | set(rep["levels"])
        if path_key:
            keepset.add(path_key)
            # the statistics OF the chosen resolution, whose suffix the path key carries
            stem, _, tag = str(path_key).partition("_path")
            for stat in ("gap", "support", "survival", "depth", "cluster", "L2"):
                for cand in (f"{stem}_{stat}{tag}", f"{stem}_{stat}"):
                    if cand in A.obs:
                        keepset.add(cand)
            for cand in (f"leiden{tag}", f"leiden_r{tag.lstrip('_r')}", "leiden"):
                if cand in A.obs:
                    keepset.add(cand)
        for c in list(A.obs.columns):
            if c in keepset:
                continue
            low = str(c).lower()
            # Never drop something that looks like design, identity or QC, however it is spelled.
            if any(h in low for h in ("sample", "group", "batch", "donor", "condition",
                                      "counts", "genes", "pct_", "score", "flag", "cell")):
                continue
            rep["obs_dropped"].append(c)
            del A.obs[c]
    rep["obs_after"] = len(A.obs.columns)

    plain_string_labels(A)
    with classic_string_encoding():
        A.write_h5ad(str(dst))
    return A, rep
