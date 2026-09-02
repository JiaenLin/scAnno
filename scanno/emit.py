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
from .force import BY_GAP, FROM_WALK

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
        # HOW the label was arrived at, not only what it is. Defaulted to `gap` because a row
        # that never passed through `scanno.force` was assigned by the walk and by nothing else.
        # The value is a statement about the row, so it is present for every row and absent for
        # none - a column with gaps in it would be read as "unknown" where it means "ordinary".
        "assignment": np.array([str(by_cluster[int(c)].get("assignment", BY_GAP)) for c in y],
                               dtype=object),
        # HOW FAR it was pushed, not merely that it was. Each forced step is a decision the walk
        # did not take, so a twice-pushed label is the end of a chain and must not read as the
        # equal of a once-pushed one. It counts decisions, not rejections - only the first is
        # below the bar by construction - so the margins in `uns` are what say how weak they
        # were. Zero means no step was forced: true of a gap-cleared row and of a withheld one
        # alike, which is what `assignment` is there to tell apart.
        "force_depth": np.array([int(by_cluster[int(c)].get("force_depth", 0)) for c in y],
                                dtype=np.int16),
        # THE RESOLVED LABEL: every walked cell on a leaf, in a column of its own.
        #
        # Defaulted to the walk's OWN answer, so a run that never called `resolve_to_leaf` gets a
        # resolved column identical to the label column rather than an empty one. `origin` is
        # what tells the two apart, and it is why these three are never written separately: a
        # column mixing reached leaves with assigned ones, and not saying which is which, is
        # worse than no column, because every consumer reads the assignments as calls.
        "resolved": np.array([by_cluster[int(c)].get("resolved_label",
                                                     by_cluster[int(c)]["label"]) for c in y],
                             dtype=object),
        "resolved_path": np.array([by_cluster[int(c)].get("resolved_path",
                                                          by_cluster[int(c)]["path"]) for c in y],
                                  dtype=object),
        # FROM_WALK, not BY_GAP: `assignment` and `resolved_origin` are different vocabularies
        # and defaulting one to the other's value would put "gap" in a column whose declared
        # values never include it.
        "resolved_origin": np.array([str(by_cluster[int(c)].get("resolved_origin", FROM_WALK))
                                     for c in y], dtype=object),
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
        # The exclusion is per NUCLEUS. A flagged nucleus sitting in a cluster that WAS forced
        # took no part in that decision, and `forced` here would claim it did.
        out["assignment"][flag] = EXCLUDED
        out["force_depth"][flag] = 0
        # Withheld before the walk, so there is no trace to descend and nothing to resolve.
        # Forcing a leaf onto a nucleus that was never annotated would be inventing the one
        # thing this column exists to make visible.
        out["resolved"][flag] = EXCLUDED
        out["resolved_path"][flag] = EXCLUDED
        out["resolved_origin"][flag] = EXCLUDED
    return out


def support_per_cell(cell_type, support):
    """Curated tier<=2 assertions behind each cell's label. NaN where unknown.

    Unknown covers three different things - no corpus was consulted, the label is a sentinel
    (`EXCLUDED`, `UNRESOLVED`), or the corpus has no entry for that node - and none of them is
    zero. Zero would mean "counted, and there were none".
    """
    return np.array([float(support.get(str(v), np.nan)) for v in cell_type], dtype=np.float32)


def annotate_obs(adata, res, y, flag=None, prefix=DEFAULT_PREFIX, support=None, suffix="",
                 assignment=False, resolved=False):
    """Write the per-cell annotation into `adata.obs`. Returns the column names written.

    `suffix` goes on the END of each column name, which is where a sweep needs it:
    `scanno resolution` reads a family of columns sharing a prefix and differing by the
    resolution, so annotating the same object at eight resolutions wants
    `scanno_path_r0p25 … scanno_path_r2p0` rather than the resolution buried in the middle.

    `assignment=True` adds TWO columns, which are one statement and are never written apart:
    `<prefix>_assignment<suffix>` — `gap` / `forced` / `EXCLUDED` per cell — and
    `<prefix>_force_depth<suffix>`, the number of forced steps behind that cell's label. They are
    OPT-IN rather than always written because a run with no scope has nothing to distinguish:
    every row would read `gap` and 0, and a column that can only hold one value is a column a
    reader has to check before learning nothing. `scanno annotate --scope` turns them on, and
    then they are written even when zero cells were forced — there, all-`gap` is a measurement.
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
    if assignment:
        put("assignment", cols["assignment"], categorical=True)
        # Written with it and never without it: the pair is one statement. `assignment` says a
        # cell was forced, `force_depth` says through how many steps outside the walk, and a
        # reader given only the first has no way to tell one decision from two.
        put("force_depth", cols["force_depth"])
    if resolved:
        # THREE columns, one statement, never written apart. `resolved` is a leaf for every
        # walked cell; `resolved_path` is the whole lineage so it can be truncated to any depth;
        # `resolved_origin` says whether that leaf was REACHED by the walk or ASSIGNED to it.
        # Without the third a reader cannot tell a confident call from a root-level guess, and
        # the guesses are exactly the cells this column adds.
        put("resolved", cols["resolved"], categorical=True)
        put("resolved_path", cols["resolved_path"], categorical=True)
        put("resolved_origin", cols["resolved_origin"], categorical=True)
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


def _uns_safe(v):
    """Coerce a record into something `anndata` can actually write to `uns`.

    It writes a dict as an HDF5 group and a list of scalars as an array, but a LIST OF DICTS has
    no representation and fails at write time with `Can't implicitly convert non-string objects
    to strings` - naming the key, not the shape, so the cause is not obvious. `None` has no
    representation either.

    Nor can an HDF5 group NAME contain a forward slash, and every label in this package is a
    path, so a dict keyed by a label fails too - which is the shape half this module produces.

    The list becomes a dict keyed by zero-padded position, so nothing is lost and a reader
    iterating `sorted(d)` gets the original order back; a slash-bearing key moves into its own
    value. Both were found by WRITING the object. The suite was green through both because it
    built one in memory and never wrote it, and the second survived the fix for the first.
    """
    if isinstance(v, dict):
        out = {str(k): _uns_safe(x) for k, x in v.items()}
        if any("/" in k for k in out):
            # HDF5 group names cannot contain "/" - it is the path separator - and a label in
            # this package IS a path, so any dict keyed by a label is unwritable by
            # construction. The key moves into the value, where a slash is ordinary text.
            enc = {f"{i:04d}": {"key": k, "value": out[k]} for i, k in enumerate(sorted(out))}
            enc["_encoding"] = ("each entry is {key, value}: HDF5 group names cannot contain "
                                "a forward slash and a label here is a path")
            return enc
        return out
    if isinstance(v, (list, tuple)):
        if v and all(isinstance(x, dict) for x in v):
            return {f"{i:04d}": _uns_safe(x) for i, x in enumerate(v)}
        return ["" if x is None else x for x in v]
    return "" if v is None else v


def sweep_path(adata, res, y, flag=None, *, prefix=DEFAULT_PREFIX, suffix="", tag,
               resolved=False):
    """ONE column for one resolution of a sweep: the label PATH, and nothing else.

    `annotate_obs` writes five to eleven columns because the answer needs all of them - the
    label, its depth, its margin, how it was assigned. A SWEEP does not: its columns exist to be
    read together by `scanno resolution` and `resolution.consensus`, both of which read paths
    and nothing else. Writing the full set per resolution put sixty-four columns on an object
    for eight resolutions, of which eight were read.

    The name is `<prefix>_resolved_path<suffix>_r<tag>` - `_r<tag>` LAST, which is where
    `context.sweep_stem` looks for it, so the per-resolution figures find the sweep by the same
    rule that names it.

    Returns the column name written.
    """
    import pandas as pd

    cols = per_cell(res, y, flag=flag)
    name = ("resolved_path" if resolved else "path")
    key = f"{prefix}_{name}{suffix}_r{tag}"
    adata.obs[key] = pd.Categorical([str(v) for v in cols[name]])
    return key


def sweep_agreement_column(adata, agreement, record, *, prefix=DEFAULT_PREFIX, suffix=""):
    """ONE column: how much of the resolution sweep agrees with the label this run delivered.

    It describes the annotation beside it and does not replace or compete with it. An earlier
    version wrote a voted LABEL here as well; that column was per cell where an annotation is
    per cluster, and `joint.reconcile` reads "route B delivers L" off the label column on the
    assumption that those two are the same set. Reporting is the whole job.

    REFUSES to overwrite, like every other column-writing path here.
    """
    key = f"{prefix}_sweep_agreement{suffix}"
    if key in adata.obs:
        raise ValueError(f"obs[{key!r}] already exists.")
    if len(agreement) != adata.n_obs:
        raise ValueError(f"agreement is {len(agreement)} for {adata.n_obs} cells")
    adata.obs[key] = np.asarray(agreement, dtype=np.float32)
    adata.uns[f"{prefix}_sweep_provenance{suffix}"] = _uns_safe(dict(record, key=key))
    return {"key": key}


def annotate_joint(adata, labels, origin, record, *, key, origin_suffix="_origin"):
    """Write the joint-route label per CELL. The only code path that does.

    `classify()` proposes and this module assigns; the joint route is a fourth proposer and it
    goes through the same door, so there is exactly one place in the package where a label
    reaches `obs`. It REFUSES to overwrite: the joint column is a view over the forced one, and
    a tool that silently replaced an existing column would make reverting impossible.
    """
    import pandas as pd

    if key in adata.obs:
        raise ValueError(
            f"obs[{key!r}] already exists. The joint route ADDS a column beside the annotation "
            f"it corrects and never replaces one - choose another name with --out-key.")
    if len(labels) != adata.n_obs or len(origin) != adata.n_obs:
        raise ValueError(f"labels/origin are {len(labels)}/{len(origin)} for {adata.n_obs} cells")
    adata.obs[key] = pd.Categorical([str(x) for x in labels])
    adata.obs[key + origin_suffix] = pd.Categorical([str(x) for x in origin])
    adata.uns["scanno_joint_route"] = _uns_safe(
        dict(record, key=key, origin_key=key + origin_suffix))
    return {"key": key, "origin_key": key + origin_suffix,
            "n_corrected": int(record.get("n_corrected", 0)),
            "n_categories": int(adata.obs[key].nunique())}


def annotate_rescue(adata, labels, origin, record, *, key, origin_suffix="_origin"):
    """Write the rescued label per CELL. The only code path that does.

    Same door as `annotate_joint`: `classify()` proposes and this module assigns, so there is
    exactly one place in the package where a label reaches `obs`. It REFUSES to overwrite - the
    rescued column sits BESIDE the column it corrects, and a tool that replaced one would make
    reverting impossible and leave no record of which cells moved.
    """
    import pandas as pd

    if key in adata.obs:
        raise ValueError(
            f"obs[{key!r}] already exists. A rescue ADDS a column beside the annotation it "
            f"corrects and never replaces one - choose another name with --out-key.")
    if len(labels) != adata.n_obs or len(origin) != adata.n_obs:
        raise ValueError(f"labels/origin are {len(labels)}/{len(origin)} for {adata.n_obs} cells")
    adata.obs[key] = pd.Categorical([str(x) for x in labels])
    adata.obs[key + origin_suffix] = pd.Categorical([str(x) for x in origin])
    adata.uns["scanno_rescue"] = _uns_safe(dict(record, key=key,
                                                origin_key=key + origin_suffix))
    return {"key": key, "origin_key": key + origin_suffix,
            "n_rescued": int(record.get("n_renamed", 0)),
            "n_categories": int(adata.obs[key].nunique())}


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

    Object-backed is lossless - the same `str` objects, a different array behind them - and it
    takes on pandas 2. On pandas 3 it does NOT: the assignment is coerced back to `str`, measured
    `str` in and `str` out. There `classic_string_encoding` carries the guarantee, and every writer
    in this package holds it, so the file is a plain-string dataset on both stacks.

    Returns only what actually LANDED, verified after the assignment. Reporting a conversion that
    was coerced away would announce a fix that did not happen, on the one stack where it cannot.
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
            else:
                adata.var.index = new
            # ONLY REPORT WHAT ACTUALLY TOOK. On pandas >= 3 the assignment above is coerced
            # straight back to `str` - measured: `str` in, `str` out - so claiming the conversion
            # happened would announce a fix that did not occur, on the one stack where it cannot.
            # `classic_string_encoding` is what makes the file readable there; this stays because
            # it is still the right thing on older pandas, where it does take.
            landed = str((adata.obs if axis == "obs" else adata.var).index.dtype) == "object"
            changed[f"{axis}_index"] = landed
        for col in list(frame.columns):
            if str(frame[col].dtype) in ("string", "str"):
                frame[col] = np.array([None if pd.isna(v) else str(v) for v in frame[col]],
                                      dtype=object)
                if str(frame[col].dtype) == "object":
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


#: Where the values in an L1 column came from. `derived` is `level_columns` truncating the deep
#: walk's own path; `independent` is a SECOND walk, against a depth-1 tree, of its own.
L1_DERIVED = "derived"
L1_INDEPENDENT = "independent"


def independent_l1(adata, res, y, flag=None, level_prefix="scAnno_L", suffix="", sep="/",
                   sentinels=(EXCLUDED, UNRESOLVED), tree="", resolved=False):
    """Replace the DERIVED L1 column with a second, independent walk's result. Returns (col, rec).

    `res` is `classify()`'s output for the DEPTH-1 tree — the same unchanged walk, the same Z,
    the same background, the same gap. Nothing here re-decides anything: it joins per cell
    exactly as `annotate_obs` does, honours the per-nucleus flag the same way, and writes the
    values into the column `level_columns` has already made.

    WHY IT OVERWRITES `scAnno_L1<suffix>` RATHER THAN TAKING A NAME OF ITS OWN

    `scAnno_L1` is the name every consumer of this cohort already keys on, and the deliverable is
    defined as two label columns, one of them called L1. A second column named
    `scAnno_L1_independent` would leave the independent answer invisible to exactly the readers
    it was built for, and would leave the derived column — the one this run means to supersede —
    still sitting under the name they read. The suffix rule is untouched and is what keeps a
    sweep from colliding: a suffixed run writes `scAnno_L1_r1p0`, never `scAnno_L1`.

    WHY THE MARK IS IN `uns` AND NOT IN THE NAME

    "Independent" is a fact about where the column came from, not about what it holds, and
    provenance encoded in a column name can only be read by a human squinting at it.
    `uns[f"{col}_provenance"]` sorts beside the column it describes, is machine-readable, and
    survives `rewrite_for_viewer`, which drops only `_tmp`/`_scratch`/`_temp` keys.

    The cost is stated rather than hidden: `anndata.concat` drops `uns` by default, so
    concatenating a `--l1-tree` object with one annotated without it gives a single `scAnno_L1`
    holding both kinds and no record of which cell got which. A distinct column name would have
    made that visible, as missing values. So the record also carries `n_disagree` against the
    derived column it replaced — a run whose two L1s would have differed says so at write time,
    while the object is still cheap to rebuild.
    """
    import pandas as pd

    col = f"{level_prefix}1{suffix}"
    cols = per_cell(res, y, flag=flag)
    vals = [str(v) for v in cols["path"]]

    # An L1 column may not hold a path. If it does, the tree was not depth 1 and the caller's
    # guard did not fire — raise rather than truncate, because truncating here would silently
    # produce a correct-LOOKING L1 out of a tree nobody meant to pass.
    deep = sorted({v for v in vals if v not in sentinels and sep in v})
    if deep:
        raise ValueError(
            f"the L1 walk returned {len(deep)} path(s) below level 1 — "
            f"{', '.join(deep[:4])}{' ...' if len(deep) > 4 else ''}. "
            f"--l1-tree needs a DEPTH-1 tree: `scanno scope --out-l1-tree`, or "
            f"`scope.truncate_tree(tree, 1)`.")

    before = [str(v) for v in adata.obs[col]] if col in adata.obs else None
    adata.obs[col] = pd.Categorical(vals)

    # The RESOLVED L1, in a column of its own and never over the top of the honest one. The L1
    # tree is depth 1, so a cell the walk left UNRESOLVED is pushed exactly one step - onto the
    # root's argmax, which the walk already recorded and which is a leaf by construction. Same
    # rule as the deep walk: additive, EXCLUDED untouched, and the origin column says which.
    rcol = ""
    if resolved:
        rcol = f"{level_prefix}1_resolved{suffix}"
        adata.obs[rcol] = pd.Categorical([str(v) for v in cols["resolved_path"]])
        adata.obs[f"{rcol}_origin"] = pd.Categorical(
            [str(v) for v in cols["resolved_origin"]])

    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    rec = {
        "column": col,
        "source": L1_INDEPENDENT,
        "tree": str(tree),
        "n_cells": len(vals),
        "labels": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "replaced": L1_DERIVED if before is not None else "",
        # -1, not 0: there was no derived column to compare against, which is not "they agreed".
        "n_disagree": (int(sum(1 for a, b in zip(before, vals) if a != b))
                       if before is not None else -1),
        "disagreements": {},
    }
    if before is not None:
        d = {}
        for a, b in zip(before, vals):
            if a != b:
                d[f"{a} -> {b}"] = d.get(f"{a} -> {b}", 0) + 1
        rec["disagreements"] = dict(sorted(d.items(), key=lambda kv: -kv[1]))
    if rcol:
        rv = [str(v) for v in cols["resolved_path"]]
        rec["resolved_column"] = rcol
        rec["n_resolved_from_unresolved"] = int(
            sum(1 for a, b in zip(vals, rv) if a == UNRESOLVED and b != UNRESOLVED))
        moved = {}
        for a, b in zip(vals, rv):
            if a != b:
                moved[f"{a} -> {b}"] = moved.get(f"{a} -> {b}", 0) + 1
        rec["resolved_moves"] = dict(sorted(moved.items(), key=lambda kv: -kv[1]))
    adata.uns[f"{col}_provenance"] = rec
    return col, rec


def format_independent_l1(rec) -> list:
    """The independent-L1 record as lines: what it replaced, and where the two differ."""
    prov = f"{rec['column']}_provenance"
    L = [f"  {rec['column']}: an INDEPENDENT walk against {rec['tree'] or 'the depth-1 tree'} — "
         f"{len(rec['labels'])} label(s) over {rec['n_cells']:,} cells",
         f"    provenance in uns[{prov!r}], source={rec['source']!r}, so a reader can tell this "
         f"column from a derived one"]
    if rec.get("replaced"):
        L.append(f"    it REPLACED the {rec['replaced']} L1 — the deep walk's path truncated to "
                 f"level 1")
    n = rec.get("n_disagree", -1)
    if n == 0:
        L.append("    the two agree on every cell, which is what an unchanged root decision "
                 "predicts. Measured here, not assumed")
    elif n > 0:
        pct = 100 * n / max(rec["n_cells"], 1)
        L.append(f"    REVIEW  they differ on {n:,} cell(s) ({pct:.3f}%). The deeper level "
                 f"columns are still the DEEP walk's, so L1 and L2 disagree for those cells:")
        for k, v in list(rec.get("disagreements", {}).items())[:6]:
            L.append(f"      {v:>8,}  {k}")
    return L


def trace_provenance(adata, res, prefix=DEFAULT_PREFIX, suffix=""):
    """Persist WHY each cluster got its label: every node scored, what won, what came second.

    `classify()` computes this at every step of every walk and returns it, and until now it was
    thrown away - the object carried the winning label and the margin of the accepted step, and
    nothing about the alternatives. A margin of 0.64 does not say 0.64 over WHAT, so a call that
    beat a near-tie and one that beat nothing close read identically, and "why is this cluster
    labelled that" could only be answered by reconstructing it from a marker table.

    Per CLUSTER rather than per cell, because it is a property of the cluster's mean profile and
    one row per nucleus would repeat it a thousand times. Sorted by cluster id as a string so it
    survives the HDF5 round trip, which cannot key a group by an integer.
    """
    key = f"{prefix}_trace{suffix}"
    out = {}
    for r in res:
        if r.get("excluded"):
            continue
        out[str(r["cluster"])] = {
            "label": str(r.get("label", "")), "depth": int(r.get("depth", 0)),
            "steps": [{"at": str(t.get("at", "")), "top": str(t.get("top", "")),
                       "second": ("" if t.get("second") is None else str(t["second"])),
                       "gap": float(t.get("gap", float("nan"))),
                       "scores": {str(k): float(v) for k, v in (t.get("scores") or {}).items()}}
                      for t in (r.get("trace") or [])],
        }
    adata.uns[key] = _uns_safe({
        "schema": "scanno/trace@1", "column": f"{prefix}_path{suffix}",
        "n_clusters": len(out), "clusters": out,
        "limit": "scores are the cluster mean standardised against the STORE'S GENE BACKGROUND, "
                 "not against this run. They are comparable between clusters and between runs "
                 "of the same store, and they are not probabilities. A large score means the "
                 "profile is unusual for that gene set against the background - which favours a "
                 "marker that is absent elsewhere over one that is merely abundant.",
    })
    return key


def force_provenance(adata, record, prefix=DEFAULT_PREFIX, suffix="", scope=""):
    """Record HOW the forced cells were assigned, in `uns`, beside the column that says WHICH.

    Returns the key written. The pair is deliberate and mirrors `independent_l1`:

      obs[`<prefix>_assignment<suffix>`]   per CELL, one of `gap` / `forced` / `EXCLUDED` — the
                                           part a filter or a groupby needs, and the part that
                                           survives `anndata.concat`.
      obs[`<prefix>_force_depth<suffix>`]  per CELL, how many forced steps produced that label.
                                           Also irreducible: a forced push can take more than one
                                           step, and nothing else in obs implies how many.
      uns[<that column>_provenance]        per CLUSTER, the FORCE node, the leaf chosen, the path
                                           after each step and the margin of each, survival,
                                           cover and cell count — the part a sensitivity check
                                           needs, and too wide to be columns.

    Why the detail is not also columns: a per-cell copy of a margin would be a second home for a
    number `<prefix>_gap` already holds, and two homes for one number is how they come to
    disagree — and a chain of them has no per-cell scalar form at all. Why the columns are not
    only `uns`: `anndata.concat` drops `uns` by default, and a cohort concatenated without it
    must still be able to tell a forced cell from a cleared one, and a doubly-forced one from a
    singly-forced one. So the irreducible per-cell facts are columns and everything derivable
    from them is not.
    """
    rec = dict(record)
    rec["column"] = f"{prefix}_assignment{suffix}"
    rec["scope"] = str(scope)
    key = f"{rec['column']}_provenance"
    adata.uns[key] = rec
    return key


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
