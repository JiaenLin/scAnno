"""What upstream QC declared about this object, and what scAnno does about it.

scAnno annotates. It computes no QC metric, applies no threshold, and cannot decide that a
nucleus is technical. That rule is not weakened here, and the distinction it turns on is worth
stating precisely because it is easy to get backwards:

    SNIFFING   "this object has a column called cluster_FLAG, I will exclude on it"
               - scAnno guessing what a column means. That WOULD break the rule.

    READING    "this object's uns says scQC wrote cluster_FLAG, here is what it means, here is
                a digest of exactly which nuclei carry it"
               - scQC deciding, in its own words, and scAnno obeying. The rule holds.

So detection keys on a DECLARATION and never on a column name. An object with `cluster_FLAG`
and no declaration gets nothing from this module: scAnno does not know what that column is, and
guessing is the failure mode, not the feature.

WHY ARMING IT IS THE SAFER DEFAULT, WHICH IS NOT WHAT 0.3.0 CONCLUDED

0.3.0 REMOVED a capability rather than defaulting it off, and the reasoning was that a
default-off capability is one argument away from running. That capability WIDENED a flag - it
withheld nuclei upstream QC had passed, 783 of 2,680 on the cohort it was written for. This one
narrows to exactly what QC rejected, and the digest proves it.

The asymmetry is in which error is silent. Annotating a flagged nucleus mints a label for a cell
QC refused, and that label is indistinguishable downstream from one anybody should believe.
Withholding it produces a visible `EXCLUDED`, and `--no-exclude` undoes it in full. The louder
error is the recoverable one, so it is the default.

Nothing here is silent either way: what was found, what it means, and what will be withheld is
printed before the walk, and travels into the object and the report.
"""
from __future__ import annotations

KEY = "scqc"
SCHEMA_PREFIX = "scqc/provenance@"

#: Understood schemas. A declaration from a future scQC is REPORTED and not acted on: its fields
#: may mean something else, and acting on a record you do not understand while citing its
#: authority is the thing this module exists to prevent.
SUPPORTED = ("scqc/provenance@1",)


def _plain(v):
    """h5ad round-trips numpy scalars and 0-d arrays; compare and print them as Python."""
    if hasattr(v, "item") and getattr(v, "ndim", 0) == 0:
        try:
            v = v.item()
        except Exception:                                                 # noqa: BLE001
            pass
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return v


def declaration(adata) -> dict | None:
    """The upstream declaration on this object, as plain Python, or None."""
    raw = getattr(adata, "uns", {}) or {}
    d = raw.get(KEY)
    if d is None:
        return None
    try:
        return {str(k): _plain(v) for k, v in dict(d).items()}
    except Exception:                                                     # noqa: BLE001
        return None


def verify(adata, decl) -> tuple[bool, str]:
    """Is the flag column still the one the declaration describes?

    A mismatch is not a warning. A column rewritten since scQC wrote it is not scQC's decision
    any more, and withholding nuclei on it while citing scQC's provenance would attribute
    someone else's choice to a pipeline that did not make it.
    """
    from .exclude import flag_digest

    schema = str(decl.get("schema", ""))
    if schema not in SUPPORTED:
        return False, (f"the declaration is {schema!r} and this scAnno understands "
                       f"{', '.join(SUPPORTED)}. It is reported and not acted on")
    col = str(decl.get("flag_column") or "")
    if not col:
        return True, "the declaration records no flag column, so there is nothing to withhold"
    if col not in adata.obs:
        return False, (f"the declaration names {col!r} and the object has no such column; it has "
                       f"been dropped since upstream wrote it")
    try:
        n_obs = int(decl.get("n_obs", -1))
    except Exception:                                                     # noqa: BLE001
        n_obs = -1
    if n_obs >= 0 and n_obs != int(adata.n_obs):
        return False, (f"the declaration is for {n_obs:,} observations and this object holds "
                       f"{int(adata.n_obs):,}; it has been subset since upstream wrote it, so "
                       f"the flag it describes is not the flag in hand")
    got = flag_digest(as_mask_column(adata, col))
    want = str(decl.get("flag_digest") or "")
    if want and got != want:
        return False, (f"{col!r} does not match the declaration: digest {got} against {want}. "
                       f"The column has been altered since upstream wrote it")
    return True, f"{col!r} matches the declaration"


def as_mask_column(adata, col):
    """The flag column as a plain boolean array. NA -> False, and that is a decision.

    `cluster_FLAG` is three-valued: True, False, and never-examined. Withholding the
    never-examined would be scAnno excluding nuclei upstream did not flag - the exact failure the
    per-nucleus rule exists to prevent - so unknown is treated as NOT flagged, and upstream's own
    declaration records the same coercion so the digests agree.
    """
    import numpy as np
    import pandas as pd

    s = pd.Series(adata.obs[col])
    if s.dtype != "boolean":
        s = s.astype("boolean")
    return np.asarray(s.fillna(False).to_numpy(dtype=bool))


class Decision:
    """What will be withheld, on whose authority, and how it was reached."""

    def __init__(self, column=None, mask=None, source="none", lines=(), refuse=None, decl=None):
        self.column = column
        self.mask = mask
        self.source = source            # "scqc" | "explicit" | "none"
        self.lines = list(lines)
        self.refuse = refuse            # a string when the run must stop
        self.declaration = decl

    @property
    def active(self) -> bool:
        return self.mask is not None

    @property
    def n(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else 0


def decide(adata, explicit=None, disabled=False) -> Decision:
    """Resolve the exclusion: explicit flag, upstream declaration, or nothing.

    Precedence is `--no-exclude` > `--exclude-flag` > declaration. The person at the keyboard
    outranks the file; the file outranks the default.
    """
    decl = declaration(adata)

    if disabled:
        lines = ["exclusion DISABLED by --no-exclude"]
        if decl and str(decl.get("flag_column") or ""):
            lines.append(f"    {decl.get('tool', 'upstream')} declared "
                         f"{decl.get('flag_column')!r} with {decl.get('n_flagged')} nuclei "
                         f"flagged; they WILL be annotated, and their labels are labels for "
                         f"nuclei upstream QC rejected")
        return Decision(source="none", lines=lines, decl=decl)

    if explicit:
        if explicit not in adata.obs:
            return Decision(source="none", decl=decl, refuse=(
                f"no obs column {explicit!r}. Boolean columns available: "
                f"{[c for c in adata.obs if str(adata.obs[c].dtype) in ('bool', 'boolean')]}"))
        mask = as_mask_column(adata, explicit)
        lines = [f"exclusion from --exclude-flag {explicit}: {int(mask.sum()):,} nuclei"]
        if decl:
            col = str(decl.get("flag_column") or "")
            if col and col != explicit:
                lines.append(f"    NOTE {decl.get('tool', 'upstream')} declared {col!r} on this "
                             f"object and you named {explicit!r}. Yours is used")
        return Decision(column=explicit, mask=mask, source="explicit", lines=lines, decl=decl)

    if not decl:
        return Decision(source="none", decl=None, lines=[
            "no upstream QC declaration on this object, and none is assumed.",
            "    scAnno does not infer a flag from a column name. If this object carries one,",
            "    name it with --exclude-flag COLUMN."])

    ok, why = verify(adata, decl)
    tool = str(decl.get("tool") or "upstream")
    col = str(decl.get("flag_column") or "")
    if not ok:
        return Decision(source="none", decl=decl, refuse=(
            f"this object carries a {tool} declaration and it does not check out.\n"
            f"        {why}.\n"
            f"        Acting on it would attribute a decision to {tool} that {tool} did not "
            f"make.\n"
            f"        Re-run upstream QC, or pass --exclude-flag COLUMN to take responsibility "
            f"for the column yourself, or --no-exclude to annotate everything."))
    if not col:
        return Decision(source="none", decl=decl, lines=[
            f"{tool} declaration found: it records no flag column, so nothing is withheld"])

    mask = as_mask_column(adata, col)
    lines = [
        f"{tool} declaration found -> exclusion ARMED on {col!r}",
        f"    {int(mask.sum()):,} of {int(adata.n_obs):,} nuclei "
        f"({100 * mask.sum() / max(int(adata.n_obs), 1):.2f}%) are withheld and labelled "
        f"EXCLUDED",
        f"    digest {decl.get('flag_digest')} verified against the column in this file",
        f"    run {decl.get('run_key') or 'unrecorded'}  commit {decl.get('commit') or 'unrecorded'}",
    ]
    meaning = str(decl.get("flag_meaning") or "")
    if meaning:
        lines.append(f"    what it means: {meaning}")
    lines.append("    scAnno did not choose these nuclei and cannot widen the set. "
                 "--no-exclude annotates them instead.")
    return Decision(column=col, mask=mask, source="scqc", lines=lines, decl=decl)
