"""Which library each object holds: from what the object and the manifest record, never a file's name.

A file's name is the caller's choice. Through a pipeline it also carries the stage that wrote it
(`<library>.ladder.h5ad`, then `<library>.ladder.rescued.h5ad`), so a library column filled from
the stem holds the library's name plus that suffix, and a later stage keyed on the cohort's own
names finds no library at all - with exit 0 everywhere before it.

THE SOURCES, IN ORDER, PER FILE (`key` is the library column the caller names):
  1. the object's own column `key`. The only record that is per cell and travels with the cells,
     so it is taken cell for cell and nothing is filled. A manifest that also names the file must
     agree with it: the column constant in the file, and the same name.
  2. for an object without `key`: the manifest's row for the file, and the name the upstream stage
     recorded in the object (scQC's declaration, `uns['scqc']['sample']`). One is used; two must
     agree. The manifest is named as the source when it holds the row.
  3. neither: refused, by file, saying what to pass.
Two candidates that disagree are refused by name (no silent pick), and a name the file format
cannot store is refused here, before a matrix is read, not at the write after the embedding.

Pure: no AnnData. The caller reads each object's obs and uns and hands over what they hold.
"""
from __future__ import annotations

import csv
from pathlib import Path

#: Where an object without the library column records its name upstream (scanno/upstream.py).
UPSTREAM_SOURCE = "uns['scqc']['sample']"
MANIFEST_SOURCE = "manifest"


class NameRefusal(Exception):
    """A library's name cannot be told. `reason` names every file it concerns; `fix` what to pass."""

    def __init__(self, reason: str, fix: str = "", code: str = "library_name"):
        super().__init__(reason)
        self.reason, self.fix, self.code = reason, fix, code


def column_problem(name) -> str | None:
    """Why `name` cannot be an obs column of a written .h5ad, or None.

    Measured on anndata 0.13.2: a `/` ("Forward slashes are not allowed in keys"), `_index` ("a
    reserved name for dataframe columns") and an empty name each fail at the write."""
    s = str(name)
    if not s.strip():
        return "is empty"
    if "/" in s:
        return "holds '/', which an .h5ad key cannot hold"
    if s == "_index":
        return "is '_index', which an .h5ad reserves for a frame's index"
    return value_problem(s)


def value_problem(name) -> str | None:
    """Why `name` cannot be stored as a library's name in a written .h5ad, or None.

    Measured on anndata 0.13.2: an embedded NUL ("VLEN strings do not support embedded NULLs") and
    a character UTF-8 cannot encode each fail at the write. An empty name is no name."""
    s = str(name)
    if not s.strip():
        return "is empty"
    if "\0" in s:
        return "holds an embedded NUL, which an .h5ad string cannot hold"
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return "holds a character UTF-8 cannot encode, which an .h5ad string cannot hold"
    return None


def read_manifest(path, key) -> dict:
    """{resolved file path: library name} from a TSV or CSV with a `path` column and a `key` column.

    Comment lines (`#`) and blank lines are skipped; the delimiter is a tab when the header holds
    more tabs than commas. A relative path is read from the manifest's folder."""
    p = Path(path)
    fix = (f"a TSV or CSV whose header holds `path` and {key!r} (the --sample-key), one row per "
           f"object")
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise NameRefusal(f"cannot read the manifest {p}: {e}", fix) from None
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise NameRefusal(f"the manifest {p} holds no header", fix)
    delim = "\t" if lines[0].count("\t") > lines[0].count(",") else ","
    rows = list(csv.reader(lines, delimiter=delim))
    hdr = [c.strip() for c in rows[0]]
    for col in ("path", str(key)):
        if hdr.count(col) != 1:
            raise NameRefusal(f"the manifest {p} has {hdr.count(col)} columns named {col!r} (its "
                              f"header: {', '.join(map(repr, hdr))})", fix)
    ip, ik = hdr.index("path"), hdr.index(str(key))
    out, twice = {}, []
    for n, r in enumerate(rows[1:], start=2):
        if len(r) != len(hdr):
            raise NameRefusal(f"the manifest {p}: data row {n - 1} has {len(r)} fields and the "
                              f"header {len(hdr)}", fix)
        fp, name = r[ip].strip(), r[ik].strip()
        if not fp:
            raise NameRefusal(f"the manifest {p}: data row {n - 1} names no path", fix)
        full = str((Path(fp) if Path(fp).is_absolute() else p.parent / fp).resolve())
        if full in out and out[full] != name:
            twice.append(f"{fp} as {out[full]!r} and as {name!r}")
        out.setdefault(full, name)
    if twice:
        raise NameRefusal(f"the manifest {p} names one file twice with two names: "
                          + "; ".join(twice), "one row per object")
    return out


def resolve(entries, *, key, manifest=None) -> list:
    """Each file's library, by the order above. Raises NameRefusal naming every file it cannot tell.

    `entries` is [{"file": path, "values": [str, ...] or None, "recorded": str or ""}] in the input
    order: `values` the distinct values of the object's column `key` (None when it has no such
    column), `recorded` the upstream stage's recorded name. `manifest` is `read_manifest`'s map.
    Returns [{"file", "fill", "names", "source"}]: `fill` is the name to write into an object
    without the column (None when the object carries it), `names` every library the file holds."""
    key = str(key)
    out, problems = [], []
    for e in entries:
        f = str(e["file"])
        base = Path(f).name
        m = manifest.get(str(Path(f).resolve())) if manifest is not None else None
        if e.get("values") is not None:
            vals = sorted({str(v) for v in e["values"]})
            if m is not None and vals != [m]:
                problems.append(
                    f"{base}: the manifest names {m!r}, and its column {key!r} holds "
                    + (f"{vals[0]!r}" if len(vals) == 1 else f"{len(vals)} libraries"))
            out.append({"file": f, "fill": None, "names": vals, "source": f"obs[{key!r}]"})
            continue
        if manifest is not None and m is None:
            problems.append(f"{base}: no column {key!r}, and the manifest has no row for it")
            continue
        cands = [(MANIFEST_SOURCE, m)] if m is not None else []
        rec = str(e.get("recorded") or "").strip()
        if rec:
            cands.append((UPSTREAM_SOURCE, rec))
        if not cands:
            problems.append(f"{base}: no column {key!r}, no manifest row and no name recorded "
                            f"upstream ({UPSTREAM_SOURCE}); its file name is not a record")
            continue
        if len({v for _s, v in cands}) > 1:
            problems.append(f"{base}: two names disagree - "
                            + " and ".join(f"{s} says {v!r}" for s, v in cands))
            continue
        source, name = cands[0]
        why = value_problem(name)
        if why:
            problems.append(f"{base}: the name {name!r} from {source} {why}")
            continue
        out.append({"file": f, "fill": name, "names": [name], "source": source})
    carried, filled = {}, {}
    for o in out:
        for n in o["names"]:
            (carried if o["fill"] is None else filled).setdefault(n, []).append(Path(o["file"]).name)
    for n, fs in filled.items():
        if len(fs) > 1 or n in carried:
            problems.append(f"{n!r} would name the cells of {', '.join(fs + carried.get(n, []))}: "
                            f"one library's name given to two files")
    if problems:
        fix = (f"pass --sample-key naming a column every object carries, or --manifest with a "
               f"`path` column and a {key!r} column naming each object's library")
        raise NameRefusal(
            f"embed cannot tell the library of {len(problems)} input(s) from what they record:\n  "
            + "\n  ".join(problems) + f"\n  {fix}", fix)
    return out


def record(resolved, key) -> dict:
    """What the joint object keeps of where each name came from: one row per library per file."""
    rows = [(n, r["source"], Path(r["file"]).name) for r in resolved for n in r["names"]]
    return {"key": str(key), "name": [a for a, _s, _f in rows],
            "source": [s for _a, s, _f in rows], "file": [f for _a, _s, f in rows]}
