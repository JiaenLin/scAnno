"""Is the declared species the data's? (single-cell-harness ADR-0027, N2)

`--species` was used for one thing: filtering the marker corpus. It was never compared with the
object. Gene names are upper-cased on both sides before matching, so a corpus of one species
scores another's genes without a word - `--species Mouse` carried onto a human cohort clears the
gene floor and labels every cluster with mouse markers. A second cohort, human where the first
was mouse, made the question concrete.

The species is in the data. Ensembl gene identifiers name it outright (ENSG human, ENSMUSG mouse,
...): DECISIVE, and a contradiction is refused. Symbol case is only a convention - mouse and rat
write Title-case, human and macaque UPPER - so a disagreement there is printed, never refused.
"""
from __future__ import annotations

import re

#: Ensembl gene-identifier prefixes, LONGEST FIRST (`ENSGALG` is chicken and starts with `ENSG`).
ENSEMBL = sorted({"ENSG": "human", "ENSMUSG": "mouse", "ENSRNOG": "rat", "ENSDARG": "zebrafish",
                  "ENSMMUG": "macaque", "ENSGALG": "chicken", "ENSSSCG": "pig",
                  "ENSBTAG": "cow"}.items(), key=lambda kv: -len(kv[0]))
CONVENTION = {"human": "upper", "macaque": "upper", "pig": "upper", "cow": "upper",
              "chicken": "upper", "mouse": "title", "rat": "title", "zebrafish": "lower"}
_ID = re.compile(r"^(ENS[A-Z]*G)\d{6,}")
DECISIVE = 0.9


def normalise(species: str) -> str:
    """The corpus's species word, as one of ENSEMBL's names where it is one ("Mouse" -> mouse,
    "Homo sapiens" -> human)."""
    s = str(species or "").strip().lower().replace("_", " ")
    return {"homo sapiens": "human", "mus musculus": "mouse", "rattus norvegicus": "rat",
            "danio rerio": "zebrafish", "macaca mulatta": "macaque"}.get(s, s)


def measure(columns: dict) -> dict:
    """{species, basis, convention} from {name: list of strings} - the var columns and the index."""
    best = None
    for name, vals in columns.items():
        hits = {}
        for v in vals:
            m = _ID.match(str(v))
            if m:
                sp = next((s for p, s in ENSEMBL if m.group(1).startswith(p)), None)
                hits[sp] = hits.get(sp, 0) + 1
        n = sum(hits.values())
        if n and (best is None or n > best[1]):
            best = (name, n, hits)
    out = {"species": None, "basis": "no Ensembl gene identifiers in the object", "convention": None}
    if best:
        name, n, hits = best
        top, k = max(hits.items(), key=lambda kv: kv[1])
        if top and k / n >= DECISIVE:
            out.update(species=top, basis=f"{k:,} of {n:,} Ensembl identifiers in {name} are {top}")
        else:
            out["basis"] = f"the identifiers in {name} are mixed ({k / n:.0%} {top})"
    case = {"upper": 0, "title": 0, "lower": 0}
    for s in columns.get("var_names") or []:
        core = re.sub(r"[^A-Za-z]", "", str(s))
        if len(core) < 3 or _ID.match(str(s)):
            continue
        if core.isupper():
            case["upper"] += 1
        elif core[0].isupper() and core[1:].islower():
            case["title"] += 1
        elif core.islower():
            case["lower"] += 1
    tot = sum(case.values())
    if tot and max(case.values()) / tot >= 0.6:
        out["convention"] = max(case, key=case.get)
    return out


def check(declared: str, var, var_names) -> tuple:
    """(refusal or None, note or None) for a declared species against an object's var."""
    cols = {"var_names": [str(v) for v in var_names]}
    for c in getattr(var, "columns", []):
        try:
            vals = [str(v) for v in var[c]]
        except Exception:                                                 # noqa: BLE001
            continue
        cols[f"var[{c!r}]"] = vals
    m = measure(cols)
    want = normalise(declared)
    if m["species"] and want in CONVENTION and m["species"] != want:
        return (f"--species {declared!r}, but {m['basis']}. The corpus would be filtered to one "
                f"species and scored on another's genes - matched by upper-cased symbol, so "
                f"nothing would fail and every label would come from the wrong species' markers.",
                None)
    if not m["species"] and m["convention"] and CONVENTION.get(want) not in (None, m["convention"]):
        return (None, f"note: --species {declared!r} writes its symbols {CONVENTION[want]}-case and "
                      f"this object's read {m['convention']}-case - a convention, not proof; the "
                      f"object carries no gene identifiers to settle it")
    return None, None
