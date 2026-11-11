#!/usr/bin/env python3
"""setup/environment.lock.yml pins every package the resolver could otherwise move.

THE DEFECT (single-cell-harness GAP-C82). The lock pinned the 15 packages scAnno imports by name
and none of what they require. The manager's pip step is `pip install -U -r`, so every one of the
36 unpinned dependencies was resolved at build time to its newest release. On 2026-11-10 one of
them, pynndescent 0.7.0, returned a `csr_array` that scanpy 1.12.3 calls `.getnnz` on, and
`scanno cluster` died on every object of 8,192 cells or more. Nothing in the lock had changed.

WHAT IS CHECKED, for the lock's own platform (linux x86_64, the lock's CPython): for every pinned
package, what that version requires - and what it requires under every extra another pin asks
for - is itself pinned, at a version the requirement accepts. The requirements are read from
setup/environment.lock.requires.tsv, which records each pinned version's own metadata; a pin with
no record is read from the installed distribution when that is the pinned version, and is a
failure when neither says what it requires. Wherever both exist they must agree, so the record
cannot drift from the packages it describes. Offline: nothing is fetched.

TO RE-RECORD after changing a pin (needs the network; nothing is installed):
    pip install --dry-run --ignore-installed --report report.json --only-binary=:all: \\
        --python-version 3.12 --implementation cp --abi cp312 \\
        --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \\
        -r <the lock's pip pins, one per line>
    python tests/test_the_lock_pins_its_closure.py --record report.json > setup/environment.lock.requires.tsv
A report that resolves a package the lock does not pin is refused: pin it first.

    python tests/test_the_lock_pins_its_closure.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "setup" / "environment.lock.yml"
RECORD = ROOT / "setup" / "environment.lock.requires.tsv"

try:
    from packaging.requirements import Requirement
except ImportError:
    try:
        from pip._vendor.packaging.requirements import Requirement
    except ImportError:
        print("SKIP the lock's closure: needs `packaging` (or pip's copy of it). A skip is not a pass.")
        raise SystemExit(0)

norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()  # noqa: E731  (PEP 503)


def read_lock(text):
    py = re.search(r"^\s+- python=(\S+)\s*$", text, re.M)
    items = re.findall(r"^ {6}- (\S+)\s*$", text, re.M)
    pins = {}
    for item in items:
        name, _, version = item.partition("==")
        pins[norm(name)] = (name, version)
    return (py.group(1) if py else None), items, pins


PYTHON, ITEMS, PINS = read_lock(LOCK.read_text(encoding="utf-8"))
TARGET = {"implementation_name": "cpython", "implementation_version": PYTHON or "",
          "os_name": "posix", "platform_machine": "x86_64", "platform_python_implementation": "CPython",
          "platform_release": "", "platform_system": "Linux", "platform_version": "",
          "python_full_version": PYTHON or "", "sys_platform": "linux",
          "python_version": ".".join((PYTHON or "").split(".")[:2])}


def applies(req, extra):
    return req.marker is None or req.marker.evaluate({**TARGET, "extra": extra})


def key(req):
    """A requirement without its marker, comparable across packaging versions."""
    return (norm(req.name), tuple(sorted(norm(e) for e in req.extras)),
            ",".join(sorted(str(s) for s in req.specifier)))


def from_metadata(requires, extras):
    """{extra: [requirement keys]} for '' and each extra, from a Requires-Dist list."""
    out = {}
    for extra in [""] + sorted(extras):
        reqs = [Requirement(s) for s in requires or []]
        base = {key(r) for r in reqs if applies(r, "")}
        out[extra] = sorted({key(r) for r in reqs if applies(r, extra)} - (base if extra else set()))
    return out


def requested_extras(rows_of):
    """Every extra some pinned package asks of another, to a fixed point."""
    want = {n: set() for n in rows_of}
    changed = True
    while changed:
        changed = False
        for n in rows_of:
            for extra in [""] + sorted(want[n]):
                for name, extras, _spec in rows_of[n](extra):
                    if name in want and not set(extras) <= want[name]:
                        want[name] |= set(extras)
                        changed = True
    return want


def record(report_path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    meta = {norm(i["metadata"]["name"]): i["metadata"] for i in report["install"]}
    stray = sorted(set(meta) - set(PINS))
    if stray:
        sys.exit("the report resolves packages the lock does not pin: " + ", ".join(stray) + ". Pin them first.")
    reqs = {n: [Requirement(s) for s in m.get("requires_dist") or []] for n, m in meta.items()}
    rows_of = {n: (lambda e, n=n: [key(r) for r in reqs[n] if applies(r, e)]) for n in meta}
    want = requested_extras(rows_of)
    print(f"# What each pinned version requires on linux x86_64 under CPython {PYTHON}, read from its own "
          f"metadata (a pip --report).\n# Written by tests/test_the_lock_pins_its_closure.py --record; "
          f"checked by the same file. extra '' is the base requirement set.")
    print("package\tversion\textra\trequirement")
    for n in sorted(meta):
        rows = from_metadata(meta[n].get("requires_dist"), want[n])
        printed = False
        for extra in [""] + sorted(want[n]):
            for name, extras, spec in rows[extra]:
                print(f"{n}\t{meta[n]['version']}\t{extra}\t{name}{'[' + ','.join(extras) + ']' if extras else ''}{spec}")
                printed = True
        if not printed:
            print(f"{n}\t{meta[n]['version']}\t\t-")


if len(sys.argv) == 3 and sys.argv[1] == "--record":
    record(sys.argv[2])
    raise SystemExit(0)

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        fails.append(name)


print("\n1 - the lock names exact versions")
check("the interpreter is pinned to a patch release", bool(PYTHON and re.fullmatch(r"\d+\.\d+\.\d+", PYTHON)),
      str(PYTHON))
loose = [i for i in ITEMS if not re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][^\s=<>!~,;]*", i)]
check("every pip line is name==version", not loose, ", ".join(loose))
check("no package is pinned twice", len(PINS) == len(ITEMS), f"{len(ITEMS)} lines, {len(PINS)} names")

recorded = {}
if RECORD.is_file():
    for line in RECORD.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("package\t") or not line.strip():
            continue
        n, v, extra, req = line.split("\t")
        recorded.setdefault(n, {"version": v, "rows": {}})
        if req != "-":
            recorded[n]["rows"].setdefault(extra, []).append(key(Requirement(req)))

try:
    import importlib.metadata as md
except ImportError:  # pragma: no cover - python < 3.8
    md = None


def installed(n):
    if md is None:
        return None
    try:
        d = md.distribution(PINS[n][0])
    except Exception:
        return None
    return d if d.version == PINS[n][1] else None


source, cross = {}, []
for n in PINS:
    if n in recorded and recorded[n]["version"] == PINS[n][1]:
        source[n] = lambda e, n=n: recorded[n]["rows"].get(e, [])
    elif installed(n) is not None:
        source[n] = lambda e, n=n: from_metadata(installed(n).requires, {e} if e else set())[e]
    else:
        source[n] = None

print(f"\n2 - every pinned package's requirements are known ({len(recorded)} recorded, "
      f"the rest read from the installed distribution)")
unknown = sorted(PINS[n][0] + "==" + PINS[n][1] for n, s in source.items() if s is None)
check("each pin says what it requires (setup/environment.lock.requires.tsv, or its installed metadata)",
      not unknown, ", ".join(unknown))
stale = sorted(f"{n} {r['version']}" for n, r in recorded.items() if n not in PINS or PINS[n][1] != r["version"])
check("the record describes the pinned versions and nothing else", not stale, ", ".join(stale))

print("\n3 - what they require is pinned, at a version the requirement accepts")
known = {n: s for n, s in source.items() if s is not None}
want = requested_extras(known)
unpinned, unmet = set(), set()
for n, rows in known.items():
    for extra in [""] + sorted(want[n]):
        for name, _extras, spec in rows(extra):
            if name not in PINS:
                unpinned.add(f"{name} (required by {PINS[n][0]})")
            elif spec and not Requirement(f"{name}{spec}").specifier.contains(PINS[name][1], prereleases=True):
                unmet.add(f"{PINS[n][0]} needs {name}{spec}, the lock has {PINS[name][1]}")
check("every requirement is pinned: nothing is left for the resolver to choose", not unpinned,
      f"{len(unpinned)} unpinned: " + ", ".join(sorted(unpinned)))
check("every pin satisfies what requires it", not unmet, "; ".join(sorted(unmet)))

print("\n4 - the record is the packages' own metadata, wherever both can be read")
for n in recorded:
    d = installed(n) if n in PINS else None
    if d is None:
        continue
    have = from_metadata(d.requires, want.get(n, set()))
    rec = {e: sorted(recorded[n]["rows"].get(e, [])) for e in have}
    if rec != have:
        cross.append(n)
n_cross = sum(1 for n in recorded if n in PINS and installed(n) is not None)
check(f"the record agrees with the installed metadata ({n_cross} of {len(recorded)} could be compared)",
      not cross, ", ".join(cross))

print("\n" + "=" * 64)
if fails:
    print(f"the lock's closure: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print(f"the lock's closure OK - {len(PINS)} pins, every requirement among them pinned and satisfied")
