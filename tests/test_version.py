"""The four places this package states its version must agree. Asserted, because they drifted.

A tool that misreports its own version does damage quietly: a run log records `0.1.0`, someone
looks up what `0.1.0` did, and reads the documentation for code that is three releases behind.
That is the same class of defect as a run citing a commit hash that does not exist, which this
project has also had.

It has now happened twice:

    0.2.0 released   VERSION said 0.2.0, scanno/__init__.py said "0.1.0"
    0.3.0 released   VERSION, CITATION.cff and __init__ said 0.3.0, pyproject.toml said "0.1.0"

The second one survived a release in which the first was explicitly noticed and fixed - the fix
updated three of the four declarations and the fourth was not looked for. So the remedy is not
more care at release time; it is this file.

`pyproject.toml` no longer states a version at all: it reads the VERSION file, the pattern scQC
uses, which removes one of the four sources rather than checking it. The check below still covers
it, so that a future edit reintroducing a literal is caught.

Stdlib only, and no build step - it must run on a bare checkout, which is how this package is
actually used (`--tool-dir`, no install).

    python tests/test_version.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def _pyproject_version(text: str):
    """(literal_version, reads_version_file). tomllib when available, regex when not.

    The regex path exists because this must run on the interpreter that happens to be present -
    macOS ships 3.9, which has no tomllib - and a test that only runs on the maintainer's machine
    is one that stops running.
    """
    try:
        import tomllib
        d = tomllib.loads(text)
        lit = d.get("project", {}).get("version")
        dyn = d.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("version", {})
        return lit, (isinstance(dyn, dict) and dyn.get("file") == "VERSION")
    except ImportError:
        body = text.split("[project]", 1)[-1].split("[", 1)[0]
        m = re.search(r'^\s*version\s*=\s*"([^"]+)"', body, re.M)
        return (m.group(1) if m else None), bool(
            re.search(r'\[tool\.setuptools\.dynamic\][^\[]*version\s*=\s*\{\s*file\s*=\s*"VERSION"',
                      text, re.S))


VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
print(f"\nVERSION file: {VERSION!r}")

print("\n1 - VERSION is a plausible version string, not a stray file")
check("non-empty", bool(VERSION))
check("looks like MAJOR.MINOR.PATCH", bool(re.fullmatch(r"\d+\.\d+\.\d+(?:[-.].+)?", VERSION)),
      VERSION)

print("\n2 - the package reports the same version it ships")
# Import when the analysis stack is present, parse the declaration when it is not. The whole
# point of this file is that it runs on a bare checkout, and `import scanno` pulls numpy - so an
# unconditional import would make the test skip exactly where a fresh clone is being checked.
try:
    import scanno  # noqa: E402
    declared, how = scanno.__version__, "imported"
except ImportError:
    src = (ROOT / "scanno" / "__init__.py").read_text(encoding="utf-8")
    m0 = re.search(r'^__version__\s*=\s*"([^"]+)"', src, re.M)
    declared, how = (m0.group(1) if m0 else None), "parsed (numpy absent)"
check("scanno/__init__.py declares __version__", declared is not None, how)
check("__version__ == VERSION", declared == VERSION,
      f"__version__={declared!r} vs VERSION={VERSION!r}  [{how}]")

print("\n3 - CITATION.cff agrees, so a citation names the code that produced the result")
cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
m = re.search(r'^version:\s*"?([^"\n]+)"?\s*$', cff, re.M)
check("CITATION.cff has a version field", m is not None)
if m:
    check("CITATION.cff == VERSION", m.group(1).strip() == VERSION,
          f"{m.group(1).strip()!r} vs {VERSION!r}")

print("\n4 - pyproject reads VERSION rather than restating it")
lit, reads_file = _pyproject_version((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
check("no literal version in [project]", lit is None,
      f"found {lit!r} - restating it is how it drifted twice")
check("[tool.setuptools.dynamic] reads the VERSION file", reads_file)
if lit is not None:
    check("...and if restated, it at least agrees", lit == VERSION, f"{lit!r} vs {VERSION!r}")

print("\n5 - the version is not accidentally the placeholder it drifted to twice")
check("not the stale 0.1.0", VERSION != "0.1.0",
      "0.1.0 was the value both drifts got stuck on")

print("\n" + "=" * 64)
if fails:
    print(f"version consistency: {len(fails)} FAILED - " + ", ".join(fails))
    raise SystemExit(1)
print(f"version consistency OK - every declaration says {VERSION}")
