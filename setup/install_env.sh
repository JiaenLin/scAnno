#!/usr/bin/env bash
# Create the environment scAnno runs in, or report what is missing from the one you have.
#
#   setup/install_env.sh --prefix ~/envs/scanno        # create it, then check it against the lock
#   setup/install_env.sh --prefix ~/envs/scanno --no-editable   # and write nothing into this checkout
#   setup/install_env.sh --check                       # audit the CURRENT interpreter, change nothing
#   setup/install_env.sh --check --python /path/to/python
#   setup/install_env.sh --check --strict --python P   # also refuse what the lock does not pin
#
# scAnno's decision layer is numpy + scipy only. This script exists for the ANALYSIS stack -
# anndata, scanpy, leiden - which `pip install -e '.[run]'` assumes you already have somewhere.
# That assumption is the whole gap: it documents the half of installation that is easy.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
LOCK="$HERE/environment.lock.yml"
PREFIX=""; MODE="create"; PY=""; STRICT=0; EDITABLE=1

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do case "$1" in
  --prefix) PREFIX="${2:?--prefix needs a path}"; shift 2 ;;
  --check)  MODE="check"; shift ;;
  --strict) STRICT=1; shift ;;
  --python) PY="${2:?--python needs a path}"; shift 2 ;;
  --no-editable) EDITABLE=0; shift ;;
  -h|--help) usage 0 ;;
  *) echo "unknown option: $1" >&2; usage 2 ;;
esac; done

# ---------------------------------------------------------------- check: report, never fix
# --strict is what a build is held to: every pin at its version, and no distribution the lock
# does not pin, except pip, setuptools and wheel (the manager's half) and scAnno itself.
check() {
  "$1" - "$LOCK" "$2" <<'PYEOF'
import sys, re, pathlib
import importlib.metadata as md
lock = pathlib.Path(sys.argv[1]).read_text()
strict = sys.argv[2] == "1"
norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()
want = dict(re.findall(r'^\s+- ([A-Za-z0-9_.-]+)==([0-9][^\s]*)', lock, re.M))
pyv = re.search(r'python=([0-9.]+)', lock)
have_py = sys.version.split()[0]
print(f"  python {have_py}" + (f"   (locked {pyv.group(1)})" if pyv else ""))
missing, differ, ok = [], [], 0
for p, v in want.items():
    try:
        got = md.version(p)
    except Exception:
        missing.append(p); continue
    if got == v: ok += 1
    else: differ.append((p, got, v))
print(f"  {ok} package(s) at the locked version, of {len(want)} locked")
for p, got, v in differ:
    print(f"  DIFFERS  {p:<14} have {got:<10} locked {v}")
for p in missing:
    print(f"  MISSING  {p}")
locked = {norm(p) for p in want}
manager = {"pip", "setuptools", "wheel"}
present = {norm(d.metadata["Name"]): d.version for d in md.distributions() if d.metadata["Name"]}
stray = sorted(f"{n}=={v}" for n, v in present.items() if n not in locked | manager | {"scanno"})
if strict:
    print("  outside the lock: " + (", ".join(stray) if stray else "nothing") + "   (the manager's: "
          + (", ".join(f"{n}=={present[n]}" for n in sorted(manager & set(present))) or "none") + ")")
core = [p for p in missing if p in ("anndata", "scanpy", "numpy", "scipy")]
opt  = [p for p in missing if p in ("matplotlib", "seaborn")]
print()
if core:
    print("  scAnno cannot READ .h5ad without: " + ", ".join(core))
    print("  -> pip install -e '.[run]'   (or use --prefix to build the locked environment)")
if opt:
    print("  Figures will be NAMED ABSENCES without: " + ", ".join(opt))
    print("  The report is still written; this is a degradation, not a failure.")
if differ and not core:
    print("  Importable, but results may not match a published number: clustering is not")
    print("  bit-reproducible across versions. Use --prefix if you are comparing to one.")
py_ok = not pyv or have_py == pyv.group(1)
if not core and not opt and not differ and not missing and py_ok and not (strict and stray):
    print("  environment matches the lock exactly")
if strict:
    sys.exit(0 if (not missing and not differ and not stray and py_ok) else 1)
sys.exit(1 if core else 0)
PYEOF
}

if [ "$MODE" = check ]; then
  PY="${PY:-$(command -v python3 || command -v python)}"
  [ -x "$PY" ] || { echo "no python found. Pass --python /path/to/python"; exit 2; }
  echo "checking: $PY"
  check "$PY" "$STRICT"
  exit $?
fi

# ---------------------------------------------------------------- create
[ -n "$PREFIX" ] || { echo "--prefix is required to create an environment" >&2; usage 2; }
MAMBA="$(command -v micromamba || command -v mamba || command -v conda || true)"
[ -n "$MAMBA" ] || {
  cat >&2 <<'MSG'
no conda, mamba or micromamba on PATH.

scAnno does not bundle one. Either install micromamba
  https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
or, if you only need the decision layer and already have python 3.10+:
  pip install -e '.[run]'
and then check it with:  setup/install_env.sh --check
MSG
  exit 2; }

# `--yes` is ASKED of the manager, not assumed: micromamba's `env create` takes it, and conda's did
# not until long after 4.10 (its `env create` never prompts there, and refuses the flag as an
# unrecognised argument). A cluster's module conda can be that old.
YES=()
if "$MAMBA" env create --help 2>&1 | grep -q -- '--yes'; then YES=(--yes); fi
# The manager reads A COPY of the lock: conda writes its pip requirements file beside the file it
# is given, which would put it inside this checkout.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/scanno-env.XXXXXX")" || { echo "cannot make a temporary directory" >&2; exit 1; }
trap 'rm -rf "$WORK"' EXIT
cp "$LOCK" "$WORK/environment.lock.yml"

echo "creating with: $MAMBA ($("$MAMBA" --version 2>&1 | head -1))${YES[*]:+, ${YES[*]}}"
echo "prefix       : $PREFIX"
"$MAMBA" env create ${YES[@]+"${YES[@]}"} --prefix "$PREFIX" --file "$WORK/environment.lock.yml" || {
  echo; echo "environment creation FAILED. Nothing was left half-built at $PREFIX?";
  echo "Check with: ls '$PREFIX'"; exit 1; }
if [ "$EDITABLE" = 1 ]; then
  "$PREFIX/bin/pip" install --no-deps -e "$ROOT" >/dev/null 2>&1 \
    || echo "  note: editable install of scAnno itself did not run; do it by hand"
else
  echo "  --no-editable: scAnno is not installed into the environment; it runs from this checkout by PYTHONPATH"
fi
echo
echo "checking the environment against the lock (strict: nothing the lock does not pin):"
check "$PREFIX/bin/python" 1 || {
  echo; echo "the environment does NOT match the lock: a version above differs, or the resolver added a"
  echo "package the lock does not pin. It was built, and is not the locked environment."; exit 1; }
echo
echo "done. Verify, then use it by PATH - do not rely on activation:"
echo "  $PREFIX/bin/python -m scanno.cli selftest"
echo
echo "On a cluster, the interpreter is the PROJECT'S and the code is the TOOL'S:"
echo "  PYTHONPATH=$ROOT  $PREFIX/bin/python -m scanno.cli report --out <inside your project>"
