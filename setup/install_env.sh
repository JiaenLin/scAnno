#!/usr/bin/env bash
# Create the environment scAnno runs in, or report what is missing from the one you have.
#
#   setup/install_env.sh --prefix ~/envs/scanno        # create it
#   setup/install_env.sh --check                       # audit the CURRENT interpreter, change nothing
#   setup/install_env.sh --check --python /path/to/python
#
# scAnno's decision layer is numpy + scipy only. This script exists for the ANALYSIS stack -
# anndata, scanpy, leiden - which `pip install -e '.[run]'` assumes you already have somewhere.
# That assumption is the whole gap: it documents the half of installation that is easy.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
LOCK="$HERE/environment.lock.yml"
PREFIX=""; MODE="create"; PY=""

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do case "$1" in
  --prefix) PREFIX="${2:?--prefix needs a path}"; shift 2 ;;
  --check)  MODE="check"; shift ;;
  --python) PY="${2:?--python needs a path}"; shift 2 ;;
  -h|--help) usage 0 ;;
  *) echo "unknown option: $1" >&2; usage 2 ;;
esac; done

# ---------------------------------------------------------------- check: report, never fix
if [ "$MODE" = check ]; then
  PY="${PY:-$(command -v python3 || command -v python)}"
  [ -x "$PY" ] || { echo "no python found. Pass --python /path/to/python"; exit 2; }
  echo "checking: $PY"
  "$PY" - "$LOCK" <<'PYEOF'
import sys, re, pathlib
import importlib.metadata as md
lock = pathlib.Path(sys.argv[1]).read_text()
want = dict(re.findall(r'^\s+- ([A-Za-z0-9_.-]+)==([0-9][^\s]*)', lock, re.M))
pyv = re.search(r'python=([0-9.]+)', lock)
print(f"  python {sys.version.split()[0]}" + (f"   (locked {pyv.group(1)})" if pyv else ""))
missing, differ, ok = [], [], 0
for p, v in want.items():
    try:
        got = md.version(p)
    except Exception:
        missing.append(p); continue
    if got == v: ok += 1
    else: differ.append((p, got, v))
print(f"  {ok} package(s) at the locked version")
for p, got, v in differ:
    print(f"  DIFFERS  {p:<14} have {got:<10} locked {v}")
for p in missing:
    print(f"  MISSING  {p}")
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
if not core and not opt and not differ:
    print("  environment matches the lock exactly")
sys.exit(1 if core else 0)
PYEOF
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

echo "creating with: $MAMBA"
echo "prefix       : $PREFIX"
"$MAMBA" env create --yes --prefix "$PREFIX" --file "$LOCK" || {
  echo; echo "environment creation FAILED. Nothing was left half-built at $PREFIX?"; 
  echo "Check with: ls '$PREFIX'"; exit 1; }
"$PREFIX/bin/pip" install -e "$ROOT" >/dev/null 2>&1 || echo "  note: editable install of scAnno itself did not run; do it by hand"
echo
echo "done. Verify, then use it by PATH - do not rely on activation:"
echo "  $PREFIX/bin/python -m scanno.cli selftest"
echo
echo "On a cluster, the interpreter is the PROJECT'S and the code is the TOOL'S:"
echo "  PYTHONPATH=$ROOT  $PREFIX/bin/python -m scanno.cli report --out <inside your project>"
