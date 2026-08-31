"""The gate. One subprocess per suite, and the exit code decides.

WHY A SUBPROCESS EACH, AND NOT AN IMPORT

Every suite in this directory is a module-scope script that ends in `raise SystemExit(1)` when
it fails. A runner that IMPORTS them therefore has to catch `SystemExit`, which inherits from
`BaseException` and escapes a bare `except Exception` - and a runner that terminates on the
first suite's `SystemExit` exits with THAT suite's code, hiding every file sorted after it.
That is not hypothetical: the same mistake in a sibling tool reported green for a whole session
while two suites were red.

A subprocess cannot do this. Its exit code is a fact about that file and nothing else.

WHY THIS FILE EXISTS AT ALL

Until it did, running the suites meant a shell loop typed at a prompt - a gate with no commit,
no history, and no way for anyone to run the same thing twice. The gate has to be one file in
the repository or there is no gate, only whatever the last person happened to type.

    python tests/run_all.py            every suite
    python tests/run_all.py -k compare  only files whose name contains 'compare'

VERIFY IT BY MAKING IT FAIL. Drop a file that exits non-zero into tests/ and confirm this
reports RED. A gate that has never been seen to fail proves nothing about the runs it passed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pat = None
    if "-k" in argv:
        pat = argv[argv.index("-k") + 1]

    files = sorted(p for p in HERE.glob("test_*.py") if pat is None or pat in p.name)
    if not files:
        print(f"run_all: no suites matched {pat!r}", file=sys.stderr)
        return 2

    # The tool is imported from the checkout, not from whatever is installed. One suite
    # (test_scope.py) has no sys.path insert of its own and depends on this entirely.
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}

    failed, skipped = [], []
    for f in files:
        r = subprocess.run([sys.executable, str(f)], cwd=str(ROOT), env=env,
                           capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        # A SKIP is not a PASS. It is reported on its own line so a missing dependency cannot
        # be read as a suite that ran.
        if "SKIP" in out:
            skipped.append(f.name)
        if r.returncode != 0:
            failed.append(f.name)
            print(f"RED   {f.name}   exit={r.returncode}")
            print("".join(f"      {ln}\n" for ln in out.strip().splitlines()[-12:]))
        else:
            print(f"green {f.name}" + ("   (contains a SKIP)" if "SKIP" in out else ""))

    print("=" * 64)
    print(f"{len(files)} suites   {len(failed)} red   {len(skipped)} contain a skip")
    if skipped:
        print("a skip is NOT a pass: " + ", ".join(skipped))
    if failed:
        print("RED: " + ", ".join(failed))
        return 1
    print("all suites green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
