"""Whether the cohort is cells or nuclei is a fact of the experiment, so `--assay` has no default.

It defaulted to `sc`. A nucleus cohort that forgot the flag was described to the agent as whole
cells, and the run recorded `sc` as its context - and nothing would say so. The same defect, the
other way round, was found in scQC by a second cohort (single-cell-harness ADR-0027): its
mitochondrial step read `assay` with a default of `snrna`, so a whole-cell cohort was bounded as
nuclei. `sch conform` S13 now refuses a literal default for any cohort axis, in any repository.

Every subcommand that takes `--assay` must refuse to run without it, and name it.

Stdlib only; no data is read - argparse refuses before anything is opened.

    python tests/test_assay_has_no_default.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

fails = []
for sub in ("background", "annotate", "calibrate", "agent"):
    r = subprocess.run([sys.executable, "-m", "scanno.cli", sub], cwd=str(ROOT),
                       capture_output=True, text=True, env={"PYTHONPATH": str(ROOT), "PATH": ""})
    said = (r.stdout or "") + (r.stderr or "")
    # IN THE "REQUIRED" LINE, not anywhere: without its other arguments every subcommand refuses
    # anyway, and the usage line it prints names `[--assay {sc,sn}]` whether or not it is required
    # - the first draft of this test passed against the default it was written to remove.
    req = [ln for ln in said.splitlines() if "the following arguments are required" in ln]
    ok = r.returncode == 2 and any("--assay" in ln for ln in req)
    print(f"  {'PASS' if ok else 'FAIL'}  scanno {sub} without --assay is refused, naming it   rc={r.returncode}")
    if not ok:
        fails.append(f"{sub}: rc={r.returncode}, said: {said.strip().splitlines()[-1:] or said!r}")

print("=" * 64)
if fails:
    print(f"assay: {len(fails)} FAILED - " + "; ".join(fails))
    raise SystemExit(1)
print("assay: every subcommand that takes it requires it")
