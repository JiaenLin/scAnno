"""What a command leaves behind so that a reader who did not watch it can tell four states apart:
ok, partial, refused, failed — from the filesystem alone.

scAnno is several commands that compose into one run directory (cluster → annotate → scope →
annotate --scope → report), so the status files carry the COMMAND in their name and sit beside
the command's primary output:

    STATUS.<cmd>.json   written FIRST as `partial`, rewritten LAST with the outcome and products
    RUNNING.<cmd>.txt   at start; replaced at exit by SEALED.<cmd>.txt or FAILED.<cmd>.txt

The directory is `--status-dir` when given, else the directory the command's primary output
lands in. A command that dies leaves `partial` and RUNNING standing — a different fact from
refused and from ok. Stdlib only; the same shape as the status contract shared with the tools
this one is orchestrated beside.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

CONTRACT = "1.0"
STATUSES = ("ok", "partial", "refused", "failed")
ROOT = Path(__file__).resolve().parents[1]

#: the last refusal a command recorded through `refuse()`, so the wrapper can carry its fix
LAST_REFUSAL: dict = {}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def commit() -> str | None:
    """The checkout's commit, read from .git by FILE — compute nodes have no git binary."""
    git = ROOT / ".git"
    try:
        if git.is_file():
            git = Path(git.read_text().split(":", 1)[1].strip())
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(None, 1)[1]
            f = git / ref
            if f.exists():
                return f.read_text().strip()
            packed = git / "packed-refs"
            if packed.exists():
                for ln in packed.read_text().splitlines():
                    if ln.endswith(" " + ref):
                        return ln.split()[0]
            return None
        return head
    except OSError:
        return None


def job() -> dict:
    for var, name in (("PBS_JOBID", "pbs"), ("SLURM_JOB_ID", "slurm")):
        if os.environ.get(var):
            return {"scheduler": name, "id": os.environ[var], "host": socket.gethostname()}
    return {"scheduler": None, "id": None, "host": socket.gethostname()}


def refuse(reason: str, fix: str = "", code: str = "") -> int:
    """Print the refusal the way every site always has, and record it for the status file."""
    print(f"scanno: REFUSE - {reason}", file=sys.stderr)
    LAST_REFUSAL.clear()
    LAST_REFUSAL.update({"reason": reason, "fix": fix, "code": code})
    return 2


def status_dir(a) -> Path | None:
    """Where this command's status lands: --status-dir, else beside its primary output."""
    sd = getattr(a, "status_dir", None)
    if sd:
        return Path(sd)
    for attr in ("out_dir", "out", "out_h5ad", "report", "out_report", "out_table"):
        v = getattr(a, attr, None)
        if v:
            v = Path(v)
            if attr in ("out_dir",) or (attr == "out" and getattr(a, "cmd", "") in ("report", "lab")):
                return v
            return v.parent
    return None


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def products_of(d: Path, cmd: str) -> list:
    out = []
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".csv", ".tsv", ".html", ".h5ad", ".npz", ".md") \
                and not p.name.startswith(("STATUS.", "RUNNING.", "SEALED.", "FAILED.")):
            out.append({"path": _rel(p, d), "bytes": p.stat().st_size})
    return out[:500]


def begin(d: Path, cmd: str, *, version: str, state_version: int, sees: list, cannot_show: list,
          argv: list | None = None, escapes: list | None = None) -> Path:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"contract": CONTRACT, "tool": "scanno", "command": cmd, "version": version,
           "commit": commit(), "state_version": state_version, "status": "partial",
           "headline": "started", "started": _now(), "finished": None, "job": job(),
           "python": sys.version.split()[0], "argv": list(argv if argv is not None else sys.argv),
           "products": [], "absent": [], "refusal": None, "sees": list(sees),
           "escapes": list(escapes or []), "cannot_show": list(cannot_show), "wrapped_versions": {}}
    (d / f"STATUS.{cmd}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    (d / f"RUNNING.{cmd}.txt").write_text(
        f"started={rec['started']}\njobid={rec['job']['id'] or 'none'}\nhost={rec['job']['host']}\n"
        f"commit={rec['commit'] or 'unidentified'}\ncommand={cmd}\n", encoding="utf-8")
    return d / f"STATUS.{cmd}.json"


def finish(d: Path, cmd: str, *, status: str, headline: str, exit_code: int,
           refusal: dict | None = None, expected: list | None = None,
           wrapped_versions: dict | None = None) -> dict:
    if status not in STATUSES:
        raise ValueError(f"status {status!r} is not one of {STATUSES}")
    d = Path(d)
    p = d / f"STATUS.{cmd}.json"
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rec = {"contract": CONTRACT, "tool": "scanno", "command": cmd, "started": None, "job": job()}
    products = products_of(d, cmd)
    present = {x["path"] for x in products if x["bytes"] > 0}
    missing = [e for e in (expected or []) if e not in present]
    if status == "ok" and missing:
        status = "failed"
        headline = f"{headline}; missing products: {', '.join(missing)}"
    if status == "refused":
        refusal = dict(refusal or LAST_REFUSAL or {})
        refusal.setdefault("reason", headline)
        refusal.setdefault("fix", "the refusal names what to change; read the line above it")
    rec.update({"status": status, "headline": headline, "finished": _now(), "exit": exit_code,
                "products": products, "missing": missing,
                "refusal": refusal if status == "refused" else None,
                "wrapped_versions": dict(wrapped_versions or rec.get("wrapped_versions") or {})})
    p.write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    sealed = status == "ok" and exit_code == 0 and not missing
    seal = d / (f"SEALED.{cmd}.txt" if sealed else f"FAILED.{cmd}.txt")
    lines = [f"exit={exit_code}", f"status={status}", f"command={cmd}",
             f"jobid={(rec.get('job') or {}).get('id') or 'none'}",
             f"commit={rec.get('commit') or 'unidentified'}", f"started={rec.get('started')}",
             f"finished={rec['finished']}", f"host={socket.gethostname()}",
             "products=" + " ".join(x["path"] for x in products[:50])]
    if missing:
        lines.append("missing=" + " ".join(missing))
    seal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    other = d / (f"FAILED.{cmd}.txt" if sealed else f"SEALED.{cmd}.txt")
    if other.exists():
        other.unlink()
    if (d / f"RUNNING.{cmd}.txt").exists():
        (d / f"RUNNING.{cmd}.txt").unlink()
    return rec
