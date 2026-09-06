"""Forbidden terms for the leak guards, from OUTSIDE the repository.

A guard that spells the cohort it guards against carries the leak it exists to catch. The terms
come from the file named by $SCANNO_FORBIDDEN_TERMS (one per line, # comments); the SHAPES of
site leakage — a home path, a login-node hostname, a job id, an e-mail — are spelled here
because they are not names. Stdlib only.
"""
import os
import re
from pathlib import Path

SHAPES = [
    (r"(?<![\w/])/(?:Users|home)/[A-Za-z][\w.-]*", "a user home path"),
    (r"/data/[A-Za-z][\w.-]*/home/", "a site home path"),
    (r"\blogin-\d{2}-\d{2}\b", "a login-node hostname"),
    (r"\bhn-\d{2}-\d{2}\b", "a scheduler head-node name"),
    (r"\b\d{6}\.hn-\d{2}-\d{2}\b", "a scheduler job id"),
    (r"[\w.+-]+@[\w-]+\.(?:edu|com|org|sg|ac\.uk)\b", "an e-mail address"),
    (r"scratch/\d{8}__", "a dated scratch directory"),
]
ATTRIBUTION = {"CITATION.cff", "pyproject.toml"}


def terms() -> list:
    f = os.environ.get("SCANNO_FORBIDDEN_TERMS")
    if not f or not Path(f).exists():
        return []
    return [l.strip() for l in Path(f).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def patterns():
    out = [(re.compile(p), why) for p, why in SHAPES]
    out += [(re.compile(re.escape(t), re.I), f"site term {t!r}") for t in terms()]
    return out


def hits(text: str, name: str = "") -> list:
    found = []
    for i, line in enumerate(text.splitlines(), 1):
        for rx, why in patterns():
            if why == "an e-mail address" and name in ATTRIBUTION:
                continue
            if rx.search(line):
                found.append((i, why, line.strip()[:90]))
    return found
