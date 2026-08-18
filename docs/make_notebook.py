#!/usr/bin/env python3
"""Generate `reading_the_output.ipynb` from `READING_THE_OUTPUT.md`.

THE MARKDOWN IS THE SOURCE. The notebook is built from it, never edited by hand, and
`tests/test_playbook.py` fails if the two disagree.

Two copies of the same instructions drift, and the drift is invisible: the notebook still runs,
still produces figures, and quietly teaches a parameter the document no longer recommends. This
project has already been bitten by a rule existing in two places, so the second copy is
generated rather than maintained.

    python docs/make_notebook.py            # write the notebook
    python docs/make_notebook.py --check    # exit 1 if it is out of date, change nothing
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / "READING_THE_OUTPUT.md"
NB = HERE / "reading_the_output.ipynb"

#: Fences that are prose, not code to run. ```text blocks are warnings meant to be READ.
RUNNABLE = {"python", "py"}


def cells(md: str):
    """Markdown -> notebook cells, preserving order: prose becomes markdown cells, ```python
    becomes code cells, and any other fence stays inside the prose it belongs to."""
    out, buf = [], []

    def flush_md():
        text = "".join(buf).strip("\n")
        buf.clear()
        if text:
            out.append({"cell_type": "markdown", "metadata": {}, "source": _lines(text)})

    i, lines = 0, md.splitlines(keepends=True)
    while i < len(lines):
        m = re.match(r"^```(\w*)\s*$", lines[i])
        if not m:
            buf.append(lines[i])
            i += 1
            continue
        lang = m.group(1).lower()
        j = i + 1
        while j < len(lines) and not re.match(r"^```\s*$", lines[j]):
            j += 1
        body = "".join(lines[i + 1:j])
        if lang in RUNNABLE:
            flush_md()
            out.append({"cell_type": "code", "execution_count": None, "metadata": {},
                        "outputs": [], "source": _lines(body.rstrip("\n"))})
        else:
            buf.append(lines[i]); buf.extend(lines[i + 1:j]); buf.append(lines[j] if j < len(lines) else "```\n")
        i = j + 1
    flush_md()
    return out


def _lines(text: str):
    """nbformat wants a list of lines, each keeping its newline except the last."""
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def build() -> str:
    nb = {
        "cells": cells(MD.read_text(encoding="utf-8")),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            # Not decoration: a reader who opens the notebook first has to be told where it came
            # from, or they will edit it and lose the edit on the next regeneration.
            "scanno": {"generated_from": MD.name,
                       "warning": "GENERATED FILE - edit READING_THE_OUTPUT.md and re-run "
                                  "docs/make_notebook.py"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main(argv):
    want = build()
    if "--check" in argv:
        if not NB.exists():
            print(f"{NB.name} does not exist. Run: python docs/make_notebook.py")
            return 1
        if NB.read_text(encoding="utf-8") != want:
            print(f"{NB.name} is OUT OF DATE with {MD.name}. Run: python docs/make_notebook.py")
            return 1
        print(f"{NB.name} matches {MD.name}")
        return 0
    NB.write_text(want, encoding="utf-8")
    n_code = sum(1 for c in json.loads(want)["cells"] if c["cell_type"] == "code")
    print(f"wrote {NB.name}: {n_code} code cell(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
