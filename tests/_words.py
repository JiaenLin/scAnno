"""Where a text says "cell(s)" as the unit of an observation. Stdlib only.

Single cells are not single nuclei (single-cell-harness rule 2), so a print that says "cells" of
nuclei names the wrong unit. The word is also the field's generic word ("cell type"), so a scan has
to remove the phrases in which it does not name the unit before it counts what is left:

    from _words import cells_said
    cells_said("12 cells or nuclei, 3 cell types")      -> []
    cells_said("withheld 12 cells", also=("T cell",))   -> ['withheld 12 cells']

`also` removes data values first: a cell type can be called anything ("T cell"), and a label is
the data's word, not the tool's. Whitespace is collapsed before the scan, because argparse wraps
help text and "cells or" can end one line with "nuclei" on the next.

Imported BY FILE, from a suite run as a script (tests/ is then on sys.path[0]), like `_terms`.
"""
from __future__ import annotations

import re

#: Phrases that contain the word and do not name the unit of an observation, or name both units.
NOT_A_UNIT = (
    r"cells or nuclei", r"cell or nucleus",           # assay.EITHER: both units, neither assumed
    r"cell[ _-]?types?", r"celltypes?",               # the field's word for a label
    r"single-cell", r"whole[ -]cells?",               # an assay's own name or definition
    r"cell ontology", r"cell names?",                 # the corpus's vocabulary
    r"\(cell, gene\)",                                # a corpus claim: (cell name, gene)
    r"not cells and not animals",                     # the snrna report saying nuclei are not cells
    r"--exclude-mode cell\b",                         # a retired option's value, quoted to name it
)
_NOT = re.compile("|".join(NOT_A_UNIT), re.I)
UNIT = re.compile(r"\bcells?\b|\bcell\(s\)", re.I)


def cells_said(text: str, also=()) -> list:
    """Each place `text` names the unit as cell(s), with a little context, after removing every
    NOT_A_UNIT phrase and every literal in `also`. Empty means the text never does."""
    t = " ".join(str(text).split())
    for lit in sorted({str(x) for x in also if str(x)}, key=len, reverse=True):
        t = t.replace(lit, " ")
    t = _NOT.sub(" ", t)
    return [t[max(0, m.start() - 40):m.end() + 25].strip() for m in UNIT.finditer(t)]
