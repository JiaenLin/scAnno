"""The names an object's genes are matched on: ONE reader, for every command that matches them.

A marker corpus is keyed by SYMBOL. An object is very often keyed by accession, with the symbols
beside it in `var` - the right way round for the object, since symbols are not unique. So which
names a command reads decides whether anything matches at all, and two commands that read
different names give two answers about the same object without either one failing.

That is what happened. `annotate` and `background` read `var['gene_symbol']` when it was present;
`build-markers --reference-genes <h5ad>` read `var_names`. On an object keyed by accession every
marker in the corpus then read as unmeasurable (`in_reference` 0 for every assertion), and the
corpus was still written, with STATUS ok. `background` also differed from `annotate` on a column
that was named and absent: `annotate` refused, `background` silently fell back to `var_names`.

The rule, for all three:
  --gene-key COLUMN   that var column; refused, by name, when the object has no such column;
  --gene-key ''       the object's var_names, said so;
  no --gene-key       var['gene_symbol'] when present, else var_names.
The names come back as stored. Case is the caller's business: `annotate` and `background` fold to
upper case, and `build-markers` matches case-insensitively and also records the exact spelling.
"""
from __future__ import annotations

#: The column read when no --gene-key is given and the object has it.
DEFAULT_GENE_KEY = "gene_symbol"


class GeneKeyMissing(ValueError):
    """A var column was named and the object has no such column. Never a fall back to var_names."""

    def __init__(self, key, columns):
        self.key, self.columns = key, list(columns)
        super().__init__(f"no var column {key!r}. Columns: {', '.join(self.columns) or 'none'}")


def gene_key_for(var, gene_key=None):
    """The var column the names are read from, or None for var_names. See the module docstring."""
    if gene_key is None:
        return DEFAULT_GENE_KEY if DEFAULT_GENE_KEY in var else None
    if gene_key == "":
        return None
    if gene_key not in var:
        raise GeneKeyMissing(gene_key, map(str, var.columns))
    return gene_key


def gene_names(obj, gene_key=None):
    """(names, source) of `obj` - an AnnData, its `.raw`, or anything with `.var` and `.var_names`.

    `names` is a list of str in the object's gene order; `source` is "var['<column>']" or
    "var_names", for a record or a message. Raises GeneKeyMissing for a named, absent column."""
    key = gene_key_for(obj.var, gene_key)
    values = obj.var[key] if key is not None else obj.var_names
    return [str(v) for v in values], (f"var[{key!r}]" if key is not None else "var_names")
