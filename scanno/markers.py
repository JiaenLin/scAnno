"""`scanno build-markers`: an evidence-graded marker corpus, built from a CellMarker 3.0 release.

WHY THIS IS IN THE TOOL (single-cell-harness B1-030). The corpus is not distributed with scAnno, and
that was read as "the ingest is not shipped": the only ingest lived in one cohort's project, so no
annotation could run on a fresh install without somebody's in-house script. The DATABASE stays
undistributed - each project builds its own from a release it fetched - and the BUILD is this.

WHAT IT BUILDS. Not a marker list. Most rows of a CellMarker release come from a marker detector run
over somebody else's dataset (`marker_source = Method`), not from a paper asserting the marker, and a
flat {cell type: [genes]} erases that on its first line. So:

  marker      every delivered row, nothing filtered - the release as delivered, normalised into
              `tissue`, `celltype`, `gene`, `disease` and `source` tables. Narrowing is a query a
              reader can see and reverse, never a deletion.
  assertion   one row per (species, tissue_class, cell_name, symbol_norm), keyed by UPPER(symbol):
              a release spells one gene `Myh6` and `MYH6`, and keying on the raw string splits its
              evidence into two weaker claims. It carries how many publications, datasets and
              methods support the claim, the evidence tier, and the claim's specificity.
  ortholog    a mouse-human bridge by case-insensitive symbol, LABELLED as that heuristic: it misses
              renamed genes and expanded families.
  corpus_info the release's sha256 and member, the row counts, the reference gene list's size and
              sha256, the tier ladder and the scanno version that built it - so a run that read the
              corpus can say which one.

THE TIERS are an ordinal restatement of stored counts, so a different ladder is a SQL statement away:
E1 two or more experimental publications (Experiment/Review), E2 one, E3 two or more single-cell
publications, E4 one, E5 algorithm-derived only. `scanno annotate` reads tiers <= 4 by default.

THE REFERENCE GENES (`--reference-genes`) are the gene axis of the data the corpus will be used on.
They fill `in_reference` (the gene can be measured there, compared case-insensitively),
`exact_reference` (spelled the same) and the spelling `symbol_display` prefers. Without them those
two columns are NULL - unknown, not "absent" - and scAnno's annotation does not read them.

WHAT STAYED IN THE PROJECT. Views and tables for one tissue, and the project's own gene universe:
both are a cohort's, and here they are a query and an argument.

Ported from the general part of the ingest the first cohort's project wrote. The statements below
are that ingest's, so a rebuild of that project's release, with its gene list, reproduces its
`assertion` table row for row.
"""
from __future__ import annotations

import hashlib
import io
import math
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path

#: The member CellMarker 3.0 ships in its archive.
MEMBER = "all_cell_marker.txt"

DIMS = {
    "tissue": ("tissue_class", "tissue_type", "uberon_id"),
    "celltype": ("species", "cell_name", "cell_name_class", "cellontology_id"),
    "gene": ("symbol", "gene_id", "gene_type", "gene_name", "uniprot_id"),
    "disease": ("disease",),
    "source": ("marker_source", "method_details", "technology_seq", "pmid", "title",
               "journal", "year", "series_id"),
}
#: Every column the build reads. A release missing one is refused by name.
REQUIRED = ("species", "marker") + tuple(c for cols in DIMS.values() for c in cols)

SCHEMA = """
PRAGMA journal_mode = OFF;
PRAGMA synchronous  = OFF;
PRAGMA temp_store   = MEMORY;
PRAGMA cache_size   = -400000;

CREATE TABLE tissue   (tissue_id   INTEGER PRIMARY KEY, tissue_class TEXT, tissue_type TEXT,
                       uberon_id TEXT, is_fetal INTEGER);
CREATE TABLE celltype (celltype_id INTEGER PRIMARY KEY, species TEXT, cell_name TEXT,
                       cell_name_class TEXT, cellontology_id TEXT);
CREATE TABLE gene     (gene_id_pk  INTEGER PRIMARY KEY, symbol TEXT, gene_id TEXT,
                       gene_type TEXT, gene_name TEXT, uniprot_id TEXT, in_reference INTEGER,
                       symbol_norm TEXT, exact_reference INTEGER);
CREATE TABLE disease  (disease_id  INTEGER PRIMARY KEY, disease TEXT, is_normal INTEGER);
CREATE TABLE source   (source_id   INTEGER PRIMARY KEY, marker_source TEXT, method_details TEXT,
                       technology_seq TEXT, pmid TEXT, title TEXT, journal TEXT, year TEXT,
                       series_id TEXT);
CREATE TABLE marker (
    row_id      INTEGER PRIMARY KEY,
    species     TEXT    NOT NULL,
    tissue_id   INTEGER NOT NULL,
    celltype_id INTEGER NOT NULL,
    gene_id_pk  INTEGER NOT NULL,
    disease_id  INTEGER NOT NULL,
    source_id   INTEGER NOT NULL,
    marker_raw  TEXT
);
"""

INDICES = """
CREATE INDEX ix_marker_species  ON marker(species);
CREATE INDEX ix_marker_tissue   ON marker(tissue_id);
CREATE INDEX ix_marker_celltype ON marker(celltype_id);
CREATE INDEX ix_marker_gene     ON marker(gene_id_pk);
CREATE INDEX ix_marker_disease  ON marker(disease_id);
CREATE INDEX ix_marker_source   ON marker(source_id);
CREATE INDEX ix_gene_symbol     ON gene(symbol);
CREATE INDEX ix_gene_inref      ON gene(in_reference);
CREATE INDEX ix_celltype_name   ON celltype(cell_name);
CREATE INDEX ix_celltype_cl     ON celltype(cellontology_id);
CREATE INDEX ix_tissue_class    ON tissue(tissue_class);
CREATE INDEX ix_source_kind     ON source(marker_source);
"""

# `curated` restates marker_source: rows a study reported (Experiment, Review, Single-cell
# sequencing) against rows the release generated by running a tool. Not a quality judgement.
VIEW_MARKER = """
CREATE VIEW v_marker AS
SELECT m.row_id, m.species,
       t.tissue_class, t.tissue_type, t.uberon_id, t.is_fetal,
       c.cell_name, c.cell_name_class, c.cellontology_id,
       g.symbol, g.symbol_norm, g.exact_reference, g.gene_type, g.gene_name, g.uniprot_id,
       g.in_reference,
       d.disease, d.is_normal,
       s.marker_source, s.method_details, s.technology_seq, s.pmid, s.title,
       s.journal, s.year, s.series_id,
       CASE WHEN s.marker_source IN ('Experiment','Review','Single-cell sequencing')
            THEN 1 ELSE 0 END AS curated,
       CASE WHEN s.marker_source IN ('Experiment','Review') THEN 1 ELSE 0 END AS experimental
FROM marker m
JOIN tissue   t ON t.tissue_id   = m.tissue_id
JOIN celltype c ON c.celltype_id = m.celltype_id
JOIN gene     g ON g.gene_id_pk  = m.gene_id_pk
JOIN disease  d ON d.disease_id  = m.disease_id
JOIN source   s ON s.source_id   = m.source_id;
"""

ASSERTION = """
CREATE TABLE assertion AS
SELECT species, tissue_class, cell_name, symbol_norm,
       COALESCE(MAX(CASE WHEN exact_reference = 1 THEN symbol END),
                MIN(symbol))                                           AS symbol_display,
       COUNT(DISTINCT symbol)                                          AS n_case_variants,
       MAX(cellontology_id)                                            AS cellontology_id,
       MAX(cell_name_class)                                            AS cell_name_class,
       MAX(in_reference)                                               AS in_reference,
       MAX(exact_reference)                                            AS exact_reference,
       MAX(gene_type)                                                  AS gene_type,
       COUNT(*)                                                        AS n_rows,
       COUNT(DISTINCT CASE WHEN pmid       <> '' THEN pmid       END)  AS n_pmids,
       COUNT(DISTINCT CASE WHEN series_id  <> '' THEN series_id  END)  AS n_series,
       COUNT(DISTINCT CASE WHEN method_details <> '' THEN method_details END) AS n_methods,
       COUNT(DISTINCT CASE WHEN experimental = 1 AND pmid <> '' THEN pmid END) AS n_pmids_exp,
       COUNT(DISTINCT CASE WHEN curated = 1 AND pmid <> '' THEN pmid END)      AS n_pmids_cur,
       SUM(CASE WHEN marker_source = 'Experiment'             THEN 1 ELSE 0 END) AS n_experiment,
       SUM(CASE WHEN marker_source = 'Review'                 THEN 1 ELSE 0 END) AS n_review,
       SUM(CASE WHEN marker_source = 'Single-cell sequencing' THEN 1 ELSE 0 END) AS n_singlecell,
       SUM(CASE WHEN marker_source = 'Method'                 THEN 1 ELSE 0 END) AS n_method,
       SUM(CASE WHEN marker_source = 'Company'                THEN 1 ELSE 0 END) AS n_company,
       SUM(is_normal)                                                  AS n_normal,
       SUM(1 - is_normal)                                              AS n_disease,
       SUM(is_fetal)                                                   AS n_fetal,
       SUM(1 - is_fetal)                                               AS n_adult,
       MIN(NULLIF(year, ''))                                           AS first_year,
       MAX(NULLIF(year, ''))                                           AS last_year
FROM v_marker
GROUP BY species, tissue_class, cell_name, symbol_norm;
"""

# The ladder. Every input is a stored column of `assertion`. Ordinal: 1 is strongest.
TIER = """
ALTER TABLE assertion ADD COLUMN evidence_tier INTEGER;
UPDATE assertion SET evidence_tier = CASE
    WHEN n_pmids_exp >= 2                                   THEN 1
    WHEN (n_experiment + n_review) > 0                      THEN 2
    WHEN n_singlecell > 0 AND n_pmids >= 2                  THEN 3
    WHEN n_singlecell > 0                                   THEN 4
    ELSE 5 END;
ALTER TABLE assertion ADD COLUMN n_celltypes_same_tissue INTEGER;
ALTER TABLE assertion ADD COLUMN specificity REAL;
ALTER TABLE assertion ADD COLUMN top_celltype_share REAL;
ALTER TABLE assertion ADD COLUMN specificity_pub REAL;
ALTER TABLE assertion ADD COLUMN n_celltypes_pub INTEGER;
"""

TIER_LABELS = {
    1: "E1 replicated experimental - 2+ independent publications, Experiment/Review",
    2: "E2 single experimental publication - Experiment/Review, one PMID",
    3: "E3 replicated single-cell literature - 2+ publications reporting it",
    4: "E4 single single-cell publication",
    5: "E5 algorithm-derived only - a marker detector ran; no paper asserts it",
}

INDICES2 = """
CREATE INDEX ix_a_species   ON assertion(species);
CREATE INDEX ix_a_tissue    ON assertion(tissue_class);
CREATE INDEX ix_a_cell      ON assertion(cell_name);
CREATE INDEX ix_a_symbol    ON assertion(symbol_norm);
CREATE INDEX ix_a_disp      ON assertion(symbol_display);
CREATE INDEX ix_a_specpub   ON assertion(specificity_pub);
CREATE INDEX ix_a_tier      ON assertion(evidence_tier);
CREATE INDEX ix_a_inref     ON assertion(in_reference);
CREATE INDEX ix_a_key       ON assertion(species, tissue_class, cell_name);
CREATE INDEX ix_orth_mouse  ON ortholog(symbol_mouse);
CREATE INDEX ix_orth_human  ON ortholog(symbol_human);
"""


class ReleaseError(ValueError):
    """The release, the reference list or the destination cannot be used. Carries fix and code."""

    def __init__(self, reason, fix="", code=""):
        super().__init__(reason)
        self.reason, self.fix, self.code = reason, fix, code


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def open_release(path, member=None):
    """(text stream, member name). A zip is read in place, never extracted; a text file as is."""
    path = Path(path)
    if not path.is_file():
        raise ReleaseError(f"no release at {path}", fix="pass the CellMarker 3.0 file you fetched",
                           code="release_missing")
    if zipfile.is_zipfile(path):
        zf = zipfile.ZipFile(path)
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
        if member is None:
            texts = [n for n in names if n.lower().endswith((".txt", ".tsv"))]
            member = MEMBER if MEMBER in names else (texts[0] if len(texts) == 1 else None)
        if member not in names:
            raise ReleaseError(f"{path.name} holds {', '.join(names[:6]) or 'nothing'}; which member "
                               f"is the release?", fix="name it with --member", code="release_member")
        return io.TextIOWrapper(zf.open(member), encoding="utf-8", errors="replace"), member
    return path.open("r", encoding="utf-8", errors="replace"), path.name


def reference_genes(paths):
    """(set of names or None, record). Text files hold one name per line; `#` starts a comment.
    An .h5ad gives its var_names. Several inputs must describe ONE gene axis, as the ingest this
    was ported from required: `in_reference` means 'measurable in this data', and two axes would
    make it mean two things."""
    if not paths:
        return None, {"reference_genes": "none"}
    sets = []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            raise ReleaseError(f"no reference gene list at {p}", code="reference_missing",
                               fix="pass a text file of names, one per line, or an .h5ad")
        if p.suffix == ".h5ad":
            import anndata as ad
            A = ad.read_h5ad(p, backed="r")
            names = {str(v) for v in A.var_names}
            A.file.close()
        else:
            names = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#")}
        sets.append((p.name, names))
    first = sets[0][1]
    if any(s != first for _n, s in sets[1:]):
        raise ReleaseError("the reference inputs describe different gene axes: "
                           + ", ".join(f"{n} {len(s):,}" for n, s in sets)
                           + ". in_reference would not be well defined", code="reference_axes",
                           fix="pass the gene axis of the data the corpus will be used on, once")
    digest = hashlib.sha256("\n".join(sorted(first)).encode("utf-8")).hexdigest()
    return first, {"reference_genes": str(len(first)), "reference_sha256": digest,
                   "reference_sources": ", ".join(n for n, _ in sets)}


def _ingest(con, fh, universe, log):
    keys = {d: {} for d in DIMS}
    rows, n_rows, bad_width, empty_field = [], 0, 0, Counter()
    cols = fh.readline().rstrip("\n").rstrip("\r").split("\t")
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ReleaseError(f"the release has no column(s) {', '.join(missing)}; its columns are "
                           f"{', '.join(cols[:24])}", code="release_columns",
                           fix="pass a CellMarker 3.0 release (the all_cell_marker layout)")
    col = {c: i for i, c in enumerate(cols)}

    def dim_id(name, values):
        table = keys[name]
        got = table.get(values)
        if got is None:
            got = len(table) + 1
            table[values] = got
        return got

    for line in fh:
        rec = line.rstrip("\n").rstrip("\r").split("\t")
        if len(rec) != len(cols):
            bad_width += 1
            continue
        n_rows += 1
        for c in ("symbol", "cellontology_id", "pmid", "uberon_id", "series_id"):
            if c in col and not rec[col[c]]:
                empty_field[c] += 1
        rows.append((n_rows, rec[col["species"]],
                     dim_id("tissue", tuple(rec[col[k]] for k in DIMS["tissue"])),
                     dim_id("celltype", tuple(rec[col[k]] for k in DIMS["celltype"])),
                     dim_id("gene", tuple(rec[col[k]] for k in DIMS["gene"])),
                     dim_id("disease", (rec[col["disease"]],)),
                     dim_id("source", tuple(rec[col[k]] for k in DIMS["source"])),
                     rec[col["marker"]]))
        if len(rows) >= 250_000:
            con.executemany("INSERT INTO marker VALUES (?,?,?,?,?,?,?,?)", rows)
            rows.clear()
            log(f"    {n_rows:,} rows")
    if rows:
        con.executemany("INSERT INTO marker VALUES (?,?,?,?,?,?,?,?)", rows)
    if not n_rows:
        raise ReleaseError("the release holds no row of the declared width", code="release_empty",
                           fix="check the file is the release and not a fragment of it")

    universe_ci = {s.upper() for s in universe} if universe is not None else None
    for name, fields in DIMS.items():
        payload = [(i,) + k for k, i in keys[name].items()]
        if name == "gene":
            # Three facts, kept apart: can the gene be measured here (case-insensitively), is it
            # spelled the same, and the key that stops one gene fragmenting into two. Without a
            # reference list the first two are unknown, and NULL says so.
            si = 1 + fields.index("symbol")
            if universe is None:
                payload = [p + (None, p[si].upper(), None) for p in payload]
            else:
                payload = [p + (1 if (p[si] in universe or p[si].upper() in universe_ci) else 0,
                                p[si].upper(),
                                1 if p[si] in universe else 0)
                           for p in payload]
        elif name == "tissue":
            ti = 1 + fields.index("tissue_type")
            payload = [p + (1 if ("fetal" in p[ti].lower() or "embry" in p[ti].lower()) else 0,)
                       for p in payload]
        elif name == "disease":
            payload = [p + (1 if p[1] == "Normal" else 0,) for p in payload]
        con.executemany(f"INSERT INTO {name} VALUES ({','.join(['?'] * len(payload[0]))})",
                        payload)
    con.commit()
    return n_rows, bad_width, empty_field, cols


def _specificity(con, log):
    """1 - normalised entropy of a gene's claims across cell types within (species, tissue).

    Two variants: `specificity` over every row, and `specificity_pub` over publication-backed rows
    only (Experiment, Review, Single-cell sequencing), where a detector's ambient tail is absent.
    Neither collapses cell-name synonyms, and neither may rank or filter on its own: it is entropy
    over CLAIMS, so a gene named once scores 1.0. Rows are updated by rowid - a four-column WHERE
    reads as precise and is quadratic.
    """
    con.execute("CREATE INDEX IF NOT EXISTS ix_a_sts ON assertion(species, tissue_class, symbol_norm)")
    con.commit()
    cur = con.execute("""
        SELECT rowid, species, tissue_class, symbol_norm, n_rows,
               n_experiment + n_review + n_singlecell
        FROM assertion ORDER BY species, tissue_class, symbol_norm""")
    upd = con.cursor()
    pending, scored, key, buf = [], 0, None, []

    def entropy_score(weights):
        live = [w for w in weights if w > 0]
        n_ct, total = len(live), sum(live)
        if n_ct <= 1 or total <= 0:
            return (1.0 if n_ct == 1 else 0.0), 1.0, n_ct
        h = 0.0
        for w in live:
            q = w / total
            h -= q * math.log(q)
        return 1.0 - h / math.log(n_ct), max(live) / total, n_ct

    def flush(group):
        nonlocal scored
        if not group:
            return
        spec, share, n_ct = entropy_score([n for _, n, _ in group])
        spec_p, _, n_ct_p = entropy_score([c for _, _, c in group])
        for rid, _, _ in group:
            pending.append((n_ct, round(spec, 6), round(share, 6), round(spec_p, 6), n_ct_p, rid))
        scored += len(group)

    sql = ("UPDATE assertion SET n_celltypes_same_tissue = ?, specificity = ?, "
           "top_celltype_share = ?, specificity_pub = ?, n_celltypes_pub = ? WHERE rowid = ?")
    for rid, sp, tc, sym, n, ncur in cur:
        k = (sp, tc, sym)
        if k != key:
            flush(buf)
            key, buf = k, []
        buf.append((rid, n, ncur))
        if len(pending) >= 200_000:
            upd.executemany(sql, pending)
            pending.clear()
            log(f"    {scored:,} scored")
    flush(buf)
    if pending:
        upd.executemany(sql, pending)
    con.commit()
    return scored


def _unify_display(con):
    """One spelling per (species, gene), chosen once: the reference's spelling, then the most rows."""
    con.execute("CREATE INDEX IF NOT EXISTS ix_a_sn ON assertion(species, symbol_norm)")
    best = {}
    for sp, sn, disp, rows, exact in con.execute("""
            SELECT species, symbol_norm, symbol_display, SUM(n_rows), MAX(exact_reference)
            FROM assertion GROUP BY species, symbol_norm, symbol_display"""):
        rank = (exact or 0, rows or 0)
        if best.get((sp, sn), ((-1, -1), None))[0] < rank:
            best[(sp, sn)] = (rank, disp)
    con.executemany("UPDATE assertion SET symbol_display = ? WHERE species = ? AND symbol_norm = ?",
                    [(d, sp, sn) for (sp, sn), (_, d) in best.items()])
    con.commit()
    return len(best)


def _ortholog(con):
    """Mouse-human symbol bridge by case-insensitive match, LABELLED as the heuristic it is."""
    from .species_check import normalise
    con.execute("CREATE TABLE ortholog (symbol_mouse TEXT, symbol_human TEXT, match_method TEXT)")
    spellings = [r[0] for r in con.execute("SELECT DISTINCT species FROM assertion")]

    def symbols(which):
        names = [s for s in spellings if normalise(s) == which]
        if not names:
            return set()
        q = ",".join("?" * len(names))
        return {r[0] for r in con.execute(
            f"SELECT DISTINCT symbol_display FROM assertion WHERE species IN ({q}) "
            f"AND symbol_norm <> ''", names)}

    mouse, human = symbols("mouse"), symbols("human")
    hu_by_upper = {}
    for s in sorted(human):
        hu_by_upper.setdefault(s.upper(), s)
    pairs = sorted((m, hu_by_upper[m.upper()], "symbol_case") for m in mouse
                   if m.upper() in hu_by_upper)
    con.executemany("INSERT INTO ortholog VALUES (?,?,?)", pairs)
    con.commit()
    return len(pairs)


def build(release, out, *, reference=None, member=None, version="", log=print) -> dict:
    """Build the corpus at `out` from `release`. Returns what `corpus_info` records.

    Written to `<out>.partial` and renamed when complete, so a corpus that exists was finished. An
    existing `out` is refused: a run may have read it, and replacing it under that run's record
    would make the record describe a different corpus.
    """
    out = Path(out)
    if out.exists():
        raise ReleaseError(f"{out} exists. A corpus a run may have read is not replaced",
                           fix="choose another --out, or move the old corpus aside yourself",
                           code="out_exists")
    universe, ref_info = reference_genes(reference)
    log(f"release {Path(release).name}: hashing (read only; the file is not modified)")
    digest = sha256(release)
    log(f"  sha256 {digest}")
    fh, member = open_release(release, member)
    tmp = out.with_name(out.name + ".partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        log(f"ingesting {member} - every row kept")
        n_rows, bad_width, empty_field, cols = _ingest(con, fh, universe, log)
        log(f"  {n_rows:,} rows written, {bad_width} malformed skipped")
        con.executescript(INDICES)
        con.executescript(VIEW_MARKER)
        con.commit()
        log("assertions: one row per species x tissue x cell name x UPPER(symbol)")
        con.executescript(ASSERTION)
        con.executescript(TIER)
        con.commit()
        n_assert = con.execute("SELECT COUNT(*) FROM assertion").fetchone()[0]
        log(f"  {n_assert:,} assertions; specificity")
        _specificity(con, log)
        n_disp = _unify_display(con)
        n_orth = _ortholog(con)
        log(f"  {n_disp:,} species-gene spellings unified; {n_orth:,} mouse-human pairs by symbol case")
        con.executescript(INDICES2)
        n_sym = con.execute("SELECT COUNT(*) FROM gene").fetchone()[0]
        n_in = con.execute("SELECT SUM(in_reference) FROM gene").fetchone()[0]
        info = {"format": "scanno-corpus/1", "built_by": f"scanno {version}".strip(),
                "source_file": Path(release).name, "source_sha256": digest,
                "source_member": member, "source_columns": "|".join(cols),
                "rows_in_file": str(n_rows + bad_width), "rows_malformed_skipped": str(bad_width),
                "rows_written": str(n_rows), "rows_removed": "0", "assertions": str(n_assert),
                "distinct_symbols": str(n_sym),
                "distinct_symbols_in_reference": "" if n_in is None else str(n_in),
                "ortholog_pairs_symbol_case": str(n_orth),
                "tiers": "; ".join(TIER_LABELS[t] for t in sorted(TIER_LABELS)),
                **ref_info,
                **{f"empty_field_{k}": str(v) for k, v in sorted(empty_field.items())}}
        con.execute("CREATE TABLE corpus_info (key TEXT PRIMARY KEY, value TEXT)")
        con.executemany("INSERT INTO corpus_info VALUES (?,?)", sorted(info.items()))
        con.commit()
        con.execute("VACUUM")
    except BaseException:
        con.close()
        if tmp.exists():
            tmp.unlink()
        raise
    finally:
        fh.close()
    con.close()
    tmp.rename(out)
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return info
