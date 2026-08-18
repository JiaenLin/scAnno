"""The reading playbook: the notebook is generated, every code cell parses, and the warnings
that make it safe are actually in it.

WHY A TEST FOR A DOCUMENT

Two copies of the same instructions drift, and the drift is invisible - the notebook still runs
and still makes figures while teaching a parameter the markdown no longer recommends. So the
notebook is GENERATED and this asserts it is current.

The content checks are not style policing. Each one pins a sentence that stops a reader drawing a
conclusion the data cannot support: that a recomputed clustering corresponds to the delivered
labels, that a sentinel is a cell type, that mitochondrial percent means the same thing on nuclei
as on cells, or that a composition percentage can be read without its denominator.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "READING_THE_OUTPUT.md"
NB = ROOT / "docs" / "reading_the_output.ipynb"
GEN = ROOT / "docs" / "make_notebook.py"

fails = []


def check(n, c, d=""):
    print(f"  {'ok  ' if c else 'FAIL'}  {n}" + (f"   {d}" if d else ""))
    if not c:
        fails.append(n)


print("A. the files exist and the notebook is GENERATED, not maintained")
check("the playbook exists", MD.exists())
check("the notebook exists", NB.exists())
check("the generator exists", GEN.exists())
r = subprocess.run([sys.executable, str(GEN), "--check"], capture_output=True, text=True)
check("the notebook is IN SYNC with the markdown", r.returncode == 0,
      (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "")

nb = json.loads(NB.read_text(encoding="utf-8"))
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
check("the notebook says it is generated",
      "GENERATED FILE" in json.dumps(nb["metadata"]))

print("\nB. every code cell is valid python")
bad = []
for i, c in enumerate(code):
    try:
        ast.parse("".join(c["source"]))
    except SyntaxError as e:
        bad.append(f"cell {i}: {e}")
check(f"all {len(code)} code cells parse", not bad, "; ".join(bad[:3]))
check("there are enough of them to be a playbook", len(code) >= 25, str(len(code)))

md = MD.read_text(encoding="utf-8")

print("\nC. the blocks the playbook promises are all present")
for block in ("Block 1", "TRACK A", "TRACK B", "PLOT BLOCKS"):
    check(f"{block} is present", block in md)
for step in ("normalize_total", "log1p", "highly_variable_genes", "sc.tl.pca",
             "sc.pp.neighbors", "sc.tl.leiden", "sc.tl.umap",
             "calculate_qc_metrics", "rank_genes_groups"):
    check(f"the pipeline covers {step}", step in md)
for plot in ("sc.pl.umap", "sc.pl.dotplot", "sc.pl.violin", "sc.pl.stacked_violin",
             "sc.pl.matrixplot", "sc.pl.heatmap", "sc.pl.tracksplot", "kind=\"bar\""):
    check(f"the plot block covers {plot}", plot in md)

print("\nD. the parameters a user has to change are NAMED, not buried")
for p in ("MIN_UMI", "MAX_UMI", "MIN_GENES", "MAX_MT_PCT", "MIN_CELLS",
          "TARGET_SUM", "N_HVG", "N_PCS", "N_NEIGHBORS", "RESOLUTION", "MIN_DIST", "SEED"):
    check(f"{p} is a named constant", f"{p}" in md)
check("the label/batch keys are set in ONE place", "LABEL       =" in md and "BATCH       =" in md)

print("\nE. the warnings that stop a wrong conclusion")
check("it says a recomputed clustering does NOT match the delivered labels",
      "no relationship to them" in md or "will NOT correspond" in md)
check("it says sentinels are not cell types",
      "not cell types" in md.lower() or "NOT cell types" in md)
check("it says mitochondrial percent means something else on NUCLEI",
      "carry-over" in md and "nucleus contains no mitochondria" in md)
check("it warns that re-filtering is a SECOND filter, not a first",
      "already filtered" in md.lower())
check("it makes the user look at what a filter removes, per design arm",
      "apparent biological difference" in md)
check("it warns that composition is compositional",
      "compositional" in md and "denominator" in md)
check("it warns that cluster markers are not condition DE",
      "pseudoreplication" in md)
check("it insists shared axis limits when splitting a UMAP by group",
      "SAME limits" in md or "Share the limits" in md)

print("\nE2. the DEG and enrichment blocks, and the API facts that were MEASURED not recalled")
for step in ("dc.pp.pseudobulk", "dc.pp.filter_by_expr", "DeseqDataSet", "DeseqStats",
             "results_df", "gp.prerank", "gp.enrichr", "gp.dotplot", "gp.gseaplot",
             "dc.mt.ulm", "dc.mt.mlm", "dc.mt.ora", "dc.mt.decouple", "dc.pp.get_obsm",
             "dc.tl.rankby_group", "dc.pl.barplot", "dc.pl.dotplot", "dc.op.collectri",
             "dc.op.progeny", "dc.op.hallmark", "dc.pp.read_gmt", "dc.op.translate"):
    check(f"covers {step}", step in md)
check("the DESeq2 result columns are named exactly as the library returns them",
      "`baseMean`, `log2FoldChange`, `lfcSE`, `stat`, `pvalue`, `padj`" in md)
check("THE decoupler footgun: AnnData writes in place and returns None",
      "returns `None` and" in md and "in place" in md)
check("...and DataFrame returns a tuple instead",
      "returns an `(es, pv)` tuple" in md)
check("it says the input must be normalised, not raw counts",
      "NOT raw counts" in md)
check("it says which nets are weighted, because that decides the method",
      "Weighted or not decides the method" in md)
check("it warns tmin silently drops sources", "silently drops sources" in md)
check("it explains the no-overlap assertion is about GENE NAMES, not tmin",
      "That message means gene names, not `tmin`" in md)
check("it insists on a real ORA background", "background decides the p-value" in md.lower()
      or "background matters more" in md)
check("it separates cluster markers from condition DE by the UNIT of replication",
      "the **cell**" in md and "the **sample**" in md)
check("it makes the reader check the per-arm sample table before believing a DE result",
      "is not a comparison" in md)
check("it drops thin pseudobulk profiles rather than averaging them in",
      "psbulk_cells" in md)
check("a skipped cell type in the DE loop is NAMED", "NAMED, not silently dropped" in md)
check("it says activity is inferred, not measured", "not measured, it is inferred" in md)
check("it gives an offline path for prior knowledge", "read_gmt" in md and "Offline" in md)

print("\nE3. the UMAP sweep")
check("it sweeps both parameters", "MIN_DISTS" in md and "N_NEIGHBORS = [" in md)
check("it is a 5x5 grid = 25 layouts", md.count("0.05, 0.1, 0.3, 0.5, 0.8") == 1
      and "5, 15, 30, 50, 100" in md and "25 layouts" in md)
check("neighbours are computed ONCE per row, not 25 times",
      "once per row" in md)
check("every layout is kept, so picking one is a rename not a re-run",
      "X_umap_nn" in md and "keep every layout" in md)
check("the choice is RECORDED or it is unreproducible", "umap_choice" in md)
check("it says a prettier UMAP is not a better result",
      "not a better result" in md)
check("it says a claim should survive all 25 panels", "survive in all 25" in md)
check("it warns this one is worth batching", "batch job" in md)

print("\nF. it tells the reader to ask the object rather than assume")
check("it prints the obs columns before using them", "for c in A.obs.columns" in md)
check("it reads the embedding provenance from uns", "scanno_embed" in md)
check("it verifies the counts layer is integral rather than trusting the name",
      "np.rint" in md)
check("names are declared per-project, not hardcoded as truth",
      "<- your file" in md or "EDIT THESE" in md)

print("\nG. no cohort is named — this ships in a public repo")
for leak in ("sambo", "aging_hfd", "young_hfd", "100,713", "109,140"):
    check(f"no {leak!r}", leak not in md.lower())

print("")
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}")
    raise SystemExit(1)
print("playbook OK - generated, parses, and carries the warnings that make it safe to follow")
