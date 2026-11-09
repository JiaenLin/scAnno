"""Writes tests/fixtures/mini_cellmarker.txt: an INVENTED release in CellMarker 3.0's 22-column layout.

Every row is invented. The genes are the textbook markers tests/_cohort.py uses; the PMIDs, titles,
journals and years are made up. Each case `scanno build-markers` must get right occurs on purpose,
and tests/test_build_markers.py checks the committed file is exactly what this writes:

    python tests/fixtures/make_mini_cellmarker.py tests/fixtures/mini_cellmarker.txt
"""
import sys
from pathlib import Path

COLS = ["species", "tissue_class", "tissue_type", "uberon_id", "disease", "cell_name_class",
        "cell_name", "cellontology_id", "marker", "symbol", "gene_id", "gene_type", "gene_name",
        "uniprot_id", "technology_seq", "marker_source", "pmid", "title", "journal", "year",
        "series_id", "method_details"]
PANELS = {  # cell name -> (class, CL id, markers), the five of tests/_cohort.py
    "T cell": ("Lymphocyte", "CL_0000084", ["CD3D", "CD3E", "CD3G", "CD2", "CD5", "CD7", "IL7R", "LCK", "TRAC", "CD28", "CD6", "LAT"]),
    "B cell": ("Lymphocyte", "CL_0000236", ["CD79A", "CD79B", "MS4A1", "CD19", "CD22", "PAX5", "BANK1", "CD72", "FCRL1", "VPREB3", "BLK", "TCL1A"]),
    "Natural killer cell": ("Lymphocyte", "CL_0000623", ["NKG7", "GNLY", "KLRD1", "KLRF1", "NCAM1", "PRF1", "GZMB", "KLRC1", "FCGR3A", "SH2D1B", "NCR1", "KIR2DL4"]),
    "Monocyte": ("Myeloid cell", "CL_0000576", ["CD14", "LYZ", "S100A8", "S100A9", "FCN1", "VCAN", "CSF1R", "CD68", "ITGAM", "MNDA", "CTSS", "FCGR1A"]),
    "Dendritic cell": ("Myeloid cell", "CL_0000451", ["FCER1A", "CD1C", "CLEC10A", "CLEC9A", "XCR1", "IRF8", "ITGAX", "LAMP3", "CCR7", "HLA-DQA1", "CD1E", "FLT3"]),
}
pm = iter(range(90000001, 90009999))


def row(species, tissue, ttype, uberon, cls, name, cl, sym, source, pmid, method="", disease="Normal",
        tech="scRNA-seq", series=""):
    return [species, tissue, ttype, uberon, disease, cls, name, cl, sym, sym, "", "protein_coding",
            f"{sym} (invented name)", "", tech, source, pmid, f"Invented study {pmid}",
            "Invented journal", "2020.0", series, method]


rows = []
# HUMAN BLOOD. Per cell type: markers 0-3 E1 (two Experiment PMIDs), 4-6 E2 (one Review), 7-8 E3
# (two single-cell PMIDs), 9-10 E4 (one single-cell PMID), 11 E5 (Method only).
for name, (cls, cl, genes) in PANELS.items():
    for i, g in enumerate(genes):
        if i <= 3:
            rows += [row("Human", "Blood", "Peripheral blood", "UBERON_0000178", cls, name, cl, g, "Experiment", str(next(pm))) for _ in range(2)]
        elif i <= 6:
            rows.append(row("Human", "Blood", "Peripheral blood", "UBERON_0000178", cls, name, cl, g, "Review", str(next(pm))))
        elif i <= 8:
            rows += [row("Human", "Blood", "Peripheral blood", "UBERON_0000178", cls, name, cl, g, "Single-cell sequencing", str(next(pm))) for _ in range(2)]
        elif i <= 10:
            rows.append(row("Human", "Blood", "Peripheral blood", "UBERON_0000178", cls, name, cl, g, "Single-cell sequencing", str(next(pm))))
        else:
            rows.append(row("Human", "Blood", "Peripheral blood", "UBERON_0000178", cls, name, cl, g, "Method", str(next(pm)), method="Invented detector"))
# a case variant: one gene written twice, one key (UPPER(symbol)), n_case_variants 2
rows.append(row("Human", "Blood", "Peripheral blood", "UBERON_0000178", "Lymphocyte", "T cell", "CL_0000084", "Cd3d", "Experiment", str(next(pm))))
# a gene claimed by two cell types (specificity below 1), by the detector only
rows.append(row("Human", "Blood", "Peripheral blood", "UBERON_0000178", "Lymphocyte", "B cell", "CL_0000236", "LCK", "Method", str(next(pm)), method="Invented detector"))
# a fetal tissue type and a disease state, kept and flagged rather than dropped
rows.append(row("Human", "Blood", "Fetal blood", "UBERON_0000178", "Myeloid cell", "Monocyte", "CL_0000576", "CD14", "Experiment", str(next(pm)), disease="Invented disease"))
# a row with no symbol: written to `marker`, keyed '' in `assertion`, never read as a marker
rows.append(row("Human", "Blood", "Peripheral blood", "UBERON_0000178", "Lymphocyte", "T cell", "CL_0000084", "", "Method", "", method="Invented detector"))
# HUMAN BRAIN: a second tissue, so a miss can list the tissues the species has
for name, cls, cl, genes in (("Neuron", "Neuron", "CL_0000540", ["RBFOX3", "SNAP25", "SYT1"]),
                             ("Astrocyte", "Glial cell", "CL_0000127", ["GFAP", "AQP4", "SLC1A3"])):
    for g in genes:
        rows.append(row("Human", "Brain", "Brain", "UBERON_0000955", cls, name, cl, g, "Experiment", str(next(pm))))
# MOUSE BLOOD: Title-case symbols, the other half of the case-match ortholog bridge
for g in ("Cd3d", "Cd3e", "Cd19", "Cd14", "Lyz2"):
    rows.append(row("Mouse", "Blood", "Peripheral blood", "UBERON_0000178", "Lymphocyte" if g.startswith("Cd3") or g == "Cd19" else "Myeloid cell",
                    "T cell" if g.startswith("Cd3") else ("B cell" if g == "Cd19" else "Monocyte"),
                    "CL_0000084", g, "Experiment", str(next(pm))))
out = Path(sys.argv[1])
lines = ["\t".join(COLS)] + ["\t".join(r) for r in rows]
lines.insert(7, "Human\tBlood\tPeripheral blood\tthis row is malformed\tit has too few fields")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out}: {len(rows)} rows + 1 malformed")
