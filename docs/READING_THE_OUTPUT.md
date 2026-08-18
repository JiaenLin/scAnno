# Reading a scAnno joint object in Python

A playbook for the single object `scanno embed` writes. Copy the blocks in order, or jump to the
one you need — every block is self-contained after **Block 1** and **Block 2**.

There are **two tracks**, and choosing between them is the only decision you have to make up front:

| | **Track A — use what ships** | **Track B — recompute** |
|---|---|---|
| speed | seconds | minutes to hours |
| embedding | the delivered one | yours |
| clustering | the delivered sweep | yours |
| matches the report | **yes** | no |
| matches the labels | **yes** | **no — see the warning in B0** |
| customizable | no | completely |

**Start with Track A.** Go to Track B only when you specifically need different parameters, and
read B0 first.

---

## Block 0 — install

```python
# minimum
# pip install scanpy anndata matplotlib
# for leiden clustering in Track B
# pip install igraph leidenalg
# for dotplot/heatmap niceties
# pip install seaborn

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sc.settings.verbosity = 1                  # 0 quiet · 1 info · 3 debug
sc.settings.set_figure_params(dpi=100, dpi_save=300, frameon=False, figsize=(5, 5))
```

---

## Block 1 — read, and ASK THE OBJECT WHAT IT HAS

Never assume the column names. They are chosen at `scanno embed --label-obs` time and differ
between projects. Print them, then set the variables once.

```python
PATH = "cohort_joint_embedding.h5ad"       # <- your file

A = sc.read_h5ad(PATH)
print(A)
```

```python
# --- what is actually in here ---
print(f"{A.n_obs:,} cells x {A.n_vars:,} genes\n")

print("obs columns:")
for c in A.obs.columns:
    n = A.obs[c].nunique()
    kind = str(A.obs[c].dtype)
    ex = A.obs[c].iloc[0]
    print(f"   {c:<28} {kind:<10} {n:>6} distinct   e.g. {ex}")

print("\nlayers :", list(A.layers))
print("obsm   :", list(A.obsm))
print("var    :", list(A.var.columns))
print("\nuns['scanno_embed'] — how the delivered embedding was made:")
for k, v in dict(A.uns.get("scanno_embed", {})).items():
    print(f"   {k}: {v}")
```

`uns['scanno_embed']` records `n_hvg`, `n_pcs`, `n_neighbors`, `min_dist`, `seed`, whether the
object is `integrated`, and `label_map` — which column is a rename of which.

### Set your variables once

```python
# EDIT THESE to match what Block 1 printed.
LABEL       = "cell_type"          # the fine label (often a path like Immune/Myeloid/Macrophage)
COMPARTMENT = "cell_compartment"   # the coarse label, if your run wrote one
BATCH       = "sample"             # the library / batch key
COUNTS      = "counts"             # layer holding RAW INTEGER counts

# Optional: a forced-resolution label with no UNRESOLVED cells, if `--resolve` was used.
FORCED      = "cell_type_forced" if "cell_type_forced" in A.obs else None

# Any experimental factors your object carries. Leave empty if it has none.
FACTORS     = [c for c in ("group", "age", "diet", "batch", "chemistry") if c in A.obs]
print("factors found:", FACTORS)
```

### Two label values that are NOT cell types

```python
SENTINELS = ["EXCLUDED", "UNRESOLVED"]

# EXCLUDED   — withheld upstream, never annotated.
# UNRESOLVED — the walk declined to call it rather than guessing.
#
# They are statements about the ANNOTATION, not populations in the tissue. Every percentage,
# every marker test and every composition figure will count them as cell types unless you say
# otherwise. Decide once, here, and be explicit in your figure captions.
counts = A.obs[LABEL].value_counts()
print(counts[counts.index.isin(SENTINELS)])
print(f"\n{counts[counts.index.isin(SENTINELS)].sum():,} of {A.n_obs:,} cells "
      f"({100 * counts[counts.index.isin(SENTINELS)].sum() / A.n_obs:.2f}%)")

# a mask you can reuse everywhere
real = ~A.obs[LABEL].isin(SENTINELS)
```

### What the matrices hold

```python
# X and layers['lognorm'] are the SAME values: log1p of library-size-normalised counts.
# layers['counts'] is raw integers. Check rather than trust:
X = A.layers[COUNTS]
d = X.data if hasattr(X, "data") and not isinstance(X, np.ndarray) else np.asarray(X).ravel()
print(f"{COUNTS}: max {d.max():.0f}, integral {np.all(d == np.rint(d))}")
print(f"X      : max {A.X.max():.3f}  <- log scale, so ~5-8 is normal")
```

---

# TRACK A — use what ships

Everything here is instant. The embedding, the clustering and the labels all came out of the same
run, so they agree with each other and with the report.

## Block A1 — plot immediately

```python
sc.pl.umap(A, color=LABEL, legend_loc="on data", legend_fontsize=6,
           title="delivered labels", frameon=False)
```

If `sc.pl.umap` complains there is no `X_umap`, your object stores it under another name:

```python
print([k for k in A.obsm if "umap" in k.lower()])
A.obsm["X_umap"] = A.obsm["X_umap_scanno"]      # example — use what was printed
```

## Block A2 — the delivered clustering

```python
leiden_cols = sorted(c for c in A.obs.columns if c.startswith("leiden"))
print("delivered clusterings:", leiden_cols)

sc.pl.umap(A, color=leiden_cols[:4], legend_loc="on data", legend_fontsize=5, ncols=2)
```

**Skip to the PLOT BLOCKS.** Track B is only needed if you want different parameters.

---

# TRACK B — recompute from raw counts

## Block B0 — READ THIS FIRST

```text
Anything you recompute here REPLACES what shipped, in memory.

  * your UMAP will not match the figures in the report
  * your leiden clusters will NOT correspond to `cell_type` — the labels were assigned to the
    DELIVERED clustering, so a new clustering has no relationship to them
  * neither UMAP nor leiden is reproducible across package versions, so even the same parameters
    on a different machine give a different picture

Work on a COPY, so you can always get back:
```

```python
B = A.copy()          # recompute on B; A stays as delivered
```

## Block B1 — QC metrics and filtering

**The object you were given is already filtered.** It has been through an upstream QC stage.
Filtering again is legitimate — you may want a stricter cut — but it is a *second* filter on top
of one already applied, not a first one. Compute the metrics and **look at them before cutting
anything**.

```python
# mitochondrial gene prefix: "MT-" human, "mt-" mouse. Check yours:
print([g for g in B.var_names if g.lower().startswith("mt-")][:15])

B.var["mt"] = B.var_names.str.lower().str.startswith("mt-")
B.var["ribo"] = B.var_names.str.lower().str.match(r"^rp[sl]")
B.var["hb"] = B.var_names.str.lower().str.match(r"^hb[ab]?[-_]?")

sc.pp.calculate_qc_metrics(B, qc_vars=["mt", "ribo", "hb"], layer=COUNTS,
                           percent_top=None, log1p=False, inplace=True)
print(B.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe())
```

| metric | what it is |
|---|---|
| `total_counts` | UMIs per cell |
| `n_genes_by_counts` | genes detected per cell |
| `pct_counts_mt` | % of UMIs from mitochondrial genes |
| `pct_counts_ribo` | % from ribosomal protein genes |
| `pct_counts_hb` | % from haemoglobin — high means blood contamination |

### Look before you cut

```python
fig, ax = plt.subplots(1, 4, figsize=(18, 3.5))
sc.pl.violin(B, ["total_counts"], jitter=0.4, ax=ax[0], show=False, log=True)
sc.pl.violin(B, ["n_genes_by_counts"], jitter=0.4, ax=ax[1], show=False, log=True)
sc.pl.violin(B, ["pct_counts_mt"], jitter=0.4, ax=ax[2], show=False)
sc.pl.scatter(B, x="total_counts", y="n_genes_by_counts", color="pct_counts_mt", ax=ax[3],
              show=False)
plt.tight_layout(); plt.show()
```

Per batch, which is where a bad library shows itself:

```python
sc.pl.violin(B, ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
             groupby=BATCH, rotation=90, jitter=0.4, multi_panel=True, log=False)
```

### The filter

```python
MIN_UMI    = 500      # cells below this many UMIs are dropped
MAX_UMI    = None     # e.g. 50000 to cut likely doublets; None = no upper bound
MIN_GENES  = 250      # cells detecting fewer genes than this are dropped
MAX_MT_PCT = 5.0      # cells above this % mitochondrial are dropped
MIN_CELLS  = 3        # genes seen in fewer cells than this are dropped

before = B.n_obs
keep = (
    (B.obs["total_counts"] >= MIN_UMI)
    & (B.obs["n_genes_by_counts"] >= MIN_GENES)
    & (B.obs["pct_counts_mt"] <= MAX_MT_PCT)
)
if MAX_UMI is not None:
    keep &= B.obs["total_counts"] <= MAX_UMI

# WHAT ARE YOU ABOUT TO REMOVE, and is it even across your design? Print it, do not assume it.
print(f"removing {(~keep).sum():,} of {before:,} cells ({100 * (~keep).mean():.2f}%)\n")
print("by cell type:")
print(pd.crosstab(B.obs[LABEL], ~keep, normalize="index").mul(100).round(2)
        .rename(columns={True: "% removed"})[["% removed"]].sort_values("% removed",
                                                                       ascending=False).head(10))
for f in FACTORS:
    print(f"\nby {f}:")
    print(pd.crosstab(B.obs[f], ~keep, normalize="index").mul(100).round(2))
```

> **A filter that removes 20% of one arm and 2% of another has turned a technical property into
> an apparent biological difference**, and nothing downstream can undo it. If the table above is
> uneven, that is a finding about your filter, not a detail.

```python
B = B[keep].copy()
sc.pp.filter_genes(B, min_cells=MIN_CELLS)
print(f"kept {B.n_obs:,} cells x {B.n_vars:,} genes")
```

### If these are NUCLEI, `pct_counts_mt` does not mean what it means for cells

A nucleus contains no mitochondria and the mitochondrial genome is transcribed in the
mitochondrial matrix, so mitochondrial reads in single-**nucleus** data measure **cytoplasmic
carry-over and ambient RNA**, not the cell's mitochondrial activity. Filtering on it is still a
reasonable way to drop dirty droplets. **Reporting it as biology is not**, and a high-`mt` group
is telling you that group dissociated dirtier.

`A.uns['scanno_embed']` and your upstream QC report say which assay this is.

## Block B2 — normalisation, HVG, scaling, PCA

```python
B.layers["counts_kept"] = B.layers[COUNTS].copy()     # keep raw counts recoverable

TARGET_SUM = 1e4        # counts per cell after normalisation. None = median library size
N_HVG      = 2000       # highly variable genes. 1000-5000 typical
N_PCS      = 50         # principal components
FLAVOR     = "seurat"   # "seurat" (log data) · "seurat_v3" (RAW counts) · "cell_ranger"
REGRESS    = []         # e.g. ["total_counts", "pct_counts_mt"] — slow, often unnecessary
SCALE_MAX  = 10         # clip scaled values at this many SDs

sc.pp.normalize_total(B, target_sum=TARGET_SUM)
sc.pp.log1p(B)
B.raw = B                               # keeps all genes for plotting after subsetting
```

```python
if FLAVOR == "seurat_v3":
    sc.pp.highly_variable_genes(B, n_top_genes=N_HVG, flavor="seurat_v3",
                                layer="counts_kept", batch_key=BATCH)
else:
    sc.pp.highly_variable_genes(B, n_top_genes=N_HVG, flavor=FLAVOR, batch_key=BATCH)

sc.pl.highly_variable_genes(B)
print(f"{B.var['highly_variable'].sum():,} HVGs")
```

> `batch_key` selects genes variable **within** batches, which stops a gene that is only variable
> *between* libraries from dominating. Drop it if you have one batch.

```python
B = B[:, B.var.highly_variable].copy()
if REGRESS:
    sc.pp.regress_out(B, REGRESS)
sc.pp.scale(B, max_value=SCALE_MAX)
sc.tl.pca(B, n_comps=N_PCS, svd_solver="arpack", random_state=0)
sc.pl.pca_variance_ratio(B, n_pcs=N_PCS, log=True)
```

The elbow in that plot is how many PCs carry signal. Use it to set `N_NEIGHBORS_PCS` below.

## Block B3 — neighbours, Leiden, UMAP

```python
N_NEIGHBORS      = 15     # 5-50. Higher = smoother, fewer small clusters
N_NEIGHBORS_PCS  = 30     # PCs to use — read off the elbow above
METRIC           = "euclidean"
RESOLUTION       = 1.0    # Leiden granularity. <1 fewer/larger, >1 more/smaller clusters
MIN_DIST         = 0.3    # UMAP compactness. 0.0-0.99; lower = tighter clumps
SPREAD           = 1.0    # UMAP scale
SEED             = 0

sc.pp.neighbors(B, n_neighbors=N_NEIGHBORS, n_pcs=N_NEIGHBORS_PCS,
                metric=METRIC, random_state=SEED)
sc.tl.leiden(B, resolution=RESOLUTION, key_added="my_leiden",
             flavor="igraph", n_iterations=2, directed=False, random_state=SEED)
sc.tl.umap(B, min_dist=MIN_DIST, spread=SPREAD, random_state=SEED)

print(f"{B.obs['my_leiden'].nunique()} clusters at resolution {RESOLUTION}")
sc.pl.umap(B, color=["my_leiden", LABEL], legend_loc="on data", legend_fontsize=5, ncols=2)
```

### Sweep the resolution instead of guessing

```python
for r in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0):
    sc.tl.leiden(B, resolution=r, key_added=f"leiden_{r}", flavor="igraph",
                 n_iterations=2, directed=False, random_state=SEED)
sc.pl.umap(B, color=[f"leiden_{r}" for r in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)],
           legend_loc="on data", legend_fontsize=4, ncols=4)
```

### How does your clustering relate to the delivered labels?

```python
ct = pd.crosstab(B.obs["my_leiden"], B.obs[LABEL], normalize="index").mul(100)
plt.figure(figsize=(10, max(4, 0.3 * len(ct))))
plt.imshow(ct.values, aspect="auto", cmap="Blues"); plt.colorbar(label="% of cluster")
plt.xticks(range(ct.shape[1]), ct.columns, rotation=90, fontsize=7)
plt.yticks(range(ct.shape[0]), ct.index, fontsize=7)
plt.xlabel("delivered label"); plt.ylabel("my_leiden"); plt.tight_layout(); plt.show()
```

A clean diagonal-ish block structure means your clustering recovered the same populations. It
does **not** mean your cluster numbers equal the labels.

---

# PLOT BLOCKS

These work on `A` (delivered) or `B` (recomputed) — set `D` to whichever you want, and `EMB` to
the embedding key.

```python
D   = A                     # or B
EMB = "X_umap"
SAVE = None                 # e.g. "_myfigure.png" — scanpy writes into ./figures/
```

## P1 — UMAP coloured by anything

```python
sc.pl.umap(D, color=[LABEL], legend_loc="on data", legend_fontsize=6, legend_fontoutline=2,
           size=3, alpha=0.9, frameon=False, title="cell type", save=SAVE)

sc.pl.umap(D, color=[BATCH] + FACTORS, ncols=2, size=3, frameon=False, wspace=0.3)
```

| parameter | effect |
|---|---|
| `color` | obs column(s) **or gene name(s)** — a list gives one panel each |
| `legend_loc` | `"right margin"` (default) · `"on data"` · `None` |
| `size` | dot size; lower for many cells (100k → 2-4) |
| `alpha` | transparency, for dense regions |
| `ncols` / `wspace` / `hspace` | panel grid |
| `vmin` / `vmax` | colour limits — `"p1"`/`"p99"` for percentiles |
| `cmap` | `"viridis"`, `"Reds"`, `"RdBu_r"` … |
| `groups` | show only these categories; the rest go grey |
| `palette` | `"tab20"`, or a dict `{category: colour}` |
| `sort_order` | `True` plots high values on top |

### One population highlighted against everything else

The figure no metric substitutes for.

```python
TARGET = D.obs[LABEL].value_counts().index[0]        # or any label / batch value
sc.pl.umap(D, color=LABEL, groups=[TARGET], size=4, frameon=False,
           title=f"{TARGET} against everything else", na_color="lightgrey")
```

### Split into one panel per group, at the same scale

```python
G = FACTORS[0] if FACTORS else BATCH
levels = list(D.obs[G].cat.categories) if hasattr(D.obs[G], "cat") else sorted(D.obs[G].unique())
fig, axes = plt.subplots(1, len(levels), figsize=(4.2 * len(levels), 4.2))
xy = D.obsm[EMB]
for ax, lv in zip(np.atleast_1d(axes), levels):
    m = (D.obs[G] == lv).values
    ax.scatter(xy[~m, 0], xy[~m, 1], s=1, c="#e0e0e0", linewidths=0, rasterized=True)
    ax.scatter(xy[m, 0], xy[m, 1], s=2, c="#c0504d", linewidths=0, rasterized=True)
    ax.set_title(f"{lv}  (n={m.sum():,})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())      # SAME limits on every panel
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
plt.tight_layout(); plt.show()
```

> Per-panel autoscaling makes a dispersed group look compact. Share the limits, always.

## P2 — feature plots: gene expression on the UMAP

```python
GENES = ["Actb", "Ttn", "Pecam1", "Ptprc"]           # <- your genes
GENES = [g for g in GENES if g in D.var_names or (D.raw and g in D.raw.var_names)]

sc.pl.umap(D, color=GENES, ncols=2, cmap="Reds", size=3, frameon=False,
           vmin=0, vmax="p99", sort_order=True, use_raw=D.raw is not None)
```

`vmax="p99"` clips the top 1%, so one very high cell cannot flatten the whole colour scale.

```python
# gene AND label side by side
sc.pl.umap(D, color=[LABEL] + GENES[:2], ncols=3, size=3, frameon=False)
```

## P3 — dotplot: the workhorse for marker panels

```python
MARKERS = {
    "Cardiomyocyte": ["Ttn", "Myh6", "Actc1"],
    "Endothelial":   ["Pecam1", "Cdh5", "Vwf"],
    "Fibroblast":    ["Dcn", "Col1a1", "Gsn"],
    "Immune":        ["Ptprc", "Cd68", "Csf1r"],
}
MARKERS = {k: [g for g in v if g in D.var_names] for k, v in MARKERS.items()}

sc.pl.dotplot(D, MARKERS, groupby=LABEL, standard_scale="var", dendrogram=True,
              cmap="Reds", dot_max=0.8, dot_min=0.0, colorbar_title="scaled mean expr",
              size_title="fraction of cells (%)", figsize=(12, 6))
```

| parameter | effect |
|---|---|
| `standard_scale` | `"var"` scales each gene 0-1 — **almost always what you want** |
| `dendrogram` | orders groups by similarity |
| `dot_max` / `dot_min` | clip the fraction-expressing dot size |
| `swap_axes` | genes on the y axis |
| `var_group_rotation` | angle of the gene-group labels |
| `mean_only_expressed` | mean over expressing cells only |

Sentinels are not cell types — exclude them from marker figures:

```python
sc.pl.dotplot(D[real], MARKERS, groupby=LABEL, standard_scale="var", dendrogram=True)
```

## P4 — violin, stacked violin, matrixplot, heatmap, tracksplot

```python
sc.pl.violin(D, GENES[:3], groupby=LABEL, rotation=90, stripplot=False, inner="box")
sc.pl.stacked_violin(D, MARKERS, groupby=LABEL, dendrogram=True, standard_scale="var",
                     figsize=(10, 6))
sc.pl.matrixplot(D, MARKERS, groupby=LABEL, standard_scale="var", cmap="RdBu_r",
                 dendrogram=True, colorbar_title="scaled mean")
sc.pl.heatmap(D, MARKERS, groupby=LABEL, standard_scale="var", cmap="viridis",
              show_gene_labels=True, dendrogram=True, figsize=(10, 8))
sc.pl.tracksplot(D, MARKERS, groupby=LABEL, dendrogram=True)
```

| plot | reach for it when |
|---|---|
| `dotplot` | you need mean **and** fraction expressing — the default choice |
| `stacked_violin` | you care about the distribution shape, not just the mean |
| `matrixplot` | mean only, most compact, best for many groups |
| `heatmap` | per-cell resolution, shows within-group heterogeneity |
| `tracksplot` | many genes, compact, good for a supplementary figure |
| `violin` | a handful of genes, full distribution |

## P5 — composition: cell-type percentages

The question most often asked of this object.

```python
GROUP = FACTORS[0] if FACTORS else BATCH

# per-group percentage table (excluding sentinels)
comp = (pd.crosstab(D.obs.loc[real.values, GROUP], D.obs.loc[real.values, LABEL],
                    normalize="index") * 100).round(2)
print(comp)                                # in Jupyter, just `comp` renders as a table
comp.to_csv("composition_percent.csv")
```

### Stacked bar — composition of each group

```python
fig, ax = plt.subplots(figsize=(1.6 * len(comp) + 3, 5))
comp.plot(kind="bar", stacked=True, ax=ax, width=0.8,
          colormap="tab20", edgecolor="white", linewidth=0.3)
ax.set_ylabel("% of cells"); ax.set_xlabel(GROUP); ax.set_ylim(0, 100)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, frameon=False)
plt.xticks(rotation=0); plt.tight_layout(); plt.show()
```

### Grouped bar — one cell type across groups

```python
sub = comp.T.sort_values(comp.index[0], ascending=False)
fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(sub)), 4.5))
sub.plot(kind="bar", ax=ax, width=0.8, colormap="Set2", edgecolor="none")
ax.set_ylabel("% of cells in group"); ax.set_xlabel("")
ax.legend(title=GROUP, fontsize=8, frameon=False)
plt.xticks(rotation=90, fontsize=8); plt.tight_layout(); plt.show()
```

### Per-sample points, which is what a reviewer will ask for

A bar of group means hides how many animals it rests on. Show the samples.

```python
per_sample = (pd.crosstab(D.obs.loc[real.values, BATCH], D.obs.loc[real.values, LABEL],
                          normalize="index") * 100)
key = D.obs.drop_duplicates(BATCH).set_index(BATCH)[GROUP] if GROUP != BATCH else None

TYPES = comp.mean().sort_values(ascending=False).head(8).index
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, ctype in zip(axes.ravel(), TYPES):
    vals, labs = [], []
    if key is None:                                   # no grouping factor: one box, all samples
        vals.append(per_sample[ctype].values); labs.append("all samples")
    else:
        for lv in key.unique():
            vals.append(per_sample.loc[key[key == lv].index, ctype].values)
            labs.append(str(lv))
    ax.boxplot(vals, showfliers=False, widths=0.6)
    ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs)   # `labels=` was renamed
    for i, v in enumerate(vals, start=1):
        ax.scatter(np.random.normal(i, 0.06, len(v)), v, s=18, zorder=3, color="#c0504d")
    ax.set_title(str(ctype)[:34], fontsize=9); ax.set_ylabel("% of sample")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
plt.tight_layout(); plt.show()
```

> **Before you read a composition difference as biology**: an upstream filter that removed cells
> unevenly across your design produces exactly this figure. Check your QC report for whether the
> exclusion was even, and say so in the caption either way.

### Absolute counts, not just percentages

```python
n = pd.crosstab(D.obs[GROUP], D.obs[LABEL])
print(n)
print("\ncells per group:", dict(D.obs[GROUP].value_counts()))
```

Percentages are compositional — one population rising forces every other down. Always show the
denominator.

## P6 — find markers for any grouping

```python
sc.tl.rank_genes_groups(D, groupby=LABEL, method="wilcoxon", use_raw=D.raw is not None,
                        pts=True, key_added="rgg")
sc.pl.rank_genes_groups(D, n_genes=15, sharey=False, key="rgg", fontsize=8)

top = sc.get.rank_genes_groups_df(D, group=None, key="rgg")
top.to_csv("markers.csv", index=False)
top.groupby("group").head(5)
```

| `method` | notes |
|---|---|
| `"wilcoxon"` | robust, the usual choice |
| `"t-test"` | fast |
| `"t-test_overestim_var"` | scanpy's conservative t-test |
| `"logreg"` | multi-class, returns no p-values |

```python
# the top markers as a dotplot, straight from the ranking
sc.pl.rank_genes_groups_dotplot(D, n_genes=4, key="rgg", standard_scale="var",
                                cmap="Reds", dendrogram=True)
```

> These are **cluster markers, not differential expression between conditions.** Every cell is a
> sample here, so p-values are inflated by pseudoreplication. For a condition contrast, aggregate
> to one pseudobulk profile per animal and test on those, from `layers['counts']`.

## P7 — export what you plotted

```python
D.obs[[LABEL, COMPARTMENT, BATCH] + FACTORS].to_csv("cell_metadata.csv")
pd.DataFrame(D.obsm[EMB][:, :2], index=D.obs_names, columns=["UMAP1", "UMAP2"]).to_csv("umap.csv")
D.write_h5ad("my_analysis.h5ad")           # your version; the delivered file is unchanged
```

---

## Common problems

| symptom | cause |
|---|---|
| `KeyError: 'X_umap'` | the embedding is under another `obsm` key — print `list(A.obsm)` |
| gene "not found" | it is in `A.raw`, not `A.var` — pass `use_raw=True`, or check the naming (symbol vs Ensembl ID) |
| `sc.tl.leiden` raises about `flavor` | old scanpy: drop `flavor=`/`n_iterations=`, or `pip install igraph leidenalg` |
| every plot is one colour | the column is numeric — `A.obs[c] = A.obs[c].astype("category")` |
| percentages do not sum to 100 | sentinels included in one place and not another — apply `real` consistently |
| clusters do not match `cell_type` | expected in Track B: labels were assigned to the *delivered* clustering |
| dotplot is unreadably wide | `swap_axes=True`, or fewer genes, or `figsize=` |
| memory blows up | `sc.read_h5ad(PATH, backed="r")` to inspect, and subset before loading fully |

## The five things worth remembering

1. **Print the obs columns before writing any code against them.** Names are per-project.
2. **`EXCLUDED` and `UNRESOLVED` are not cell types.** Decide once, apply everywhere, say so in captions.
3. **Track B does not reproduce the report** — different embedding, different clusters, no correspondence to the labels.
4. **On nuclei, `pct_counts_mt` measures carry-over, not mitochondrial biology.** Filter on it if you like; do not interpret it.
5. **Composition is compositional.** One population rising pushes the rest down. Show counts and per-sample points, not only group means.
