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

### Sweep the UMAP: 25 layouts, and pick the one you can read

`min_dist` and `n_neighbors` change the **picture**, not the data. Neither one makes an embedding
more correct — they trade local detail against global shape — so the honest way to set them is to
look at a grid and choose, rather than to accept a default you never saw an alternative to.

```python
MIN_DISTS   = [0.05, 0.1, 0.3, 0.5, 0.8]        # tight clumps  ->  even spread
N_NEIGHBORS = [5, 15, 30, 50, 100]              # local detail  ->  global structure
COLOR_BY    = LABEL                             # or "my_leiden", or a gene
POINT_SIZE  = 1.5
SEED        = 0
```

Neighbours are recomputed once per `n_neighbors` (the expensive step), then each `min_dist` is a
cheap re-layout on the same graph — 5 graphs and 25 layouts rather than 25 of each.

```python
import itertools, time

vals = B.obs[COLOR_BY].astype(str)
cats = sorted(set(vals))
cmap = {c: plt.cm.tab20(i % 20) for i, c in enumerate(cats)}
colors = np.array([cmap[v] for v in vals])

fig, axes = plt.subplots(len(N_NEIGHBORS), len(MIN_DISTS),
                         figsize=(3.1 * len(MIN_DISTS), 3.1 * len(N_NEIGHBORS)))
t0 = time.time()
for i, nn in enumerate(N_NEIGHBORS):
    sc.pp.neighbors(B, n_neighbors=nn, n_pcs=N_NEIGHBORS_PCS, random_state=SEED)  # once per row
    for j, md in enumerate(MIN_DISTS):
        sc.tl.umap(B, min_dist=md, spread=SPREAD, random_state=SEED)
        xy = B.obsm["X_umap"]
        ax = axes[i, j]
        ax.scatter(xy[:, 0], xy[:, 1], s=POINT_SIZE, c=colors, linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(f"min_dist={md}", fontsize=10)
        if j == 0:
            ax.set_ylabel(f"n_neighbors={nn}", fontsize=10)
        B.obsm[f"X_umap_nn{nn}_md{str(md).replace('.', 'p')}"] = xy.copy()   # keep every layout
    print(f"  n_neighbors={nn} done  ({time.time() - t0:.0f}s)", flush=True)

fig.suptitle(f"UMAP sweep, coloured by {COLOR_BY} — 25 layouts of the SAME data",
             fontsize=13, y=1.002)
plt.tight_layout(); plt.savefig("umap_sweep.png", dpi=130, bbox_inches="tight"); plt.show()
```

Every layout is kept in `obsm` as `X_umap_nn<N>_md<M>`, so choosing one is a rename, not a re-run:

```python
PICK_NN, PICK_MD = 15, 0.3                       # <- the one you liked
B.obsm["X_umap"] = B.obsm[f"X_umap_nn{PICK_NN}_md{str(PICK_MD).replace('.', 'p')}"].copy()
B.uns["umap_choice"] = {"n_neighbors": PICK_NN, "min_dist": PICK_MD, "seed": SEED,
                        "n_pcs": N_NEIGHBORS_PCS}      # record it, or it is unreproducible
```

| you want | reach for |
|---|---|
| tight, well-separated blobs | low `min_dist` (0.05–0.1) |
| an even, space-filling cloud | high `min_dist` (0.5–0.8) |
| fine local structure, more fragments | low `n_neighbors` (5–15) |
| global relationships, fewer islands | high `n_neighbors` (50–100) |

**On 100k cells this is 25 UMAPs and will take a while** — it is the one block in this playbook
worth running as a batch job rather than interactively. Shrink the grid to 3×3 first, or subsample:

```python
# Bsub = B[np.random.default_rng(0).choice(B.n_obs, 20000, replace=False)].copy()
```

> **A prettier UMAP is not a better result.** Distances between clusters carry little meaning and
> cluster sizes carry none; the sweep is for choosing something legible, not something true. Any
> claim you can make from one panel of this grid should survive in all 25 — if it does not, it is
> a property of the layout.

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

---

# DIFFERENTIAL EXPRESSION

```python
# pip install pydeseq2
```

## D1 — which question are you asking?

These are different questions and they need different tests.

| question | unit | tool |
|---|---|---|
| what marks cluster X against the others? | the **cell** | `sc.tl.rank_genes_groups` — Block P6 |
| does gene G differ between conditions? | the **sample** | pseudobulk + DESeq2 — below |

Running a per-cell test between conditions treats every cell as an independent replicate. It is
not: cells from one animal share that animal. The p-values come out spectacular and do not
replicate. **Aggregate to one profile per sample first.**

## D2 — pseudobulk

```python
import decoupler as dc

SAMPLE_COL = BATCH          # one value per biological replicate
GROUPS_COL = LABEL          # aggregate WITHIN cell type, so each type is tested separately
MODE       = "sum"          # "sum" for DESeq2 · "mean" · a callable

pdata = dc.pp.pseudobulk(D, sample_col=SAMPLE_COL, groups_col=GROUPS_COL,
                         layer=COUNTS,          # RAW COUNTS. DESeq2 models counts.
                         mode=MODE)
print(pdata)
print(pdata.obs[["psbulk_cells", "psbulk_counts"]].describe())
```

`pseudobulk` adds `psbulk_cells` (how many cells went into each profile) and `psbulk_counts`.
**A profile built from 12 cells is not the equal of one built from 4,000** — drop the thin ones:

```python
MIN_CELLS_PER_PROFILE = 30
pdata = pdata[pdata.obs["psbulk_cells"] >= MIN_CELLS_PER_PROFILE].copy()
print(pd.crosstab(pdata.obs[GROUPS_COL], pdata.obs[FACTORS[0] if FACTORS else SAMPLE_COL]))
```

That table is the one to read before believing any result below: a cell type with two samples in
one arm and five in the other is not a comparison, whatever p-value comes out.

## D3 — DESeq2, one cell type at a time

```python
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference

CONDITION = FACTORS[0] if FACTORS else None     # the obs column to contrast
TEST, REF = "trt", "ctrl"                       # <- EDIT: the two levels, test first
CELLTYPE  = "Fibroblast"                        # <- EDIT: which population
N_CPUS    = 4

sub = pdata[(pdata.obs[GROUPS_COL].astype(str) == CELLTYPE)].copy()
sub = sub[sub.obs[CONDITION].astype(str).isin([TEST, REF])].copy()
print(f"{sub.n_obs} samples: {dict(sub.obs[CONDITION].value_counts())}")

dc.pp.filter_by_expr(sub, group=CONDITION, min_count=10, min_total_count=15)
print(f"{sub.n_vars:,} genes kept")

sub.X = np.rint(np.asarray(sub.X)).astype(int)     # DESeq2 requires INTEGER counts
inf = DefaultInference(n_cpus=N_CPUS)
dds = DeseqDataSet(adata=sub, design=f"~{CONDITION}", refit_cooks=True, inference=inf, quiet=True)
dds.deseq2()
stat = DeseqStats(dds, contrast=[CONDITION, TEST, REF], inference=inf, quiet=True)
stat.summary()
res = stat.results_df.sort_values("stat", ascending=False)
res.head(10).round(4)
```

`results_df` columns: `baseMean`, `log2FoldChange`, `lfcSE`, `stat`, `pvalue`, `padj`.

| parameter | effect |
|---|---|
| `design` | `"~group"`, or `"~batch + group"` to control for a covariate — the term of interest goes **last** |
| `contrast` | `[column, test, reference]` — the sign follows this order |
| `refit_cooks` | re-fit after removing outliers |
| `min_count` / `min_total_count` | `filter_by_expr` thresholds; raise for noisier data |
| `alpha` | on `DeseqStats`, the FDR level used for independent filtering |

```python
SIG_PADJ, SIG_LFC = 0.05, 1.0
sig = res[(res["padj"] < SIG_PADJ) & (res["log2FoldChange"].abs() > SIG_LFC)]
print(f"{len(sig)} genes at padj<{SIG_PADJ}, |log2FC|>{SIG_LFC}")
res.to_csv(f"de_{CELLTYPE}_{TEST}_vs_{REF}.csv")
```

### Volcano

```python
plt.figure(figsize=(6, 5))
x = res["log2FoldChange"]; y = -np.log10(res["padj"].clip(lower=1e-300))
plt.scatter(x, y, s=6, c="lightgrey", linewidths=0)
m = (res["padj"] < SIG_PADJ) & (x.abs() > SIG_LFC)
plt.scatter(x[m], y[m], s=8, c=np.where(x[m] > 0, "#D62728", "#1F77B4"), linewidths=0)
for g in res[m].head(10).index:
    plt.annotate(g, (x[g], y[g]), fontsize=7)
plt.axhline(-np.log10(SIG_PADJ), ls="--", lw=0.8, c="k")
plt.axvline(SIG_LFC, ls="--", lw=0.8, c="k"); plt.axvline(-SIG_LFC, ls="--", lw=0.8, c="k")
plt.xlabel(f"log2FC  ({TEST} vs {REF})"); plt.ylabel("-log10 padj")
plt.title(f"{CELLTYPE}"); plt.tight_layout(); plt.show()
```

### Every cell type in a loop

```python
de = {}
for ct in pdata.obs[GROUPS_COL].astype(str).unique():
    s = pdata[(pdata.obs[GROUPS_COL].astype(str) == ct)
              & pdata.obs[CONDITION].astype(str).isin([TEST, REF])].copy()
    if s.n_obs < 4 or s.obs[CONDITION].nunique() < 2:
        print(f"skip {ct}: {s.n_obs} samples")           # NAMED, not silently dropped
        continue
    try:
        dc.pp.filter_by_expr(s, group=CONDITION, min_count=10, min_total_count=15)
        s.X = np.rint(np.asarray(s.X)).astype(int)
        d = DeseqDataSet(adata=s, design=f"~{CONDITION}", inference=inf, quiet=True)
        d.deseq2()
        st = DeseqStats(d, contrast=[CONDITION, TEST, REF], inference=inf, quiet=True)
        st.summary()
        de[ct] = st.results_df
        print(f"{ct}: {(st.results_df['padj'] < 0.05).sum()} genes at padj<0.05")
    except Exception as e:
        print(f"FAILED {ct}: {type(e).__name__}: {e}")
```

---

# ENRICHMENT

Two libraries, for two different shapes of question.

| | **gseapy** | **decoupler** |
|---|---|---|
| input | one ranked gene list, or a gene list | a whole **matrix** |
| gives you | enriched terms for **one contrast** | a score **per observation** |
| use it for | "what is up in my DE result?" | "what is this cell's TF/pathway activity?" |
| gene sets | Enrichr libraries, MSigDB, `.gmt` | CollecTRI, DoRothEA, PROGENy, hallmark, `.gmt` |

**Both fetch prior knowledge over the network.** On an offline machine, download a `.gmt` once
and read it locally — see the offline note in E4.

## E1 — gseapy: preranked GSEA on a DE result

Uses the **whole ranking**, so it needs no significance cutoff — the right choice when few genes
survive FDR.

```python
import gseapy as gp

rnk = res["stat"].dropna().sort_values(ascending=False)      # the DESeq2 Wald statistic
print(rnk.head(3)); print(rnk.tail(3))

pre = gp.prerank(rnk=rnk,
                 gene_sets="MSigDB_Hallmark_2020",   # a name, a .gmt path, or a dict
                 organism="Mouse",
                 min_size=15, max_size=500,
                 permutation_num=1000,
                 threads=4, seed=0, outdir=None)
gsea_res = pre.res2d.sort_values("NES", ascending=False)
gsea_res[["Term", "NES", "NOM p-val", "FDR q-val", "Tag %"]].head(10)
```

| column | meaning |
|---|---|
| `NES` | normalised enrichment score — sign gives direction |
| `NOM p-val` | nominal permutation p |
| `FDR q-val` | **use this**, not the nominal p |
| `Lead_genes` | the genes driving it |

```python
gp.dotplot(gsea_res, column="FDR q-val", title="GSEA", cmap="viridis_r",
           size=5, top_term=15, figsize=(4, 6))
```

```python
# the classic running-enrichment curve for one term
term = gsea_res["Term"].iloc[0]
gp.gseaplot(term=term, **pre.results[term], rank_metric=pre.ranking, figsize=(6, 5.5))
```

### Which libraries exist

```python
libs = gp.get_library_name(organism="Mouse")
print(len(libs), "libraries")
print([l for l in libs if "Hallmark" in l or "GO_Biological" in l or "KEGG" in l][:10])
```

## E2 — gseapy: over-representation on a gene LIST

```python
up   = sig[sig["log2FoldChange"] > 0].index.tolist()
down = sig[sig["log2FoldChange"] < 0].index.tolist()
background = res.index.tolist()          # every TESTED gene — not the whole genome
print(f"{len(up)} up, {len(down)} down, background {len(background):,}")

enr = gp.enrichr(gene_list=up,
                 gene_sets=["MSigDB_Hallmark_2020", "GO_Biological_Process_2023"],
                 organism="Mouse",
                 background=background,
                 outdir=None)
enr.results.sort_values("Adjusted P-value").head(10)[
    ["Gene_set", "Term", "Overlap", "Adjusted P-value", "Genes"]]
```

> **The background matters more than people expect.** ORA asks whether your list is enriched
> *relative to what could have been picked*. Using the whole genome when you only tested 12,000
> genes inflates every p-value. Pass the genes you actually tested.

```python
gp.barplot(enr.results, column="Adjusted P-value", top_term=15, figsize=(5, 6),
           color="salmon", title="up in " + TEST)
```

Offline equivalent, with a local `.gmt` and no network:

```python
# enr = gp.enrich(gene_list=up, gene_sets="path/to/sets.gmt", background=background, outdir=None)
```

## E3 — decoupler: an activity score for every cell

This is the part that is not just "enrichment of a list". It scores **each observation**, so you
can put TF or pathway activity on the UMAP.

### Get a network

```python
ORGANISM = "mouse"          # or "human"

tf_net   = dc.op.collectri(organism=ORGANISM)                 # weighted, ~1,165 TFs
pw_net   = dc.op.progeny(organism=ORGANISM, top=500)          # weighted, 14 pathways
hall_net = dc.op.hallmark(organism=ORGANISM)                  # UNWEIGHTED, 50 sets

for nm, n in (("collectri", tf_net), ("progeny", pw_net), ("hallmark", hall_net)):
    print(f"{nm:10s} {n.shape[0]:>7,} rows  {n['source'].nunique():>4} sources  "
          f"weighted={'weight' in n.columns}  columns={list(n.columns)}")
```

> **Weighted or not decides the method.** `ulm` and `mlm` use the weights, so they need a
> weighted net. `ora`, `gsea` and `aucell` do not. Handing hallmark to `ulm` is a category error.

| net | sources | weighted | pair with |
|---|---|---|---|
| `collectri` | TF regulons | **yes** | `ulm`, `mlm` |
| `dorothea` | TF regulons, confidence levels | **yes** | `ulm`, `mlm` |
| `progeny` | 14 signalling pathways | **yes** | `ulm`, `mlm` |
| `hallmark` | 50 MSigDB sets | **no** | `ora`, `gsea`, `aucell` |
| `dc.pp.read_gmt(path)` | your own | no | `ora`, `gsea`, `aucell` |

### Score every cell

```python
# INPUT MUST BE NORMALISED (log1p), NOT raw counts. D.X already is.
dc.mt.ulm(D, tf_net, tmin=5, verbose=True)          # returns None; writes into D.obsm
print([k for k in D.obsm if k.startswith(("score_", "padj_"))])
```

> **The single most-missed thing about this library.** With an **AnnData** it returns `None` and
> writes `D.obsm["score_ulm"]` / `D.obsm["padj_ulm"]` **in place**. With a **DataFrame** it
> returns an `(es, pv)` tuple instead. `es = dc.mt.ulm(adata, net)` silently gives you `None`.

```python
acts = dc.pp.get_obsm(D, key="score_ulm")     # an AnnData: cells x sources
print(acts)                                    # var_names are the TFs
```

`verbose=True` prints how many sources survived `tmin`. **`tmin` silently drops sources with
fewer than that many targets present in your data** — on a small net, lower it (`tmin=3`).

### Plot activity like any other feature

```python
acts.obs = D.obs.copy()
acts.obsm["X_umap"] = D.obsm[EMB]
TFS = acts.var_names[:4].tolist()
sc.pl.umap(acts, color=TFS, cmap="RdBu_r", vcenter=0, ncols=2, size=3, frameon=False)
sc.pl.violin(acts, TFS[:2], groupby=LABEL, rotation=90, stripplot=False)
```

### Which TFs mark which cell type

```python
ranked = dc.tl.rankby_group(acts, groupby=LABEL, reference="rest",
                            method="t-test_overestim_var")
ranked.head(10)
```

```python
dc.pl.dotplot(ranked, x="group", y="name", c="stat", s="meanchange", top=5,
              cmap="RdBu_r", vcenter=0, figsize=(9, 7))
```

### Pathway activity

```python
dc.mt.mlm(D, pw_net, tmin=5, verbose=True)
pw = dc.pp.get_obsm(D, key="score_mlm")
pw.obs = D.obs.copy(); pw.obsm["X_umap"] = D.obsm[EMB]
sc.pl.umap(pw, color=list(pw.var_names)[:6], cmap="RdBu_r", vcenter=0, ncols=3, size=3)
sc.pl.matrixplot(pw, list(pw.var_names), groupby=LABEL, cmap="RdBu_r",
                 vcenter=0, dendrogram=True, colorbar_title="pathway activity")
```

## E4 — decoupler on a DE result, several methods, and consensus

`decoupler` also takes a **contrast**: one row of statistics indexed by gene, which is exactly
what `results_df["stat"]` is.

```python
mat = res[["stat"]].dropna().T
mat.index = [f"{TEST}_vs_{REF}"]
print(mat.shape)                       # 1 x n_genes

es, pv = dc.mt.ora(mat, hall_net, tmin=5)      # DataFrame in -> (es, pv) OUT
out = (es.T.join(pv.T, lsuffix="_score", rsuffix="_padj")
         .set_axis(["score", "padj"], axis=1).sort_values("score", ascending=False))
out.head(10).round(4)
```

```python
dc.pl.barplot(es, name=mat.index[0], top=20, vertical=True, cmap="RdBu_r", figsize=(5, 7))
```

Run several methods and combine them — no method is best on every dataset:

```python
scores = dc.mt.decouple(mat, hall_net, methods=["ora", "gsea", "zscore"], tmin=5, cons=True)
print(list(scores))          # score_ora, padj_ora, ..., score_consensus, padj_consensus
scores["score_consensus"].T.sort_values(mat.index[0], ascending=False).head(10).round(3)
```

`consensus` is the mean of the per-method z-scores, so it is only as good as the methods in it.
Adding a method that fails on your data drags it.

### Offline, and non-matching gene names

```python
# Offline: download a .gmt once (MSigDB / Enrichr), then
# net = dc.pp.read_gmt("mh.all.v2024.1.Mm.symbols.gmt")     # -> source/target long format
# gp.prerank(rnk=rnk, gene_sets="mh.all.v2024.1.Mm.symbols.gmt", ...)
```

If your object is indexed by Ensembl IDs, or your net is human and your data mouse, **nothing
overlaps and decoupler refuses**:

```text
AssertionError: No sources with more than tmin=5 targets after
    filtering by shared features in mat.
```

That message means gene names, not `tmin`. Check first, convert second:

```python
print("overlap:", len(set(D.var_names) & set(tf_net["target"])), "of", D.n_vars)
# human net -> mouse symbols
# tf_net = dc.op.translate(tf_net, columns="target", target_organism="mouse")
```

## E5 — what enrichment cannot tell you

- **A gene set is a hypothesis someone else wrote down.** Terms are redundant, unevenly curated,
  and biased toward what has been studied. An enriched term is a pointer, not a conclusion.
- **ORA throws away the ranking**; GSEA keeps it. If few genes clear FDR, prefer preranked GSEA
  over ORA on a short list.
- **Activity is not measured, it is inferred** from the targets' expression under an assumed
  regulon. A TF with a wrong or incomplete regulon gets a confident, wrong score.
- **The background decides the p-value** in ORA. Use the genes you tested.
- **Nothing here fixes the design.** If a contrast is confounded, the enrichment of it is
  confounded too, and it will read as clean biology.

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
