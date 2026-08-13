"""The `scanno` command.

Deliberately small. At 0.1.0 there is no driver and no report, so the CLI exposes what
actually exists rather than pretending to a pipeline: build a corpus panel, inspect a
store, run the adversarial suite.

Exit codes follow scQC: 0 pass or review, 1 error, 2 refusal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REFUSE = 2


def _selftest(_):
    import subprocess
    t = Path(__file__).resolve().parents[1] / "tests" / "test_adversarial.py"
    if not t.exists():
        print(f"scanno: no test suite at {t}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, str(t)])


def _resolution(a):
    """Choose a resolution from an already-annotated sweep. Reads obs only."""
    import anndata as ad
    import numpy as np

    from .resolution import format_report, pick_resolution

    depths = tuple(int(x) for x in str(a.depths).split(",") if x.strip())
    labels, clusters, groups, res = {}, {}, [], None
    for path in a.objects:
        A = ad.read_h5ad(path, backed="r")
        cols = [c for c in A.obs.columns if c.startswith(a.prefix)]
        got = [c[len(a.prefix):] for c in cols]
        if res is None:
            res = got
        elif got != res:
            print(f"scanno: {path} carries resolutions {got}, expected {res}. A sweep pooled "
                  f"across objects must be the same sweep in each.", file=sys.stderr)
            return REFUSE
        if not res:
            print(f"scanno: no obs column starts with {a.prefix!r} in {path}", file=sys.stderr)
            return REFUSE
        tag = Path(path).stem
        groups.append(A.obs[a.group_key].astype(str).values
                      if a.group_key in A.obs else np.full(A.n_obs, tag))
        for r, c in zip(res, cols):
            labels.setdefault(r, []).append(A.obs[c].astype(str).values)
            # Cluster ids are per object; qualify them or every object's cluster 0 merges.
            ck = a.cluster_prefix + r.replace("p", ".")
            if ck in A.obs:
                clusters.setdefault(r, []).append(
                    np.char.add(tag + ":", A.obs[ck].astype(str).values))

    labels = {r: np.concatenate(v) for r, v in labels.items()}
    clusters = ({r: np.concatenate(v) for r, v in clusters.items()}
                if len(clusters) == len(labels) else None)
    tree = json.loads(a.tree.read_text(encoding="utf-8")) if a.tree else None
    if tree is None:
        print("  no --tree given: completeness is not measured and that tie-break is skipped")
    out = pick_resolution(labels, tree=tree, groups=np.concatenate(groups), depths=depths,
                          clusters_by_res=clusters)
    print(format_report(out, depths=depths))
    return 0


def _read_manifest(path):
    """TSV: path, source_id, label_key, provenance. One row per annotated dataset.

    `source_id` groups releases that are not independent. `provenance` says how the labels
    were obtained and gates promotion; unrecorded provenance is never assumed clean.
    """
    rows, hdr = [], None
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        f = ln.rstrip("\n").split("\t")
        if hdr is None:
            hdr = [c.strip() for c in f]
            continue
        rows.append(dict(zip(hdr, f)))
    need = {"path", "source_id", "label_key", "provenance"}
    for r in rows:
        missing = need - set(r)
        if missing:
            raise SystemExit(f"scanno: manifest row is missing {sorted(missing)}: {r}")
    return rows


def _calibrate(a):
    """Learn marker reliability from annotated atlases. Consumes atlases, emits numbers."""
    import numpy as np
    from .calibrate import calibrate, save
    from .corpus import load_assertions
    from .store import build_store
    try:
        import anndata as ad
    except ImportError:
        print("scanno: calibrate needs anndata + scanpy.  pip install -e '.[run]'",
              file=sys.stderr)
        return 1

    man = _read_manifest(a.manifest)
    tree = json.loads(Path(a.tree).read_text(encoding="utf-8"))
    ctx = {"species": a.species, "tissue": a.tissue, "assay": a.assay}

    print(f"scanno calibrate — {ctx['species']}/{ctx['tissue']}/{ctx['assay']}, "
          f"{len(man)} dataset(s)")

    # Pass 1 — gene spaces only. Atlases rarely share one, and silently intersecting
    # would make the store's coverage depend on load order. Harmonisation is therefore
    # explicit, reported, and floored.
    spaces = {}
    for r in man:
        A = ad.read_h5ad(r["path"], backed="r")
        src = A.raw if (a.use_raw and A.raw is not None) else A
        spaces[r["source_id"]] = np.array([str(v).upper() for v in src.var_names])
        A.file.close()
    keep = None
    for g in spaces.values():
        keep = set(g) if keep is None else (keep & set(g))
    sizes = {k: len(v) for k, v in spaces.items()}
    same = len({tuple(v) for v in spaces.values()}) == 1

    if not same and not a.harmonise:
        spread = ", ".join(f"{k}:{v:,}" for k, v in sizes.items())
        print(f"scanno: REFUSE — the datasets do not share a gene space ({spread}); "
              f"the intersection is {len(keep):,}."
              f"\n        Pass --harmonise to intersect them explicitly. Doing it "
              f"silently would make"
              f"\n        the store's coverage depend on which atlas loaded first.",
              file=sys.stderr)
        return REFUSE
    if not same:
        med = int(np.median(list(sizes.values())))
        print(f"  harmonised to {len(keep):,} shared genes "
              f"({', '.join(f'{k} {v:,}' for k, v in sizes.items())})")
        if len(keep) < a.min_shared_genes:
            print(f"scanno: REFUSE — only {len(keep):,} genes are shared, below "
                  f"--min-shared-genes {a.min_shared_genes:,}."
                  f"\n        A store this narrow cannot represent most cell types, and "
                  f"every downstream"
                  f"\n        score would be computed on whatever the thinnest atlas "
                  f"happened to include.", file=sys.stderr)
            return REFUSE
        if len(keep) < 0.5 * med:
            print(f"  REVIEW  the intersection is {100*len(keep)/med:.0f}% of the median "
                  f"input ({med:,}); the narrowest atlas is deciding coverage")

    order = np.array(sorted(keep))

    def stream():
        for r in man:
            A = ad.read_h5ad(r["path"])
            src = A.raw if (a.use_raw and A.raw is not None) else A
            if r["label_key"] not in A.obs:
                raise SystemExit(f"scanno: {r['path']} has no obs column "
                                 f"{r['label_key']!r}")
            g = np.array([str(v).upper() for v in src.var_names])
            sel = np.array([np.where(g == x)[0][0] for x in order])
            X = src.X[:, sel]
            print(f"  {r['source_id']:<16} {A.shape[0]:>7,} cells  "
                  f"{len(order):>6,} genes  labels={r['label_key']}  "
                  f"provenance={r['provenance']}")
            yield (r["source_id"], order, X,
                   A.obs[r["label_key"]].astype(str).values, r["provenance"])

    store = build_store(stream(), ctx)
    asr = load_assertions(a.db, a.species, a.tissue, a.min_tier)
    if not asr:
        print(f"scanno: REFUSE — the corpus has nothing for {a.species}/{a.tissue} at "
              f"tier<={a.min_tier}. There is no panel to calibrate.", file=sys.stderr)
        return REFUSE
    cal = calibrate(store, asr, tree, ctx)
    out = save(cal, store, a.out)

    c = cal.census
    print(f"\nstore   {c['celltypes']} cell types, {c['genes']:,} genes, "
          f"digest {store.digest}")
    print(f"grades  C1 {c['C1']} · C2 {c['C2']} · C3 {c['C3']}   "
          f"(C1 needs >=5 sources and >=3 label-clean)")
    print(f"panels  {c['claims_scored']:,} claims over {len(cal.nodes)} nodes  "
          f"-> {c['promoted']} promoted, {c['demoted']} demoted")
    if c["C1"] == 0:
        print("\nNOTE  no cell type reached C1. With few independent, label-clean sources"
              "\n      the learned weights are weakly supported and are pooled toward the"
              "\n      gene-level prior. Add sources before trusting a promotion.")
    print(f"\nwrote {out}/  store.npz  reliability.tsv  panels.tsv  calibration.json")
    return 0


def _annotate(a):
    """Label the clusters of one object. The main command."""
    import numpy as np
    from .classify import classify
    from .corpus import load_assertions
    from .query import OOD_MIN_COVERED, cluster_profile, standardise
    from .store import build_store
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError:
        print("scanno: annotate needs anndata + scanpy.  pip install -e '.[run]'",
              file=sys.stderr)
        return 1

    tree = json.loads(Path(a.tree).read_text(encoding="utf-8"))
    A = ad.read_h5ad(a.h5ad)
    if a.cluster_key not in A.obs:
        print(f"scanno: {a.h5ad} has no obs column {a.cluster_key!r}. Available: "
              f"{', '.join(list(A.obs.columns)[:12])}", file=sys.stderr)
        return 1
    src = A.raw if (a.use_raw and A.raw is not None) else A
    X, genes = src.X, np.array([str(v).upper() for v in src.var_names])

    # Counts or already normalised? Measured, not assumed - feeding scaled or raw values
    # to a scorer expecting log1p returns a number for the wrong quantity.
    import scipy.sparse as sp
    head = X[:200].toarray() if sp.issparse(X) else np.asarray(X[:200])
    if float(np.max(head)) > 50 and np.allclose(head, np.round(head)):
        print("  .X looks like raw counts -> normalize_total(1e4) + log1p")
        tmp = ad.AnnData(X=X.copy())
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        X = tmp.X
    elif float(np.min(head)) < 0:
        print("scanno: REFUSE - .X contains negative values, so it is scaled rather than"
              "\n        log-normalised. Pass --use-raw, or supply an object whose .X is"
              "\n        log1p counts.", file=sys.stderr)
        return REFUSE

    lab = A.obs[a.cluster_key].astype(str).values
    cats = sorted(set(lab), key=lambda s: (len(s), s))
    y = np.array([cats.index(v) for v in lab])
    M, D, counts = cluster_profile(X, y, len(cats))

    # The gene background. Without one there is nothing external to standardise against,
    # and a cluster's score becomes a property of what else was sequenced beside it.
    if a.store:
        from .calibrate import load_store
        store = load_store(a.store)
        bg = f"store {Path(a.store).name} (digest {store.digest})"
    elif a.background_from_clusters:
        store = build_store([("query", genes, X, lab)],
                            {"species": a.species, "tissue": a.tissue, "assay": a.assay})
        bg = "THIS OBJECT's own clusters"
        print("  REVIEW  the gene background comes from this object's own clusters, so a\n"
              "          score is not fully independent of what else was sequenced in it.\n"
              "          Use ONE store across every sample if you intend to compare them.")
    else:
        print("scanno: REFUSE - no gene background.\n"
              "        Pass --store from `scanno calibrate`, or "
              "--background-from-clusters to\n"
              "        derive one from this object and accept that scores are then not\n"
              "        independent of its composition.", file=sys.stderr)
        return REFUSE

    Z, usable, st = standardise(M, D, genes, store)
    if st["ood_covered"] < OOD_MIN_COVERED:
        print(f"scanno: REFUSE - the background covers only "
              f"{100*st['ood_covered']:.0f}% of the genes this object expresses "
              f"(floor {100*OOD_MIN_COVERED:.0f}%).\n"
              f"        It cannot speak to this data.", file=sys.stderr)
        return REFUSE

    asr = None
    if a.db:
        asr = load_assertions(a.db, a.species, a.tissue, a.min_tier)
        if not asr:
            print(f"scanno: REFUSE - the corpus has nothing for {a.species}/{a.tissue} "
                  f"at tier<={a.min_tier}.", file=sys.stderr)
            return REFUSE
    tree["genes"] = store.genes
    res = classify(Z, usable, tree, store=None if asr else store, assertions=asr,
                   gap_min=a.gap_min)

    support = {}
    if a.db:
        from .corpus import node_support
        support = node_support(a.db, a.species, a.tissue, tree.get("patterns", {}))

    src_txt = "corpus" if asr else "atlas profiles"
    print("")
    print(f"{a.cluster_key}: {len(cats)} clusters, {A.shape[0]:,} cells   "
          f"weights={src_txt}   background={bg}")
    print(f"genes usable {st['genes_usable']:,} / {st['genes_detected']:,} detected   "
          f"OOD coverage {100*st['ood_covered']:.0f}%")
    print("")
    print(f"{'cluster':>10} {'n':>8}  {'label':<40}{'depth':>6}{'gap':>7}{'support':>9}")
    for r in res:
        c = r["cluster"]
        sup = support.get(r["label"], "")
        warn = " *" if isinstance(sup, int) and 0 < sup < 10 else ""
        print(f"{cats[c]:>10} {counts[c]:>8,.0f}  {r['path']:<40}"
              f"{r['depth']:>6}{r['gap']:>7.2f}{str(sup):>9}{warn}")
    thin = sorted({r["label"] for r in res
                   if isinstance(support.get(r["label"]), int)
                   and 0 < support[r["label"]] < 10})
    if thin:
        print("")
        print("  * fewer than 10 curated assertions behind the panel: " + ", ".join(thin))
        print("    A thin panel can win with a healthy gap. Treat these as provisional.")
    unres = sum(counts[r["cluster"]] for r in res if r["path"] == "UNRESOLVED")
    n_un = sum(1 for r in res if r["path"] == "UNRESOLVED")
    print("")
    print(f"UNRESOLVED {n_un} clusters = {unres:,.0f} cells "
          f"({100*unres/counts.sum():.1f}%)")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(a.out).open("w", encoding="utf-8") as fh:
            fh.write("\t".join(["cluster", "n_cells", "label", "path", "depth",
                                "gap"]) + "\n")
            for r in res:
                c = r["cluster"]
                fh.write("\t".join([cats[c], f"{counts[c]:.0f}", r["label"], r["path"],
                                    str(r["depth"]), f"{r['gap']:.4f}"]) + "\n")
        print(f"wrote {a.out}")
    return 0


def _panel(a):
    """Show the corpus panel for one context — what the untrained path would score on."""
    from .corpus import load_assertions
    try:
        asr = load_assertions(a.db, a.species, a.tissue, a.min_tier)
    except Exception as e:                                          # noqa: BLE001
        print(f"scanno: cannot read {a.db}: {e}", file=sys.stderr)
        return 1
    if not asr:
        print(f"scanno: REFUSE — no assertions for {a.species}/{a.tissue} at "
              f"tier<={a.min_tier}. The corpus cannot speak to this context.",
              file=sys.stderr)
        return REFUSE
    rows = sorted(((c, len(g)) for c, g in asr.items()), key=lambda r: -r[1])
    print(f"{a.species} / {a.tissue}   tier<={a.min_tier}   "
          f"{len(asr)} cell names, {sum(n for _, n in rows):,} (cell, gene) claims")
    for c, n in rows[:a.top]:
        best = sorted(asr[c].items(), key=lambda kv: -kv[1])[:8]
        print(f"  {c[:38]:<40}{n:>5}   {', '.join(g for g, _ in best)}")
    if len(rows) > a.top:
        print(f"  ... and {len(rows) - a.top} more cell names")
    return 0


def _store_info(a):
    import numpy as np
    d = np.load(a.store, allow_pickle=True)
    print(json.dumps({k: (v.tolist() if getattr(v, "ndim", 0) == 0 else f"array{v.shape}")
                      for k, v in d.items()}, indent=2)[:2000])
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="scanno", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="run the adversarial suite")
    s.set_defaults(fn=_selftest)

    s = sub.add_parser("annotate", help="label the clusters of one object")
    s.add_argument("--h5ad", required=True, type=Path)
    s.add_argument("--cluster-key", required=True)
    s.add_argument("--tree", required=True, type=Path)
    s.add_argument("--species", required=True)
    s.add_argument("--tissue", required=True)
    s.add_argument("--assay", default="sc", choices=["sc", "sn"])
    s.add_argument("--db", type=Path, help="marker corpus (corpus weight path)")
    s.add_argument("--store", type=Path, help="store.npz from `scanno calibrate`")
    s.add_argument("--background-from-clusters", action="store_true",
                   help="derive the gene background from this object; reported as REVIEW")
    s.add_argument("--gap-min", type=float, default=None,
                   help="override the descent threshold (0.30 corpus, 0.15 profiles)")
    s.add_argument("--min-tier", type=int, default=4)
    s.add_argument("--use-raw", action="store_true")
    s.add_argument("--out", type=Path)
    s.set_defaults(fn=_annotate)

    s = sub.add_parser("panel", help="show the corpus panel for one species x tissue")
    s.add_argument("--db", required=True, type=Path)
    s.add_argument("--species", required=True)
    s.add_argument("--tissue", required=True)
    s.add_argument("--min-tier", type=int, default=4)
    s.add_argument("--top", type=int, default=20)
    s.set_defaults(fn=_panel)

    s = sub.add_parser("calibrate", help="learn marker reliability from annotated atlases")
    s.add_argument("--manifest", required=True, type=Path,
                   help="TSV: path, source_id, label_key, provenance")
    s.add_argument("--db", required=True, type=Path, help="marker corpus")
    s.add_argument("--tree", required=True, type=Path, help="declared tree, JSON")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--species", required=True)
    s.add_argument("--tissue", required=True)
    s.add_argument("--assay", default="sc", choices=["sc", "sn"])
    s.add_argument("--min-tier", type=int, default=4)
    s.add_argument("--harmonise", action="store_true",
                   help="intersect gene spaces across datasets, explicitly and reported")
    s.add_argument("--min-shared-genes", type=int, default=2000,
                   help="refuse below this many shared genes (default 2000)")
    s.add_argument("--use-raw", action="store_true",
                   help="read .raw (log-normalised) rather than .X, which is often scaled")
    s.set_defaults(fn=_calibrate)

    s = sub.add_parser("store-info", help="describe a saved profile store")
    s.add_argument("--store", required=True, type=Path)
    s.set_defaults(fn=_store_info)

    s = sub.add_parser("resolution",
                       help="choose a clustering resolution from an annotated sweep")
    s.add_argument("objects", nargs="+", type=Path,
                   help="annotated .h5ad file(s), pooled - which is what you want when "
                        "clustering was done per sample")
    s.add_argument("--prefix", default="scanno_path_r",
                   help="obs column prefix; the rest of the name is the resolution")
    s.add_argument("--tree", type=Path,
                   help="declared tree, JSON. Without it completeness cannot be measured and "
                        "that tie-break is skipped rather than guessed")
    s.add_argument("--depths", default="1,2",
                   help="tree depths the choice must serve (default 1,2)")
    s.add_argument("--group-key", default="sample",
                   help="obs column naming the biological unit, for rare-label presence")
    s.add_argument("--cluster-prefix", default="leiden_",
                   help="obs prefix for cluster ids, used only for the parsimony tie-break. "
                        "Absent, distinct labels are counted and are SAID to be")
    s.set_defaults(fn=_resolution)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
