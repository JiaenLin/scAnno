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


def _http_presets():
    from .agent import HTTPProvider
    return HTTPProvider.PRESETS


def _agent(a):
    """A second opinion on each cluster, from a model you supply. Never replaces `annotate`."""
    import csv as _csv

    import anndata as ad
    import numpy as np

    from . import (check_gene_space, classify, cluster_profile, load_assertions,
                   standardise)
    from .agent import CommandProvider, HTTPProvider, agreement, annotate_agentic
    from .store import build_store

    if a.command:
        if a.web:
            print("scanno: --web applies to hosted providers. A --command agent brings its own "
                  "tools; scAnno neither adds nor removes them.", file=sys.stderr)
            return REFUSE
        prov = CommandProvider(a.command, model=a.model)
    else:
        try:
            prov = HTTPProvider(preset=a.provider, model=a.model, url=a.url,
                                temperature=a.temperature, web=a.web)
        except (RuntimeError, ValueError) as e:
            print(f"scanno: {e}", file=sys.stderr)
            return REFUSE
    print(f"provider {prov.name}  model {getattr(prov, 'model', '?')}  votes {a.votes}"
          f"  web {'on' if getattr(prov, 'web', False) else 'off'}")

    A = ad.read_h5ad(a.object)
    genes = np.array([str(v).upper() for v in
                      (A.var["gene_symbol"] if "gene_symbol" in A.var else A.var_names)])
    lab = A.obs[a.cluster_key].astype(str).values
    cats = sorted(set(lab), key=lambda v: (len(v), v))
    y = np.array([cats.index(v) for v in lab])
    M, D, counts = cluster_profile(A.X, y, len(cats))

    tree = json.loads(a.tree.read_text(encoding="utf-8"))
    store = build_store([("query", genes, A.X, lab)],
                        {"species": a.species, "tissue": a.tissue, "assay": a.assay})
    Z, usable, _ = standardise(M, D, genes, store)
    tree["genes"] = store.genes

    corpus = None
    if a.db:
        asr = load_assertions(str(a.db), a.species, a.tissue, min_tier=4)
        check_gene_space(asr, store.genes)
        corpus = classify(Z, usable, tree, store=None, assertions=asr)
        print("the corpus call is computed for COMPARISON and is never shown to the model")

    rows = annotate_agentic(prov, Z, usable, store.genes, tree, db=a.db, top_n=a.top_n,
                            votes=a.votes, species=a.species, tissue=a.tissue, assay=a.assay,
                            corpus_calls=corpus, log=print)
    for r, c in zip(rows, cats):
        r["cluster_id"], r["n_cells"] = c, int(counts[r["cluster"]])

    cols = ["cluster_id", "n_cells", "tier", "resolved", "cl", "label", "confidence",
            "consensus", "votes", "reason", "corpus_path", "comparable", "agrees",
            "provider", "model", "prompt_sha", "top_genes", "error", "raw"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {a.out}")
    ag = agreement(rows)
    if corpus is not None and ag["n_comparable"]:
        print(f"\non the taxonomy, where the agent used it: {ag['agree']:.0%} of "
              f"{ag['n_comparable']} clusters agree with the corpus")
        for c, cp, al in ag["disagreements"]:
            print(f"  cluster {cats[c]:<6} corpus {cp:<38} agent {al}")
    if ag["off_tree"]:
        print(f"\n{len(ag['off_tree'])} cluster(s) the agent placed OUTSIDE the taxonomy. "
              f"These are not disagreements - they say the tree has no word for this:")
        for c, tier, lab, cl, cp in ag["off_tree"]:
            print(f"  cluster {cats[c]:<6} [{tier}] {lab}{f' [{cl}]' if cl else ''}"
                  f"{f'   corpus called it {cp}' if cp else ''}")
    print("\nNothing here changed a classifier call. `scanno annotate` is unaffected; this is a "
          "second column to read beside it.")
    return 0


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
    from .exclude import (CELL, EXCLUDED, cluster_flags, exclusion_record,
                          exclusion_record_cells, unprofilable)
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

    # --- nuclei upstream QC flagged: excluded from the walk, never from the object ---
    #
    # TWO MODES, and the default is `cell`. In cell mode the flagged nuclei are dropped from the
    # PROFILE - they contribute to no cluster's mean and to no detection rate - and are labelled
    # EXCLUDED individually. In cluster mode a whole cluster goes once it is `share` flagged,
    # which also takes its unflagged members. See scanno/exclude.py for why cell is the default.
    drop, excl, flag = None, None, None
    if a.exclude_flag:
        if a.exclude_flag not in A.obs:
            print(f"scanno: {a.h5ad} has no obs column {a.exclude_flag!r}. Boolean columns "
                  f"available: {[c for c in A.obs if A.obs[c].dtype == bool]}", file=sys.stderr)
            return 1
        flag = A.obs[a.exclude_flag].fillna(False).astype(bool).to_numpy()

    if a.exclude_flag and a.exclude_mode == CELL:
        # Profiled over the KEPT cells only. This is the whole of the mode: a flagged nucleus
        # cannot influence the label of the cluster it sat in, because it is not in the mean.
        M, D, counts = cluster_profile(X[~flag], y[~flag], len(cats))
        drop = unprofilable(y, ~flag, len(cats))
        excl = exclusion_record_cells(flag, y, len(cats), reason=f"obs[{a.exclude_flag!r}]")
        print(f"--exclude-flag {a.exclude_flag} (mode cell): excluding "
              f"{excl['cells_excluded']:,} flagged nuclei "
              f"({100*excl['fraction_excluded']:.1f}% of the object), 0 passengers. They keep "
              f"their place and are labelled {EXCLUDED}; nothing is deleted.")
        if drop.any():
            print(f"    {int(drop.sum())} cluster(s) had every cell flagged and cannot be "
                  f"profiled: {', '.join(cats[i] for i in np.flatnonzero(drop))}")
    else:
        M, D, counts = cluster_profile(X, y, len(cats))
        if a.exclude_flag:
            drop = cluster_flags(y, flag, len(cats), share=a.exclude_share)
            excl = exclusion_record(drop, counts, reason=f"obs[{a.exclude_flag!r}]",
                                    share=a.exclude_share)
            if not drop.any():
                print(f"--exclude-flag {a.exclude_flag}: no cluster reaches "
                      f"{100*a.exclude_share:.0f}% flagged cells; nothing excluded")
            else:
                print(f"--exclude-flag {a.exclude_flag} (mode cluster): excluding "
                      f"{len(excl['clusters_excluded'])} of {len(cats)} clusters, "
                      f"{excl['cells_excluded']:,} cells "
                      f"({100*excl['fraction_excluded']:.1f}% of the object), of which "
                      f"{int((~flag & np.isin(y, excl['clusters_excluded'])).sum()):,} carry no "
                      f"flag. They keep their place and are labelled {EXCLUDED}.")
                for i in excl["clusters_excluded"]:
                    print(f"    cluster {cats[i]}   {int(counts[i]):,} cells")

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

    # `drop` goes to BOTH: the usable-gene set is `any` over clusters, so without it here an
    # excluded cluster still decides which genes the kept ones are scored on.
    Z, usable, st = standardise(M, D, genes, store, exclude=drop)
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
                   gap_min=a.gap_min, exclude=drop)

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
    s.add_argument("--exclude-flag", metavar="OBS_COLUMN",
                   help="obs column marking cells whose CLUSTER upstream QC flagged - doublet- "
                        "or debris-dominated, say. Flagged clusters are not walked and are "
                        "labelled EXCLUDED. Nothing is deleted: the cells keep their place, and "
                        "for a fixed clustering every other cluster gets exactly the label it "
                        "would have got had these cells never been there")
    s.add_argument("--exclude-mode", choices=("cell", "cluster"), default="cell",
                   help="'cell' (default) excludes exactly the flagged nuclei: they contribute "
                        "to no cluster profile and are labelled EXCLUDED. 'cluster' excludes "
                        "whole clusters that are --exclude-share flagged, which also removes "
                        "their unflagged members and makes the excluded set depend on this "
                        "run's clustering granularity")
    s.add_argument("--exclude-share", type=float, default=0.5, metavar="F",
                   help="share of a cluster's cells that must carry the flag before the CLUSTER "
                        "counts as flagged (default 0.5, a majority)")
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

    s = sub.add_parser("agent",
                       help="a second opinion per cluster from a model you supply. Never "
                            "replaces `annotate` - it writes a separate table")
    s.add_argument("object", type=Path, help="clustered .h5ad")
    s.add_argument("--cluster-key", default="leiden_1.0")
    s.add_argument("--tree", required=True, type=Path,
                   help="declared tree, JSON. The PREFERRED vocabulary, not a closed one: the "
                        "model answers freely and the reply is resolved onto a node via the "
                        "tree's synonym patterns, onto a standard cell type name, or kept "
                        "verbatim as a proposal")
    s.add_argument("--db", type=Path,
                   help="CellMarker db. Supplies three things: what the corpus knows about each "
                        "cluster's genes (shown to the model), the standard cell type names and "
                        "Cell Ontology ids for this tissue (offered to the model), and the "
                        "corpus call itself (used for comparison and NEVER shown to it)")
    s.add_argument("--species", required=True)
    s.add_argument("--tissue", required=True)
    s.add_argument("--assay", default="sc", choices=["sc", "sn"])
    g = s.add_mutually_exclusive_group()
    g.add_argument("--provider", default="openai", choices=sorted(_http_presets()),
                   help="hosted API; the key comes from the environment and is never stored")
    g.add_argument("--command", help="BRING YOUR OWN AGENT: any command that reads the prompt "
                                     "on stdin and writes the reply on stdout, e.g. "
                                     "--command 'ollama run llama3'")
    s.add_argument("--model")
    s.add_argument("--url", help="override the endpoint for an OpenAI-compatible server")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--web", action="store_true",
                   help="let a hosted provider search the web. Refused for a provider this "
                        "package has no tool block for, rather than silently ignored. A "
                        "--command agent brings its own tools and this flag does not apply")
    s.add_argument("--top-n", type=int, default=30, dest="top_n",
                   help="genes shown per cluster (default 30)")
    s.add_argument("--votes", type=int, default=1,
                   help="ask each cluster this many times and report the agreement rate. A "
                        "label that changes between identical calls is a finding")
    s.add_argument("--out", type=Path, default=Path("scanno_agent.csv"))
    s.set_defaults(fn=_agent)

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
