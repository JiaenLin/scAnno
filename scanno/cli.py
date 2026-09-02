"""The `scanno` command.

Deliberately small. There is no driver, no report and no task graph, so the CLI exposes what
actually exists rather than pretending to a pipeline: `annotate`, `calibrate`, `panel`,
`store-info`, `resolution`, `agent`, `selftest`.

No version is stated here, deliberately. This docstring used to open by naming one, kept naming
the first release for three of them, and the list of subcommands beneath it went stale in the
same breath - a version restated in prose is a version that drifts, and the remedy that worked
for `pyproject.toml` was to remove the declaration rather than to keep checking it.
`tests/test_version.py` §6 guards against reintroducing one here.

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


def _rescue(a):
    """Targeted rescue: a label a unit lacks, looked for in that unit alone."""
    import csv as _csv
    import datetime as _dt

    import numpy as np
    try:
        import anndata as ad
    except ImportError:
        print("scanno: anndata is required for `rescue`", file=sys.stderr)
        return 1
    from . import __version__
    from .emit import annotate_rescue, write_h5ad
    from .rescue import document, rescue, summarise

    tree = _json.loads(Path(a.tree).read_text(encoding="utf-8")) if a.tree else None

    objs, labels, sweep, clus, units = {}, {}, {}, {}, []
    for path in a.h5ad:
        A = ad.read_h5ad(path)
        u = (str(A.obs[a.unit_key].iloc[0]) if a.unit_key and a.unit_key in A.obs
             else Path(path).stem.split(".")[0])
        if u in objs:
            print(f"scanno: two objects claim unit {u!r}; --unit-key must name a column that "
                  f"is CONSTANT within each file", file=sys.stderr)
            return 1
        if a.label_key not in A.obs:
            print(f"scanno: {path} has no obs column {a.label_key!r}", file=sys.stderr)
            return 1
        # THE RUNGS ARE READ OFF THE OBJECT, not requested. A caller who annotated four
        # resolutions and asks for six would otherwise get a silent four-rung search.
        found = {}
        for c in A.obs.columns:
            if not str(c).startswith(a.sweep_prefix):
                continue
            tag = str(c)[len(a.sweep_prefix):]
            try:
                r = float(tag.replace("p", "."))
            except ValueError:
                continue
            k = f"{a.cluster_prefix}{tag}"
            if k in A.obs:
                found[r] = (c, k)
        if len(found) < 2:
            print(f"scanno: {path} carries fewer than two rungs under {a.sweep_prefix!r} with "
                  f"matching {a.cluster_prefix!r} columns. Annotate it at several resolutions "
                  f"first: `scanno annotate --cluster-key ... --cluster-key ...`", file=sys.stderr)
            return REFUSE
        objs[u] = (path, A)
        labels[u] = np.asarray(A.obs[a.label_key].astype(str))
        sweep[u] = {r: np.asarray(A.obs[c].astype(str)) for r, (c, _k) in found.items()}
        clus[u] = {r: np.asarray(A.obs[k].astype(str)) for r, (_c, k) in found.items()}
        units.append(u)

    common = sorted(set.intersection(*(set(sweep[u]) for u in units)))
    lo = a.from_resolution if a.from_resolution is not None else common[0]
    hi = a.to_resolution if a.to_resolution is not None else common[-1]
    rungs = [r for r in common if lo <= r <= hi]
    if len(rungs) < 2:
        print(f"scanno: only {len(rungs)} rung(s) between {lo} and {hi}; a search needs at "
              f"least two. Rungs present in every object: {common}", file=sys.stderr)
        return REFUSE
    print(f"{len(units)} unit(s); searching rungs {', '.join(str(r) for r in rungs)}")
    print(f"  label being corrected: obs[{a.label_key!r}]")

    new, origin, rec = rescue(labels, sweep, clus, rungs, tree=tree)
    summ = summarise(labels, new)

    print("")
    print(f"{len(rec['trigger'])} imbalanced label(s) -> {rec['n_targets']} targeted searches")
    for lab, v in sorted(rec["trigger"].items()):
        print(f"    {lab:<40} carried by {len(v['with'])}, searched in {len(v['without'])}")
    print("")
    print(f"found {rec['n_found']}, not found {rec['n_not_found']}, "
          f"{rec['n_renamed']:,} cell(s) renamed")
    for m in rec["moved"]:
        src = ", ".join(f"{k.split('/')[-1]} {v}" for k, v in
                        sorted(m["from"].items(), key=lambda kv: -kv[1]))
        print(f"    {m['label']:<34} {m['unit']:<14} rung {m['rung']:<6} "
              f"{m['n_renamed']:>6,} cell(s)   from {src}")
    if rec["not_found"]:
        print("")
        print("  nothing found - and whether that means anything depends on whether the "
              "finest\n  clustering reached could have HELD the population:")
        for m in rec["not_found"]:
            print(f"    {m['label']:<34} {m['unit']:<14} "
                  f"would be {m['expected_cells']:>6.0f} cell(s), finest cluster "
                  f"{m['mean_cluster_finest']:>6.0f}   "
                  f"{'a real absence' if m['could_form'] else 'UNDECIDED - out of reach'}")

    out_key = a.out_key or f"{a.label_key}_rescued"
    if a.out_dir:
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        for u in units:
            path, A = objs[u]
            info = annotate_rescue(A, new[u], origin[u], rec, key=out_key)
            dest = Path(a.out_dir) / f"{Path(path).stem}.rescued.h5ad"
            write_h5ad(A, dest)
            print(f"wrote {dest}   +obs[{info['key']!r}] and obs[{info['origin_key']!r}]")
    if a.out_table:
        Path(a.out_table).parent.mkdir(parents=True, exist_ok=True)
        cols = ["unit", "label", "n_before", "n_after", "n_delta", "pct_before", "pct_after",
                "pct_delta", "is_sentinel"]
        with open(a.out_table, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in summ["rows"]:
                w.writerow({k: r[k] for k in cols})
        print(f"wrote {a.out_table}   {len(summ['rows'])} unit x label row(s) that change")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(_json.dumps(rec, indent=1, default=str), encoding="utf-8")
        print(f"wrote {a.out}")
    if a.out_report:
        Path(a.out_report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_report).write_text(document(
            {"record": rec, "summary": summ, "label_key": a.label_key, "version": __version__,
             "generated": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}),
            encoding="utf-8")
        print(f"wrote {a.out_report}")
    return 0


def _annotate(a):
    """Label the clusters of one object. The main command."""
    import numpy as np
    from .classify import classify
    from .corpus import load_assertions
    from .exclude import EXCLUDED, exclusion_record_cells, unprofilable
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

    # --- the decided scope, applied to the tree BEFORE the object is read ---
    #
    # `--scope` makes the annotation fully driven by the vote instead of by a tree somebody
    # sealed by hand and passed as --tree. Two things follow from reading the DECISION rather
    # than its rendering:
    #
    #   - the seal cannot drift from the vote. `--tree declared.json --scope scope.json` is one
    #     statement; `--tree sealed.json` is two files that agree only as long as nobody edits
    #     either. Sealing here is idempotent, so an already-sealed tree plus its own scope is
    #     still valid input and produces the same walk.
    #   - FORCE becomes actionable. Which nodes carry it is in the vote and nowhere else: a
    #     sealed tree cannot express "this node keeps its children AND nothing may stop on it",
    #     because that is a statement about annotation, not about taxonomy.
    #
    # The walk is untouched either way. Only the tree handed to it changes, and what happens to
    # a cluster AFTER it returns.
    scope, force_paths = None, []
    if getattr(a, "scope", None):
        from .force import check_scope, force_nodes, scope_verdicts, sealed_nodes
        from .scope import seal_tree
        scope = json.loads(Path(a.scope).read_text(encoding="utf-8"))
        try:
            verdicts = scope_verdicts(scope)
        except ValueError as e:
            print(f"scanno: REFUSE - --scope {a.scope} is {e}", file=sys.stderr)
            return REFUSE
        problems = check_scope(scope, tree)
        if problems:
            print(f"scanno: REFUSE - --scope {a.scope} cannot be walked against --tree "
                  f"{a.tree}:", file=sys.stderr)
            for p in problems:
                print(f"        - {p}", file=sys.stderr)
            return REFUSE
        tree, removed = seal_tree(tree, verdicts)
        force_paths = force_nodes(verdicts)
        n_lost = sum(sum(d.values()) for d in (scope.get("removed_labels") or {}).values())
        print(f"scope {a.scope}: voted over {scope.get('n_samples', '?')} sample(s)")
        print(f"  SEAL   {len(sealed_nodes(verdicts))} node(s) become leaves"
              + (f", removing {n_lost:,} cell(s)' worth of subtype label" if n_lost else "")
              + (f": {', '.join(sorted(removed))}" if removed
                 else " (already applied to --tree)"))
        print(f"  FORCE  {len(force_paths)} node(s) keep their children and may not be "
              f"terminal: {', '.join(force_paths) or '(none)'}")

    # --- the independent L1 tree, checked BEFORE the object is read ---
    #
    # A depth-1 tree is the only thing that can produce an L1 column, and the check is on the
    # DECLARATION so that all ten samples of a cohort get the same verdict. It runs here, ahead
    # of the h5ad read and the background, because a refusal after the expensive part is a
    # refusal the caller pays for twice.
    l1_tree = None
    if getattr(a, "l1_tree", None):
        from .scope import root_child_diff, tree_depth
        l1_tree = json.loads(Path(a.l1_tree).read_text(encoding="utf-8"))
        d = tree_depth(l1_tree)
        if d != 1:
            print(f"scanno: REFUSE - --l1-tree {a.l1_tree} is a depth-{d} tree.\n"
                  f"        The L1 column holds one level and nothing below it. Build the tree "
                  f"with\n"
                  f"        `scanno scope --out-l1-tree`, which emits exactly this.",
                  file=sys.stderr)
            return REFUSE
        only_deep, only_l1 = root_child_diff(tree, l1_tree)
        if only_deep or only_l1:
            print("  REVIEW  the two trees do not agree at the ROOT, so the L1 column and the "
                  "label\n          column's first level are not the same taxonomy:")
            if only_deep:
                print(f"          only in --tree:    {', '.join(only_deep)}")
            if only_l1:
                print(f"          only in --l1-tree: {', '.join(only_l1)}")

    A = ad.read_h5ad(a.h5ad)
    # ONE key or many. `a.cluster_key` is collapsed to the FIRST here and never read as a list
    # again, so every line below this is the single-resolution run it always was - which is the
    # whole of "adding resolutions changes nothing that was already written".
    sweep_keys = list(a.cluster_key) if isinstance(a.cluster_key, (list, tuple)) \
        else [a.cluster_key]
    seen_k = set()
    sweep_keys = [k for k in sweep_keys if not (k in seen_k or seen_k.add(k))]
    a.cluster_key = sweep_keys[0]
    missing_k = [k for k in sweep_keys if k not in A.obs]
    if missing_k:
        print(f"scanno: {a.h5ad} has no obs column(s) {', '.join(map(repr, missing_k))}. "
              f"Available: {', '.join(list(A.obs.columns)[:12])}", file=sys.stderr)
        return 1
    src = A.raw if (a.use_raw and A.raw is not None) else A
    # WHICH NAMES THE CORPUS WILL BE MATCHED ON. A corpus is keyed by SYMBOL, and an object is
    # very often keyed by accession with the symbols beside it in `var` - which is the correct
    # way round for the object, since symbols are not unique. Reading `var_names` regardless
    # produced 0 overlapping genes and 100% UNRESOLVED, silently: every marker panel came out
    # empty, every node was dropped, and the run exited 0 with a full table of UNRESOLVED.
    # `scanno agent` had used the symbol column since it was written; this path had not.
    gene_key = a.gene_key
    if gene_key is None and "gene_symbol" in src.var:
        gene_key = "gene_symbol"
    if gene_key and gene_key not in src.var:
        print(f"scanno: {a.h5ad} has no var column {gene_key!r}. Columns: "
              f"{', '.join(map(str, src.var.columns))}", file=sys.stderr)
        return 1
    X = src.X
    genes = np.array([str(v).upper() for v in
                      (src.var[gene_key] if gene_key else src.var_names)])
    print(f"gene names from {'var[' + repr(gene_key) + ']' if gene_key else 'var_names'}"
          f"   e.g. {', '.join(genes[:3])}")

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
    # ONE path, and it is the flag itself. The flagged nuclei are dropped from the PROFILE -
    # they contribute to no cluster's mean and to no detection rate - and each is labelled
    # EXCLUDED individually. scAnno does not decide which nuclei are technical and has no code
    # that turns the flag into a different set of cells: no share, no threshold, no dependence
    # on the clustering. A cluster-share mode existed until 0.3.0 and was REMOVED rather than
    # defaulted off, because it excluded nuclei upstream QC had passed. See scanno/exclude.py.
    # Where the withheld set comes from: `--no-exclude`, then `--exclude-flag`, then the object's
    # own upstream declaration. Detection keys on a DECLARATION and never on a column name - an
    # object carrying `cluster_FLAG` and no declaration gets nothing, because scAnno does not
    # know what that column is and guessing is the failure mode. See scanno/upstream.py.
    from . import upstream as up

    decision = up.decide(A, explicit=a.exclude_flag, disabled=a.no_exclude)
    if decision.refuse:
        print(f"scanno: REFUSE - {decision.refuse}", file=sys.stderr)
        return REFUSE
    for line in decision.lines:
        print(line)
    flag = decision.mask
    drop, excl = None, None

    if flag is not None:
        # Profiled over the KEPT cells only. This is the whole of it: a flagged nucleus cannot
        # influence the label of the cluster it sat in, because it is not in the mean.
        M, D, counts = cluster_profile(X[~flag], y[~flag], len(cats))
        drop = unprofilable(y, ~flag, len(cats))
        excl = exclusion_record_cells(flag, y, len(cats),
                                      reason=f"{decision.source}:{decision.column}")
        print(f"    withholding {excl['cells_excluded']:,} nuclei "
              f"({100*excl['fraction_excluded']:.1f}% of the object), 0 passengers - exactly "
              f"the flag and nothing else. They keep their place and are labelled {EXCLUDED}; "
              f"nothing is deleted.")
        print(f"    mask digest {excl['flag_digest']}  "
              f"(fingerprint of the exact set that ran, for the caller's record)")
        if drop.any():
            print(f"    {int(drop.sum())} cluster(s) had every cell flagged and cannot be "
                  f"profiled: {', '.join(cats[i] for i in np.flatnonzero(drop))}")
    else:
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
        # The guard that exists for exactly this and was never called here. Without it an
        # accession-keyed object against a symbol-keyed corpus returns UNRESOLVED for every
        # cluster and exits 0 - which reads as a finding about the data rather than a naming
        # mismatch, and is the single most expensive way for this tool to be wrong.
        from .corpus import GeneSpaceMismatch, check_gene_space
        try:
            check_gene_space(asr, genes)
        except GeneSpaceMismatch as e:
            print(str(e), file=sys.stderr)      # it already says "scanno: REFUSE - ..."
            return REFUSE
    tree["genes"] = store.genes
    res = classify(Z, usable, tree, store=None if asr else store, assertions=asr,
                   gap_min=a.gap_min, exclude=drop)

    # --- FORCE: nothing terminates on a node the cohort agreed to split ---
    #
    # POST-WALK, on the rows `classify` already returned. The FIRST child a stranded cluster
    # moves to is `trace[-1]["top"]` - the argmax the unchanged walk recorded at the very node it
    # stopped on. If that child is itself internal the push is not finished, so it continues:
    # `node_scorer` scores the next node with the SAME weights over the SAME Z the walk used, and
    # the descent repeats until a leaf. No bar is moved and classify.py is not touched. Applied
    # to `res` ONLY: `res_l1` below is the independent L1 and no verdict of the scope reaches it.
    force_rec = None
    _scorer = None
    if force_paths:
        from .force import BY_FORCE, apply_force, bare_force, format_force, internal_terminals
        from .step import node_scorer
        _scorer = node_scorer(Z, usable, tree, store=None if asr else store, assertions=asr)
        res, force_rec = apply_force(
            res, force_paths, counts=counts, tree=tree, scorer=_scorer)
        # Both post-conditions read the FINISHED rows rather than trusting the function that
        # produced them. A bare FORCE-node label is the original defect; a FORCED row on any
        # other internal node is a recursion that stopped short, which delivers the same kind of
        # compartment name one level down.
        stuck = bare_force(res, force_paths)
        inner = internal_terminals(res, tree)
        seen = {s["cluster"] for s in stuck}
        stuck += [t for t in inner if t["assignment"] == BY_FORCE and t["cluster"] not in seen]
        if stuck:
            print(f"scanno: REFUSE - {len(stuck)} cluster(s) still terminate on an internal node "
                  f"after reassignment, so this object would deliver a compartment name where "
                  f"the\n        scope says a subtype belongs:", file=sys.stderr)
            for s in sorted(stuck, key=lambda s: s["cluster"]):
                print(f"        - cluster {s['cluster']} on {s['node']}", file=sys.stderr)
            for line in format_force(force_rec, gap_min=a.gap_min):
                print(f"        {line.strip()}", file=sys.stderr)
            print("        Nothing was written. Seal that node in the scope, or give its "
                  "children corpus support.", file=sys.stderr)
            return REFUSE
        # Clusters the WALK truncated on a node the scope left open. Not a defect and not
        # refused - the scope's own verdict is that stopping there is admissible, and refusing
        # would replace truncation with abstention, which classify.py exists not to do - but it
        # is the number behind "nothing terminates on an internal node", so it is measured and
        # printed rather than assumed to be zero.
        open_stop = [t for t in inner if t["assignment"] != BY_FORCE]
        n_open = sum(counts[t["cluster"]] for t in open_stop)
        print(f"  terminating on an internal node: {len(open_stop)} cluster(s) / "
              f"{n_open:,.0f} cell(s)"
              + (" - none, so every delivered label is a leaf of the scoped tree"
                 if not open_stop else ", each on a node the scope did not force:"))
        for t in sorted(open_stop, key=lambda t: -counts[t["cluster"]]):
            print(f"      cluster {t['cluster']:>6}  {counts[t['cluster']]:>9,.0f} cell(s)  "
                  f"{t['node']}   (children kept: {t['children']})")

    # THE SECOND WALK. The same `classify`, the same Z, the same usable-gene set, the same
    # background and the same bar - only the tree differs. Nothing about the walk is
    # parameterised for this and nothing in classify.py was touched: an independent L1 is a
    # second CALL, not a second mode. `drop` goes here too, or an unprofilable cluster would be
    # EXCLUDED in one column and labelled in the other.
    res_l1 = None
    if l1_tree is not None:
        l1_tree["genes"] = store.genes
        res_l1 = classify(Z, usable, l1_tree, store=None if asr else store, assertions=asr,
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

    if force_rec is not None:
        from .force import format_force
        print("")
        for line in format_force(force_rec, gap_min=a.gap_min):
            print(line)

    if res_l1 is not None:
        # Per CLUSTER, printed whether or not an object is written, because a run that only
        # prints is still a run somebody reads. The comparison is against the DERIVED L1 -
        # `path[:1]` - which is what the L1 column would have held without --l1-tree.
        diff = [(cats[r["cluster"]], p["path"].split("/")[0], r["path"])
                for p, r in zip(res, res_l1) if p["path"].split("/")[0] != r["path"]]
        print("")
        print(f"independent L1 from {a.l1_tree}: "
              f"{len({r['path'] for r in res_l1})} label(s) over {len(res_l1)} clusters")
        if diff:
            print(f"  REVIEW  {len(diff)} cluster(s) differ from the derived L1 (path[:1]):")
            for cl, der, ind in diff[:10]:
                print(f"    {cl:>10}   derived {der:<28} independent {ind}")
        else:
            print("  every cluster agrees with the derived L1 (path[:1]) - measured, not assumed")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with Path(a.out).open("w", encoding="utf-8") as fh:
            head = ["cluster", "n_cells", "label", "path", "depth", "gap"]
            # `assignment` rides beside `gap` and not somewhere else on purpose: the two are read
            # together or neither is read at all. `gap` is the margin under both verdicts, and
            # `assignment` is the only thing that says whether that margin cleared the bar.
            if force_rec is not None:
                head += ["assignment", "force_depth"]
            if res_l1 is not None:
                head += ["l1_independent", "l1_gap"]
            fh.write("\t".join(head) + "\n")
            for i, r in enumerate(res):
                c = r["cluster"]
                row = [cats[c], f"{counts[c]:.0f}", r["label"], r["path"],
                       str(r["depth"]), f"{r['gap']:.4f}"]
                if force_rec is not None:
                    row += [str(r.get("assignment", "")), str(r.get("force_depth", 0))]
                if res_l1 is not None:
                    row += [res_l1[i]["path"], f"{res_l1[i]['gap']:.4f}"]
                fh.write("\t".join(row) + "\n")
        print(f"wrote {a.out}")

    # --- RESOLVED: a second label column with no holes in it ---
    #
    # Runs AFTER the FORCE post-conditions above, never before: those refuse a run that would
    # deliver a compartment name where the scope says a subtype belongs, and that refusal must
    # fire on the principled columns rather than be masked by a resolved column that has quietly
    # filled the same gap.
    #
    # Additive and reversible. `<prefix>_cell_type` and `<prefix>_path` are untouched, so the
    # decision NOT to guess is still in the object; `<prefix>_resolved*` sits beside them for the
    # consumers that need a column with no holes - a composition table, a viewer colour-by, a
    # label handed to a semi-supervised model - and `<prefix>_resolved_origin` says, per cell,
    # whether its leaf was reached or assigned.
    resolve_rec = None
    if a.resolve:
        from .force import format_resolved, resolve_to_leaf
        from .step import node_scorer
        if _scorer is None:
            _scorer = node_scorer(Z, usable, tree, store=None if asr else store, assertions=asr)
        res, resolve_rec = resolve_to_leaf(res, tree=tree, scorer=_scorer, counts=counts)
        print("")
        for line in format_resolved(resolve_rec):
            print(line)
        # THE INDEPENDENT L1 IS RESOLVED SEPARATELY, against its OWN depth-1 tree. Resolving it
        # from the deep tree would push an L1 cell to a subtype, which is not an L1 answer; and
        # the scope's verdicts must not reach this walk at all. Depth 1 means the root's argmax
        # is already a leaf, so the push is one step and the scorer is never needed - it is
        # passed anyway so a deeper --l1-tree would descend rather than stop short.
        if res_l1 is not None:
            res_l1, resolve_l1_rec = resolve_to_leaf(
                res_l1, tree=l1_tree, counts=counts,
                scorer=node_scorer(Z, usable, l1_tree,
                                   store=None if asr else store, assertions=asr))
            print("")
            print("  independent L1:")
            for line in format_resolved(resolve_l1_rec):
                print("  " + line)

    # The annotated object. Everything above is per CLUSTER; this is the only place the labels
    # become per CELL, which is the form every consumer of an annotation actually wants.
    # Imported here, not inside `if a.out_h5ad`, because --report uses it too: a run with
    # --report and no --out-h5ad raised UnboundLocalError after every library had been
    # annotated. An import scoped to one branch and used in another is invisible until the
    # branch that does not import it runs.
    # --- THE SWEEP: the SAME walk at every further resolution ----------------------------
    #
    # A per-sample clustering cannot separate a population too small to form a cluster in it,
    # and neither can a joint one at a single granularity. The same cells partitioned finer
    # reach a size a cluster can hold; partitioned coarser they do not. So the granularity is
    # not a setting the run happens to have - it is a hypothesis the run can TEST, by walking
    # every resolution and reading how much of the sweep agrees about each cell.
    #
    # Nothing here is a second MODE. It is the same `classify` over the same Z-machinery against
    # the same tree, the same scope, the same gene background and the same bar - only `y`
    # differs, which is what a resolution IS. The store is loaded once above and shared, so the
    # sweep's resolutions are comparable to each other by construction rather than by hope: a
    # background rebuilt per resolution would make every column a different question.
    def _res_tag(key):
        """`leiden_1p0` -> `1p0`. The tag the sweep column is named by, as the key spells it."""
        t = str(key).rsplit("_", 1)[-1]
        t = t[1:] if t[:1] == "r" else t
        try:
            float(t.replace("p", "."))
            return t
        except ValueError:
            return "".join(ch for ch in str(key) if ch.isalnum())

    sweep = [(_res_tag(a.cluster_key), res, y)]
    if len(sweep_keys) > 1:
        from .force import apply_force, resolve_to_leaf
        from .step import node_scorer
        print("")
        print(f"sweep: {len(sweep_keys)} resolutions, one walk each, same tree / scope / "
              f"background / bar")
        for k in sweep_keys[1:]:
            lab_k = A.obs[k].astype(str).values
            cats_k = sorted(set(lab_k), key=lambda t: (len(t), t))
            y_k = np.array([cats_k.index(v) for v in lab_k])
            if flag is not None:
                M_k, D_k, cn_k = cluster_profile(X[~flag], y_k[~flag], len(cats_k))
                drop_k = unprofilable(y_k, ~flag, len(cats_k))
            else:
                M_k, D_k, cn_k = cluster_profile(X, y_k, len(cats_k))
                drop_k = None
            Z_k, usable_k, _st_k = standardise(M_k, D_k, genes, store, exclude=drop_k)
            res_k = classify(Z_k, usable_k, tree, store=None if asr else store, assertions=asr,
                             gap_min=a.gap_min, exclude=drop_k)
            sc_k = None
            if force_paths:
                sc_k = node_scorer(Z_k, usable_k, tree,
                                   store=None if asr else store, assertions=asr)
                res_k, _ = apply_force(res_k, force_paths, counts=cn_k, tree=tree, scorer=sc_k)
            if a.resolve:
                if sc_k is None:
                    sc_k = node_scorer(Z_k, usable_k, tree,
                                       store=None if asr else store, assertions=asr)
                res_k, _ = resolve_to_leaf(res_k, tree=tree, scorer=sc_k, counts=cn_k)
            n_un_k = sum(1 for r in res_k if r["path"] == "UNRESOLVED")
            print(f"    {k:<20} {len(cats_k):>4} clusters   "
                  f"{len({r['path'] for r in res_k}):>3} label(s)   "
                  f"{n_un_k} UNRESOLVED")
            sweep.append((_res_tag(k), res_k, y_k))

    from .emit import (annotate_obs, force_provenance, trace_provenance, format_independent_l1,
                       format_plain_labels, format_readiness, format_reindex, independent_l1,
                       lab_readiness, reindex_by_symbol, write_h5ad)

    def write_columns():
        """Every obs column this run contributes, in one place so both branches write the same.

        --report and --out-h5ad each annotate the object, and until this was factored out only
        the --out-h5ad branch would have carried the independent L1 - a report describing columns
        the object it points at does not have.
        """
        w = annotate_obs(A, res, y, flag=flag, prefix=a.label_prefix,
                         support=support or None, suffix=a.label_suffix,
                         assignment=force_rec is not None,
                         resolved=resolve_rec is not None)
        # ALWAYS, not only when something was forced. The walk computes what every label beat and
        # by how much, at every step, and it was thrown away - a run could say a cluster is
        # `Neural, gap 0.64` and nothing about what Neural beat, so the reason for a call had to
        # be reconstructed from a marker table rather than read from the run.
        _tk = trace_provenance(A, res, prefix=a.label_prefix, suffix=a.label_suffix)
        print(f"    why each cluster got its label in uns[{_tk!r}]: every node scored, the "
              f"winner, the runner-up and the margin between them")
        if force_rec is not None:
            # WRITTEN EVEN WHEN NOTHING WAS FORCED. An all-`gap` column on a scoped run is the
            # statement "this scope was honoured and stranded nobody here", which is a result;
            # an absent column is indistinguishable from a run that never saw a scope.
            key = force_provenance(A, force_rec, prefix=a.label_prefix,
                                   suffix=a.label_suffix, scope=str(a.scope))
            print("")
            print(f"  {a.label_prefix}_assignment{a.label_suffix}: how each cell was assigned - "
                  f"{force_rec['clusters_by_assignment']} by cluster")
            print(f"  {a.label_prefix}_force_depth{a.label_suffix}: how many forced steps stand "
                  f"behind the label - {force_rec['clusters_by_force_depth'] or '{}'} by cluster")
            print(f"    provenance in uns[{key!r}]: the FORCE node, the leaf chosen and the "
                  f"margin of EACH step, per cluster")
        if res_l1 is not None:
            col, rec = independent_l1(A, res_l1, y, flag=flag, suffix=a.label_suffix,
                                      tree=str(a.l1_tree),
                                      resolved=resolve_rec is not None)
            print("")
            for line in format_independent_l1(rec):
                print(line)
            if col not in w:
                w.append(col)

        # THE SWEEP COLUMNS AND THE VOTE OVER THEM. Inside `write_columns` and not beside the
        # walk above, because this is the function both --out-h5ad and --report go through, and
        # a sweep written in only one of them is a report describing columns its object lacks -
        # the defect this function was factored out to stop.
        if len(sweep) > 1:
            from .emit import sweep_agreement_column, sweep_path
            from .resolution import sweep_agreement
            cols_by_tag = {}
            for tag, res_k, y_k in sweep:
                key = sweep_path(A, res_k, y_k, flag=flag, prefix=a.label_prefix,
                                 suffix=a.label_suffix, tag=tag,
                                 resolved=resolve_rec is not None)
                cols_by_tag[float(tag.replace("p", "."))] = key
                w.append(key)
            # AGAINST THE LABEL THIS RUN DELIVERED, which is the first --cluster-key's own
            # column. The sweep describes that annotation; it does not vote a competing one.
            prim = cols_by_tag[float(_res_tag(a.cluster_key).replace("p", "."))]
            labs_by_res = {r: np.asarray(A.obs[c].astype(str))
                           for r, c in sorted(cols_by_tag.items())}
            agree = sweep_agreement(labs_by_res, np.asarray(A.obs[prim].astype(str)))
            info = sweep_agreement_column(
                A, agree, {
                    "resolutions": [float(r) for r in sorted(cols_by_tag)],
                    "columns": [cols_by_tag[r] for r in sorted(cols_by_tag)],
                    "reference": prim, "cluster_keys": list(sweep_keys),
                    "store_digest": str(getattr(store, "digest", "")),
                    "tree": str(a.tree), "scope": str(a.scope or ""),
                    "rule": "share of the sweep whose label equals the delivered one. "
                            "REPORTED, never acted on: no label is chosen or changed here.",
                },
                prefix=a.label_prefix, suffix=a.label_suffix)
            w.append(info["key"])
            print("")
            print(f"  {info['key']}: how much of the {len(sweep)}-resolution sweep agrees with "
                  f"{prim}")
            print(f"    {100*float((agree == 1.0).mean()):.1f}% of cells agree at EVERY "
                  f"resolution; median {float(np.median(agree)):.2f}; "
                  f"{100*float((agree <= 0.5).mean()):.1f}% at half the sweep or less")
            # WHICH resolution the sweep itself prefers - REPORTED, never applied.
            try:
                from .resolution import pick_resolution as _pick
                _p = _pick(labs_by_res, tree=tree,
                           groups=(np.asarray(A.obs[a.sample_key].astype(str))
                                   if a.sample_key and a.sample_key in A.obs else None),
                           depths=(1,))
                print(f"    the sweep's own preferred resolution is {_p['pick']} "
                      f"(decided by {_p['reason']}); this run used {a.cluster_key}. "
                      f"REPORTED, not applied.")
            except Exception as _e:                      # noqa: BLE001 - reporting only
                print(f"    could not score the sweep: {_e}")
        return w

    if a.out_h5ad:
        written = write_columns()
        # The WRITTEN object is keyed by symbol, because that is the name a reader looks a
        # gene up by. Accessions are the right index for an object being computed on - symbols
        # are not unique - so the change is made here, at the boundary, and never silently: the
        # accession is preserved and every duplicated symbol keeps its own row.
        if a.out_gene_key:
            rep = reindex_by_symbol(A, key=a.out_gene_key)
            for line in format_reindex(rep):
                print(line)
        for line in format_plain_labels(write_h5ad(A, a.out_h5ad, compression="gzip")):
            print(line)
        print("")
        print(f"wrote {a.out_h5ad}   {A.n_obs:,} x {A.n_vars:,}")
        print(f"  obs columns added: {', '.join(written)}")
        print("")
        # X, var and obsm are the object's, untouched - the annotation is added, never a new
        # object built around it. An embedding or a symbol column that came in comes out.
        print("  what a viewer will find in it")
        label_key = (f"{a.label_prefix}_cell_type" if not a.label_suffix
                     else f"{a.label_prefix}_label{a.label_suffix}")
        checks = lab_readiness(A, label_key)
        for line in format_readiness(checks):
            print(line)
        if any(lvl == "missing" for lvl, _ in checks):
            print("")
            print("  MISSING items are things scAnno cannot supply and will not invent. The "
                  "object\n  is written either way; add them upstream and re-run.")

    # The report. Assembled from the run that produced the annotation rather than recomputed:
    # a report that derives its own numbers can disagree with the run it describes, and nothing
    # on the page would say which was right.
    if a.report:
        from . import report as rp
        from . import __version__
        label_key = (f"{a.label_prefix}_cell_type" if not a.label_suffix
                     else f"{a.label_prefix}_label{a.label_suffix}")
        if label_key not in A.obs:
            write_columns()
        doc = rp.collect(
            A, res, cats, y, label_key=label_key, decision=decision, support=support or None,
            store_digest=getattr(store, "digest", ""), tree_path=a.tree, db_path=a.db or "",
            species=a.species, tissue=a.tissue, cluster_key=a.cluster_key,
            sample_key=a.sample_key, condition_key=a.condition_key,
            gap_min=a.gap_min, weights=src_txt, background=bg, stats=st, version=__version__)
        # The panels the classifier actually scored on, and the SAME normalised matrix it read -
        # not A.X, which may be raw counts. A dotplot drawn from a different matrix than the one
        # behind the call is a picture of something else.
        panels = rp.marker_panels(asr, tree.get("patterns", {}),
                                  [e["label"] for e in doc["composition_l1"]]) if asr else None
        figs = rp.draw(doc, A, label_key, X=X, genes=genes, markers=panels)
        html, js = rp.write(a.report, doc, figs)
        print("")
        print(f"wrote {html}")
        print(f"      {js}   every number in the document, machine-readable")
        if doc.get("defects"):
            print(f"  {len(doc['defects'])} defect(s) counted on the report's own front page")
    elif not a.out:
        print("")
        print("scanno: nothing was written. The labels above exist only in this output.\n"
              "        --out-h5ad PATH  writes the object with the annotation per CELL\n"
              "        --out PATH       writes the per-CLUSTER table as TSV")
    return 0


def _cluster(a):
    """Step 1: cluster, and select nothing."""
    try:
        import anndata as ad
        import scanpy  # noqa: F401
    except ImportError:
        print("scanno: cluster needs anndata + scanpy.  pip install -e '.[run]'",
              file=sys.stderr)
        return 1
    from .cluster import cluster, parse_resolutions, split

    try:
        res = parse_resolutions(a.resolutions)
    except ValueError as e:
        print(f"scanno: {e}", file=sys.stderr)
        return 1
    if not res:
        print("scanno: REFUSE - no resolutions to compute.", file=sys.stderr)
        return REFUSE

    inputs = list(a.h5ad)
    if len(inputs) > 1 and a.out:
        print("scanno: REFUSE - --out names ONE file and you gave "
              f"{len(inputs)} inputs. Use --out-dir; each object is written as "
              "<stem>_clustered.h5ad, so the pieces cannot silently overwrite each other.",
              file=sys.stderr)
        return REFUSE
    if len(inputs) > 1 and a.split_by:
        print("scanno: REFUSE - --split-by splits ONE object into groups; with several inputs "
              "they are already separate. Pass the files, or pass one object and --split-by.",
              file=sys.stderr)
        return REFUSE
    if len(inputs) == 1 and not a.split_by and not a.out:
        print("scanno: REFUSE - one object and no --split-by needs --out.", file=sys.stderr)
        return REFUSE

    print(f"resolutions: {', '.join(str(r) for r in res)}   seed {a.seed}")
    print("every resolution is KEPT and none is chosen; no gene class is excluded from "
          "selection")
    if len(inputs) > 1:
        # Several objects is the ordinary shape here: upstream QC delivers one file per library
        # and clustering them separately is the point, not a convenience. Taking them all in one
        # invocation is what keeps a whole cohort a single command rather than a shell loop.
        print(f"{len(inputs)} objects, each clustered INDEPENDENTLY - no shared variable genes, "
              f"no joint embedding, no batch key")

    out_dir = Path(a.out_dir) if a.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for src in inputs:
        A = ad.read_h5ad(src)
        pieces = [(None, A)]
        if a.split_by:
            if a.split_by not in A.obs:
                print(f"scanno: {src} has no obs column {a.split_by!r}. Available: "
                      f"{', '.join(list(A.obs.columns)[:12])}", file=sys.stderr)
                return 1
            pieces = list(split(A, a.split_by))
            print(f"--split-by {a.split_by}: {len(pieces)} groups, each clustered "
                  f"INDEPENDENTLY - no shared variable genes, no joint embedding, no batch key")
            if "scqc" in A.uns:
                print("    the upstream declaration is dropped from each piece: it describes "
                      "the cohort,\n    so carrying it onto a subset would fail verification "
                      "downstream and read as\n    tampering. The flag column travels with the "
                      "cells - name it with --exclude-flag.")
        else:
            print(f"\n{Path(src).name}: {A.n_obs:,} x {A.n_vars:,}")

        for name, piece in pieces:
            if name is not None:
                print(f"\n{name}: {piece.n_obs:,} nuclei")
            try:
                info = cluster(piece, resolutions=res, n_top_genes=a.n_top_genes,
                               n_pcs=a.n_pcs, n_neighbors=a.n_neighbors, seed=a.seed)
            except (ValueError, AssertionError) as e:
                print(f"scanno: REFUSE - {e}", file=sys.stderr)
                return REFUSE
            print(f"    {info['n_highly_variable']:,} variable genes of {info['n_genes']:,}; "
                  f"raw counts kept in layers['counts']")
            if name is not None:
                p = (out_dir or Path(".")) / f"{name}_clustered.h5ad"
            elif out_dir:
                # Named from the INPUT's stem, so ten libraries land in ten files rather than
                # taking turns overwriting one.
                p = out_dir / f"{Path(src).stem}_clustered.h5ad"
            else:
                p = Path(a.out)
                p.parent.mkdir(parents=True, exist_ok=True)
            from .emit import format_plain_labels, write_h5ad
            for line in format_plain_labels(write_h5ad(piece, p)):
                print(line)
            written.append(p)
            print(f"    wrote {p}")

    print(f"\n{len(written)} object(s) written. Next: `scanno annotate --cluster-key "
          f"leiden_{ _RES_TAG(res[len(res)//2]) }` (or any of the others), then "
          f"`scanno resolution` to choose one on the LABEL rather than the partition.")
    return 0


def _RES_TAG(r):
    from .cluster import res_tag
    return res_tag(r)


def _background(a):
    """Build ONE gene background from a cohort's own clusters, and save it.

    `--background-from-clusters` derives a background from the object in front of it, which makes
    a cluster's score depend on that object's composition. Across a cohort that is exactly wrong:
    composition is often the thing being compared, and a per-sample background couples the labels
    to the design. `scanno calibrate` builds a shareable store but wants annotated ATLASES, which
    is a different input from the study's own libraries.

    So: pool every library, build the background once, write it, and let `annotate --store` use
    the same numbers for all of them. What this does NOT do is make the background independent of
    the study - it is still this cohort's. It makes it independent of WHICH LIBRARY a nucleus
    sits in, which is the comparison the design rests on, and the run says so.
    """
    try:
        import anndata as ad
        import numpy as np
    except ImportError:
        print("scanno: background needs anndata.  pip install -e '.[run]'", file=sys.stderr)
        return 1
    from .store import build_store

    parts, n_total = [], 0
    for src in a.h5ad:
        A = ad.read_h5ad(src)
        if a.cluster_key not in A.obs:
            print(f"scanno: {src} has no obs column {a.cluster_key!r}. Available: "
                  f"{', '.join(list(A.obs.columns)[:12])}", file=sys.stderr)
            return 1
        # Same default as `annotate`: the store and the annotation must be built on the same
        # naming or `standardise` matches nothing, and the two commands disagreeing about which
        # column holds the gene names is a silent way to get an empty background.
        gk = a.gene_key
        if gk is None and "gene_symbol" in A.var:
            gk = "gene_symbol"
        genes = np.array([str(v).upper() for v in
                          (A.var[gk] if gk and gk in A.var else A.var_names)])
        # Qualified by object, so cluster 0 of one library is not pooled with cluster 0 of
        # another. They are different populations and averaging them is not a background, it is
        # a blur.
        tag = Path(src).stem
        lab = np.char.add(tag + ":", A.obs[a.cluster_key].astype(str).values)
        parts.append((tag, genes, A.X, lab))
        n_total += int(A.n_obs)
        print(f"  {tag:<28} {A.n_obs:>8,} cells  {len(set(lab)):>4} clusters")
    if not parts:
        print("scanno: REFUSE - no objects given.", file=sys.stderr)
        return REFUSE

    store = build_store(parts, {"species": a.species, "tissue": a.tissue, "assay": a.assay})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # The same keys `calibrate.load_store` reads, written the same way. Not shared with
    # `calibrate.save` because that writes a whole calibration directory - reliability, panels,
    # the promotion ladder - none of which exists here: this is a background, not a calibration,
    # and emitting empty tables beside it would suggest otherwise.
    np.savez_compressed(
        a.out, context=json.dumps(store.context), genes=store.genes,
        celltypes=np.array(store.celltypes, dtype=object), mean=store.mean,
        detect=store.detect, n_cells=store.n_cells, n_present=store.n_present,
        n_sources=store.n_sources, n_clean=store.n_clean, between_sd=store.between_sd,
        gene_mu=store.gene_mu, gene_sd=store.gene_sd, digest=store.digest)
    print(f"\nwrote {a.out}   digest {store.digest}")
    print(f"  {n_total:,} cells from {len(parts)} object(s), {len(store.genes):,} genes")
    print("  Pass it to every `scanno annotate` in this cohort with --store, so a cluster's")
    print("  score does not depend on which library it happened to sit in.")
    print("  It is still THIS cohort's background: independent of the library, not of the study.")
    return 0


def _compare(a):
    """Two routes to the same labels, and how far they agree."""
    try:
        import anndata as ad
    except ImportError:
        print("scanno: compare needs anndata.  pip install -e '.[run]'", file=sys.stderr)
        return 1
    import json as _json

    from .compare import compare, format_report

    A = ad.read_h5ad(a.a, backed="r")
    B = ad.read_h5ad(a.b, backed="r")
    for obj, path, key in ((A, a.a, a.path_key), (B, a.b, a.path_key_b or a.path_key)):
        if key not in obj.obs:
            print(f"scanno: {path} has no obs column {key!r}. Annotate it first with "
                  f"`scanno annotate --out-h5ad`.", file=sys.stderr)
            return 1
    if a.path_key_b and a.path_key_b not in B.obs:
        print(f"scanno: {a.b} has no obs column {a.path_key_b!r}.", file=sys.stderr)
        return 1
    if a.agreement_key and a.agreement_key not in B.obs:
        print(f"scanno: {a.b} has no obs column {a.agreement_key!r}. Annotate route B with more "
              f"than one --cluster-key first.", file=sys.stderr)
        return 1
    res = compare(A.obs, B.obs, path_key=a.path_key, path_key_b=a.path_key_b,
                  sample_key=a.sample_key, cluster_key=a.cluster_key, group_key=a.group_key,
                  agreement_key=a.agreement_key)
    print("")
    for line in format_report(res, a_name=Path(a.a).stem, b_name=Path(a.b).stem):
        print(line)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(_json.dumps(res, indent=1, default=str), encoding="utf-8")
        print(f"\nwrote {a.out}")
    if a.out_table:
        mc = res.get("merge_candidates")
        if mc is None:
            print("scanno: --out-table needs --sample-key and --cluster-key", file=sys.stderr)
            return 1
        import csv as _csv
        cols = ["cluster", "n_cluster", "label_absent", "label_carried", "samples_with",
                "samples_lacking", "n_cells", "n_route_a_agrees", "pct_route_a_agrees",
                "pct_sweep_agrees", "top_sample", "top_share_pct", "moving_by_group"]
        Path(a.out_table).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_table, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in mc["candidates"]:
                w.writerow({k: (";".join(r[k]) if isinstance(r.get(k), list)
                                else ";".join(f"{a_}={b_}"
                                              for a_, b_ in sorted(r[k].items()))
                                if isinstance(r.get(k), dict) else r.get(k)) for k in cols})
        print(f"wrote {a.out_table}   {mc['n_candidates']} candidate(s)")
    if a.out_impact:
        mc = res.get("merge_candidates")
        if mc is None:
            print("scanno: --out-impact needs --sample-key and --cluster-key", file=sys.stderr)
            return 1
        import csv as _csv
        ips = mc["impact_per_sample"]
        cols = ["sample", "label", "n_sample_total", "n_before", "n_after", "n_delta",
                "pct_before", "pct_after", "pct_delta", "is_sentinel"]
        Path(a.out_impact).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_impact, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in ips["rows"]:
                w.writerow({k: r[k] for k in cols})
        moved = sum(1 for r in ips["rows"] if r["n_delta"])
        print(f"wrote {a.out_impact}   {len(ips['rows'])} sample x label rows, "
              f"{moved} of them change")

    if a.out_h5ad or a.out_report:
        mc = res.get("merge_candidates")
        if mc is None:
            print("scanno: the joint route needs --sample-key and --cluster-key",
                  file=sys.stderr)
            return 1
        import numpy as np
        import pandas as pd

        import datetime as _dt
        import hashlib as _hashlib

        from .emit import annotate_joint, write_h5ad
        from .joint import document, reconcile, summarise

        # Route B's clustering aligned onto route A BY BARCODE, never by position: the joint
        # object is a different clustering of the same cells and nothing guarantees it was
        # written in route A's order. Comparing positionally would measure a shuffle.
        A2 = ad.read_h5ad(a.a)
        bi = B.obs.index.astype(str)
        clu = pd.Series(np.asarray(B.obs[a.cluster_key].astype(str)), index=bi)
        sam = pd.Series(np.asarray(B.obs[a.sample_key].astype(str)), index=bi)
        clu, sam = clu[~clu.index.duplicated()], sam[~sam.index.duplicated()]
        ai = A2.obs.index.astype(str)
        n_shared = int(ai.isin(clu.index).sum())
        if not n_shared:
            print(f"scanno: {a.a} and {a.b} share no barcodes; they are not the same cells.",
                  file=sys.stderr)
            return 1
        clu = clu.reindex(ai).fillna("__not_in_route_b__").to_numpy()
        sam = sam.reindex(ai).fillna("__not_in_route_b__").to_numpy()

        # Route B's LABEL, aligned by barcode like its clustering. Needed because the joint
        # route is not only what it resolved: a label route B delivers nowhere is one its
        # partition could not separate, and the column has to carry that too.
        key_b = a.path_key_b or a.path_key
        lb = pd.Series(np.asarray(B.obs[key_b].astype(str)), index=bi)
        lb = lb[~lb.index.duplicated()].reindex(ai).fillna("").to_numpy()
        labels = np.asarray(A2.obs[a.path_key].astype(str))
        new_labels, origin, record = reconcile(labels, lb, clu, sam, mc["candidates"])
        record["n_not_in_route_b"] = int(A2.n_obs - n_shared)
        summ = summarise(labels, new_labels, sam)
        out_key = a.out_key or f"{a.path_key}_joint"

        if a.out_h5ad:
            info = annotate_joint(A2, new_labels, origin, record, key=out_key)
            # THE SWEEP AGREEMENT TRAVELS WITH THE CORRECTION, per cell, onto the object the
            # correction is delivered in. It is a property of route B measured on these same
            # barcodes, and leaving it behind in route B's object would mean a reader of the
            # deliverable could see that a cell was corrected and not how much of the sweep
            # backed it - which is the one thing that separates a recovered population from a
            # property of the granularity. NaN for a barcode route B never saw; never 0, which
            # would read as "the sweep disagreed" rather than "the sweep was not there".
            if a.agreement_key and a.agreement_key in B.obs:
                _ag = pd.Series(pd.to_numeric(np.asarray(B.obs[a.agreement_key]),
                                              errors="coerce"), index=bi)
                _ag = _ag[~_ag.index.duplicated()].reindex(ai)
                A2.obs[out_key + "_sweep_agreement"] = np.asarray(_ag, dtype="float32")
                print(f"    +obs[{out_key + '_sweep_agreement'!r}] from "
                      f"obs[{a.agreement_key!r}] of route B")
            Path(a.out_h5ad).parent.mkdir(parents=True, exist_ok=True)
            write_h5ad(A2, a.out_h5ad)
            print(f"wrote {a.out_h5ad}   +obs[{info['key']!r}] and "
                  f"obs[{info['origin_key']!r}]   {info['n_corrected']:,} cells corrected")

        if a.out_report:
            cols = [
                {"column": a.path_key, "what it is": "the annotation being corrected",
                 "cells differing from the one above": ""},
                {"column": out_key,
                 "what it is": "the same labels with every merge candidate applied",
                 "cells differing from the one above": record["n_corrected"]},
            ]
            rev = None
            # THE ADAPTOR, NOT AN AGENT. No key, no resident model, no subprocess: the run
            # WRITES DOWN what a reviewer has to read, and `scanno joint-review` reads a verdict
            # back. A working agent - a person, or one in a session - does the judging in
            # between, following skills/joint-route-review. A tool that needed a provider to be
            # reviewable would be unreviewable on a compute node, which is where it runs.
            from .joint import review as _review
            from .joint import review_prompt as _prompt
            _lost = mc.get("lost_labels")
            _gk = (mc.get("impact") or {}).get("group_key")
            _req = ["# Review request — " + str(Path(a.a).name),
                    "",
                    f"{len(mc['candidates'])} candidate(s). Grade each one: adopt, refuse or "
                    "undecided, with a reason citing the numbers below.",
                    "Record them with:",
                    "",
                    "    scanno joint-review --payload <this run>/report/joint_route.json \\",
                    "        --verdict '<cluster>=<grade>:<reason>' ... \\",
                    "        --out <this run>/compare/verdicts.json \\",
                    "        --out-report <this run>/report/joint_route.html",
                    "",
                    "The procedure, the four criteria and the four prohibitions are in "
                    "skills/joint-route-review.", ""]
            _req += ["", _prompt(mc["candidates"][0], lost=_lost, group_key=_gk).split(
                "CANDIDATE -")[0].rstrip(), ""] if mc["candidates"] else []
            for _c in mc["candidates"]:
                _req += ["", "---", "",
                         _prompt(_c, lost=_lost, group_key=_gk, brief=False)]
            if a.out_report:
                _rp = Path(a.out_report).with_name("review_request.md")
                _rp.parent.mkdir(parents=True, exist_ok=True)
                _rp.write_text("\n".join(_req), encoding="utf-8")
                print(f"wrote {_rp}   {len(mc['candidates'])} candidate(s) awaiting a verdict")
            if a.verdicts and Path(a.verdicts).exists():
                rev = _json.loads(Path(a.verdicts).read_text(encoding="utf-8"))
            elif a.verdicts:
                # NO REVIEWER WAS ATTACHED, and that is a result rather than an absence. Every
                # candidate is recorded as ungraded with its cell count, so a reader meets the
                # same file whether or not anyone judged the run - and cannot mistake a review
                # that did not happen for one that found nothing to object to.
                from .joint import review as _review
                rev = _review(mc["candidates"], {},
                              provenance={"source": "none",
                                          "limit": "no reviewer was attached to this run. "
                                                   "Every candidate is ungraded; ungraded is "
                                                   "not approved."})
                Path(a.verdicts).parent.mkdir(parents=True, exist_ok=True)
                Path(a.verdicts).write_text(_json.dumps(rev, indent=1, default=str),
                                            encoding="utf-8")
                print(f"wrote {a.verdicts}   NO reviewer attached - "
                      f"{rev['n_candidates']} candidate(s) ungraded, "
                      f"{rev['n_cells_ungraded']:,} cells")
            payload = {
                "generated": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "version": __import__("scanno").__version__,
                "a_name": Path(a.a).name, "b_name": Path(a.b).name,
                "forced_key": a.path_key, "out_key": out_key, "columns": cols,
                "record": record, "summary": summ, "compare": res, "review": rev,
            }
            html = document(payload)
            Path(a.out_report).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out_report).write_text(html, encoding="utf-8")
            # THE PAYLOAD, BESIDE THE PAGE. Everything the document was built from, so a verdict
            # can be folded in later WITHOUT re-clustering, re-annotating or re-comparing - the
            # review is a reader's step and must not cost a run.
            _pj = Path(a.out_report).with_suffix(".json")
            _pj.write_text(_json.dumps(payload, indent=1, default=str), encoding="utf-8")
            print(f"wrote {_pj}   the document's own inputs, for `scanno joint-review`")
            print(f"wrote {a.out_report}   {summ['n_changed']:,} cells differ")
    return 0


def _joint_review(a):
    """Record verdicts against a finished run, and re-render its document. No analysis."""
    import json as _json

    from .joint import document, review

    payload = _json.loads(Path(a.payload).read_text(encoding="utf-8"))
    mc = ((payload.get("compare") or {}).get("merge_candidates")) or {}
    if not mc.get("candidates"):
        print(f"scanno: {a.payload} carries no candidates to grade.", file=sys.stderr)
        return 1
    verdicts = {}
    for spec in a.verdict:
        if "=" not in spec or ":" not in spec.split("=", 1)[1]:
            print(f"scanno: REFUSE - {spec!r} is not CLUSTER=GRADE:REASON", file=sys.stderr)
            return REFUSE
        cl, rest = spec.split("=", 1)
        grade, reason = rest.split(":", 1)
        verdicts[cl.strip()] = (grade.strip(), reason)

    rec = review(mc["candidates"], verdicts,
                 provenance={"source": "recorded", "reviewer": str(a.reviewer or "unnamed"),
                             "payload": str(a.payload),
                             "limit": "a verdict is a reader's note recorded against the run. "
                                      "It changes no label."})
    for e in rec["errors"]:
        print(f"scanno: REFUSE - {e}", file=sys.stderr)
    if rec["errors"]:
        return REFUSE

    print(f"{rec['n_graded']} of {rec['n_candidates']} candidate(s) graded"
          f"{' by ' + a.reviewer if a.reviewer else ''}")
    for g in rec["grades"]:
        print(f"  {g:<10} {rec['n_cells_by_grade'].get(g, 0):>7,} cells")
    if rec["ungraded"]:
        print(f"  UNGRADED   {rec['n_cells_ungraded']:>7,} cells in cluster(s) "
              f"{', '.join(rec['ungraded'])}   — ungraded is NOT approved")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(_json.dumps(rec, indent=1, default=str), encoding="utf-8")
        print(f"wrote {a.out}")
    if a.out_report:
        payload["review"] = rec
        Path(a.out_report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_report).write_text(document(payload), encoding="utf-8")
        print(f"wrote {a.out_report}")
    return 0


def _scope(a):
    """The common scope: the splits every sample agreed to make, as a sealed tree."""
    try:
        import anndata as ad
    except ImportError:
        print("scanno: scope needs anndata.  pip install -e '.[run]'", file=sys.stderr)
        return 1
    import json as _json

    from .scope import (bare_names_unique, format_report, format_tree, internal_nodes,
                        scope_labels, seal_tree, sealed_labels, truncate_tree, vote)

    tree = _json.loads(Path(a.tree).read_text(encoding="utf-8"))
    dup = bare_names_unique(tree)
    if dup:
        # A seal is applied by BARE name because that is how `children` is keyed. If a name sits
        # at two positions, sealing one seals both — silently, and in the wrong lineage.
        print(f"scanno: scope refuses — these names appear at more than one position in the "
              f"tree, so a seal cannot be applied unambiguously: {dup}", file=sys.stderr)
        return 1

    paths_by_sample = {}
    for p in a.h5ad:
        obs = ad.read_h5ad(p, backed="r").obs
        if a.path_key not in obs:
            print(f"scanno: {p} has no obs column {a.path_key!r}. Run pass 1 first with "
                  f"`scanno annotate --out-h5ad`.", file=sys.stderr)
            return 1
        name = (str(obs[a.sample_key].iloc[0]) if a.sample_key in obs and len(obs)
                else Path(p).stem)
        if name in paths_by_sample:
            print(f"scanno: two objects both call themselves {name!r}; a vote would count that "
                  f"animal twice. Give them distinct {a.sample_key!r} values.", file=sys.stderr)
            return 1
        paths_by_sample[name] = [str(v) for v in obs[a.path_key]]

    verdicts = vote(paths_by_sample, tree, min_support=a.min_support,
                    min_reach=a.min_reach, descend_rule=a.descend_rule)
    lost = sealed_labels(verdicts, paths_by_sample)
    sealed_tree, removed = seal_tree(tree, verdicts)

    print("")
    print(f"scope over {len(paths_by_sample)} samples: {', '.join(sorted(paths_by_sample))}")
    print(f"rule: seal unless support >= {a.min_support} (descend-rule {a.descend_rule!r}, "
          f"min-reach {a.min_reach})\n")
    for line in format_report(verdicts, removed=removed, sealed=lost,
                              n_samples=len(paths_by_sample)):
        print(line)

    # THE SCOPE ITSELF, drawn. The table above says what the vote decided; this says what you
    # get, which is what a reader checks against their knowledge of the tissue.
    print("\nTHE SCOPE — the taxonomy pass 2 will walk\n")
    for line in format_tree(sealed_tree, verdicts, paths_by_sample):
        print(line)

    if a.out:
        payload = {"rule": {"min_support": a.min_support, "min_reach": a.min_reach,
                            "descend_rule": a.descend_rule, "path_key": a.path_key},
                   "samples": sorted(paths_by_sample),
                   "tree": str(a.tree),
                   "nodes": verdicts,
                   # THE SCOPE ITSELF — the labels the next annotation may deliver. The vote and
                   # the sealed tree are how it is derived and applied; this is the result, and a
                   # consumer should not have to re-derive it from a tree to state it.
                   "scope": scope_labels(tree, verdicts),
                   "sealed": {k: list(v) for k, v in removed.items()},
                   "removed_labels": lost,
                   "declared_internal_nodes": internal_nodes(tree),
                   # The DRAWN scope travels with the JSON. A consumer that re-derived it would
                   # need every pass-1 object just to render a picture, and would drift from
                   # what the vote actually printed the day it ran.
                   "tree_lines": format_tree(sealed_tree, verdicts, paths_by_sample),
                   "n_samples": len(paths_by_sample)}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(_json.dumps(payload, indent=1, default=str), encoding="utf-8")
        print(f"\nwrote {a.out}")
    if a.out_tree:
        Path(a.out_tree).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_tree).write_text(_json.dumps(sealed_tree, indent=1), encoding="utf-8")
        print(f"wrote {a.out_tree}   <- pass 2 reads THIS as --tree")
    if a.out_l1_tree:
        Path(a.out_l1_tree).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_l1_tree).write_text(_json.dumps(truncate_tree(tree, 1), indent=1),
                                       encoding="utf-8")
        print(f"wrote {a.out_l1_tree}   <- the INDEPENDENT L1 run reads THIS as --tree")
    return 0


def _report(a):
    """The delivery: one cohort document, and one comprehensive page per sample.

    The cohort is the document. The questions that matter most in a study are the ones a single
    library cannot answer - how composition varies between samples, whether the calls are
    supported, whether an exclusion fell evenly across the design - and those need every object
    together. A per-sample page exists for when a cohort number looks wrong, and then it carries
    everything about that one sample rather than one aspect of all of them.

    Matrices are opened only for the figures that need expression. Composition, reliability,
    per-sample spread and the exclusion rates are all obs quantities.
    """
    try:
        import anndata as ad
    except ImportError:
        print("scanno: report needs anndata.  pip install -e '.[run]'", file=sys.stderr)
        return 1
    from . import __version__
    from .context import Context
    from .document import write_all
    from .palette import Palette

    path_key = a.path_key or a.label_key.replace("_cell_type", "_path")
    objs, missing = [], []
    for src in a.h5ad:
        A = ad.read_h5ad(src)
        if a.label_key not in A.obs and path_key not in A.obs:
            missing.append(str(src))
            continue
        objs.append((Path(src).stem.replace("_annotated", ""), A))
    if missing:
        print(f"scanno: REFUSE - {len(missing)} object(s) carry neither {a.label_key!r} nor "
              f"{path_key!r}: {', '.join(missing[:4])}\n"
              f"        Annotate them first, or name the column with --label-key.",
              file=sys.stderr)
        return REFUSE
    if not objs:
        print("scanno: REFUSE - no objects given.", file=sys.stderr)
        return REFUSE

    # --l1-key must name a SECOND walk's answer, not the deep walk's own level-1 prefix. Both
    # refusals below catch a column that cannot possibly be independent; a column that merely
    # AGREES everywhere is not caught, and must not be - perfect agreement is the result this
    # section exists to report, and refusing it would refuse the good outcome.
    if a.l1_key:
        if a.l1_key == path_key or a.l1_key == a.label_key:
            print(f"scanno: REFUSE - --l1-key {a.l1_key!r} is the deep walk's own column.\n"
                  f"        The section would compare that column with itself and report 100%\n"
                  f"        agreement, which measures nothing. Annotate a depth-1 tree with\n"
                  f"        `scanno annotate --l1-tree` and name the column it writes.",
                  file=sys.stderr)
            return REFUSE
        deep = sorted({str(v) for _n, A in objs if a.l1_key in A.obs
                       for v in A.obs[a.l1_key].astype(str) if "/" in str(v)})
        if deep:
            print(f"scanno: REFUSE - --l1-key {a.l1_key!r} holds {len(deep)} value(s) below\n"
                  f"        level 1, e.g. {', '.join(deep[:3])}. That is a PATH column, not an\n"
                  f"        L1 column: it was written by a deep walk, so comparing it against\n"
                  f"        the deep walk's root is not an independent measurement.",
                  file=sys.stderr)
            return REFUSE
        if not any(a.l1_key in A.obs for _n, A in objs):
            print(f"scanno: REFUSE - no object carries {a.l1_key!r}. A silently absent L1\n"
                  f"        column renders as a cohort that never had one.", file=sys.stderr)
            return REFUSE

    # The flag is DISCOVERED, not assumed: an object carrying an upstream provenance declaration
    # names its own withheld nuclei, and a report that guessed the column would describe a
    # different set of cells than the annotation withheld.
    flag = a.flag_key
    declaration = {}
    if flag is None:
        from .upstream import declaration as read_declaration
        for _n, A in objs:
            d = read_declaration(A)
            if d:
                declaration = d
                flag = d.get("flag_column") or d.get("column")
                break
    if flag is None:
        for cand in ("cluster_FLAG", "scqc_flag", "FLAG"):
            if any(cand in A.obs for _n, A in objs):
                flag = cand
                break

    joint = ad.read_h5ad(a.joint) if a.joint else None

    panels, panel_missing = None, {}
    if a.panels and str(a.panels) != "auto":
        panels = json.loads(Path(a.panels).read_text(encoding="utf-8"))
        if panels and all(str(k).isdigit() for k in panels):
            panels = {int(k): v for k, v in panels.items()}
    elif str(a.panels or "") == "auto" or (a.db and a.tree):
        # Built from the SAME corpus the classifier scored on, per node, at every level the
        # taxonomy has. A hand-picked panel would show whether the labels match the genes
        # someone chose to plot, which is a question about that person.
        from .corpus import load_assertions
        from .report import panels_by_depth
        if not (a.db and a.tree):
            print("scanno: REFUSE - --panels auto needs --db and --tree.", file=sys.stderr)
            return REFUSE
        asr = load_assertions(a.db, a.species, a.tissue, a.min_tier)
        tree = json.loads(Path(a.tree).read_text(encoding="utf-8"))
        # SYMBOLS, not accessions. Matching a symbol corpus against Ensembl var_names finds
        # nothing and reports every corpus gene as absent from the object - which is what
        # happened, with exit status zero and an empty marker section.
        _vk = a.gene_key
        if _vk is None:
            for _c in ("gene_symbol", "gene_symbols", "symbol", "feature_name", "gene_name"):
                if _c in objs[0][1].var:
                    _vk = _c
                    break
        have = {str(v).upper() for v in
                (objs[0][1].var[_vk] if _vk else objs[0][1].var_names)}
        print(f"  gene space: {'var[' + repr(_vk) + ']' if _vk else 'var_names'}, "
              f"{len(have):,} unique names")
        seen = {}
        for _n, A in objs:
            col = a.path_key or a.label_key.replace("_cell_type", "_path")
            if col in A.obs:
                for v in A.obs[col].astype(str):
                    if v in ("UNRESOLVED", "EXCLUDED"):
                        continue
                    parts = v.split("/")
                    for i in range(len(parts)):
                        seen.setdefault(i + 1, set()).add("/".join(parts[:i + 1]))
        panels, panel_missing = panels_by_depth(asr, tree, {k: sorted(v)
                                                            for k, v in seen.items()},
                                                have=have)
        n_tot = sum(len(v) for d in panels.values() for v in d.values())
        print(f"  marker panels built from the corpus: {n_tot} genes over "
              + ", ".join(f"{len(panels[d])} node(s) at level {d}" for d in sorted(panels)))
        for d, nodes in sorted(panel_missing.items()):
            for node, gone in list(nodes.items())[:4]:
                print(f"    level {d} {node}: in the corpus, NOT in this object - "
                      f"{', '.join(gone)}")
    # The DECLARED tree, loaded whenever it was given rather than only on the --panels auto path.
    # It is what separates a label the taxonomy has nothing below (a complete call) from one whose
    # children the cohort removed (a recoverable one); without it every terminal label reads as
    # complete, which is the more reassuring of the two and wrong for exactly the sealed nodes.
    declared_tree = (json.loads(Path(a.tree).read_text(encoding="utf-8")) if a.tree else None)

    sweep = tol = pick = reason = None
    if a.sweep and str(a.sweep).endswith(".csv"):
        # The table form. One row per (depth, resolution); the chosen value and tolerance come
        # from the .json beside it when there is one.
        import csv as _csv
        rows = list(_csv.DictReader(Path(a.sweep).open(encoding="utf-8")))
        sweep = {}
        for r in rows:
            d = int(float(r.get("depth", 1)))
            rec = {"resolution": r.get("resolution")}
            for k in ("modal", "neighbour", "complete", "truncated", "unresolved",
                      "smallest", "n_units", "n_labels", "min_groups"):
                v = r.get(k)
                rec[k] = None if v in (None, "", "None") else float(v)
            sweep.setdefault(d, []).append(rec)
        sweep = sweep or None
        js = Path(str(a.sweep)[:-4] + ".json")
        if js.exists():
            sw = json.loads(js.read_text(encoding="utf-8"))
            tol = sw.get("tolerance", sw.get("tolerance_points"))
            pick, reason = sw.get("pick"), sw.get("reason")
        print(f"  resolution sweep read from {a.sweep}: "
              + ", ".join(f"{len(v)} candidate(s) at depth {k}" for k, v in sorted(sweep.items()))
              if sweep else f"  {a.sweep} carried no rows")
    elif a.sweep:
        sw = json.loads(Path(a.sweep).read_text(encoding="utf-8"))
        # Tolerant of the two shapes this file has been written in. A sweep silently parsed as
        # empty removes the whole section, and the page then looks like a run that had nothing
        # to say about resolution rather than one that was handed the wrong file.
        # Only a MAPPING of depth -> rows. `depths` in one of these files is the list of
        # depths that were swept, not the rows; accepting it produced a sweep of integers and a
        # section that rendered from nothing.
        raw = sw.get("per_depth") or sw.get("by_depth") or {}
        if not isinstance(raw, dict):
            raw = {}
        sweep = {int(k): v for k, v in raw.items() if isinstance(v, list)} or None
        if sweep is None and str(a.sweep).endswith(".csv"):
            sweep = None
        tol = sw.get("tolerance", sw.get("tolerance_points"))
        pick, reason = sw.get("pick"), sw.get("reason")
        if not sweep:
            print(f"scanno: {a.sweep} carries no per-depth rows "
                  f"(keys: {', '.join(sorted(sw)[:8])}). The resolution section will say so.",
                  file=sys.stderr)

    # The common scope, carried in VERBATIM. It is a decision, not a measurement, so the report
    # renders the file rather than re-deriving anything from the annotated objects: re-deriving
    # would describe what pass 2 produced instead of what pass 2 was told to do, and those differ
    # exactly where something went wrong.
    scope = None
    if a.scope:
        scope = json.loads(Path(a.scope).read_text(encoding="utf-8"))
        if not isinstance(scope, dict) or "nodes" not in scope:
            print(f"scanno: REFUSE - {a.scope} is not a `scanno scope --out` result "
                  f"(no 'nodes' key). Rendering it would put an unrecognised file on the page "
                  f"under a heading that claims it is the scope.", file=sys.stderr)
            return REFUSE
        _seals = [n for n, v in scope["nodes"].items() if v.get("verdict") == "SEAL"]
        _lost = sum(sum(d.values()) for d in (scope.get("removed_labels") or {}).values())
        print(f"  scope read from {a.scope}: {len(_seals)} sealed node(s) over "
              f"{scope.get('n_samples', '?')} sample(s), "
              f"{_lost:,} nuclei re-labelled to a parent"
              + (f"  ({', '.join(sorted(_seals)[:4])})" if _seals else ""))
        if not scope.get("tree_lines"):
            print(f"scanno: {a.scope} carries no 'tree_lines'; the drawn scope will be a named "
                  f"absence. Re-run `scanno scope --out` to include it.", file=sys.stderr)

    print(f"{len(objs)} object(s), {sum(A.n_obs for _n, A in objs):,} nuclei")
    if flag:
        print(f"  withheld nuclei read from obs[{flag!r}]"
              + ("  (declared upstream)" if declaration else "  (detected)"))

    ctx = Context(objs, joint_route_key=a.joint_route_key,
                  label_key=a.label_key, path_key=a.path_key,
                  sample_key=a.sample_key, group_key=a.condition_key, joint=joint,
                  panels=panels, chosen_resolution=a.resolution, sweep=sweep, tolerance=tol,
                  sweep_pick=pick, sweep_reason=reason, flag_column=flag,
                  declaration=declaration, version=__version__,
                  tree_path=str(a.tree or ""), species=a.species, tissue=a.tissue,
                  factors=a.factor, pinned_colours=Palette.load(a.palette),
                  gene_key=a.gene_key, joint_key=a.joint_key,
                  forced_key=a.forced_key, forced_l1_key=a.forced_l1_key,
                  group_order=a.group_order, scope=scope, l1_key=a.l1_key,
                  tree=declared_tree)
    print(f"  taxonomy depth {ctx.depth}; "
          f"{', '.join(f'{len(ctx.label_order(d))} labels at level {d}' for d in ctx.levels)}")
    if ctx.auto_factors:
        print(f"  design factors AUTO-DETECTED (declare them with --factor): "
              f"{', '.join(ctx.auto_factors)}")

    out = Path(a.out)
    cohort, payload = write_all(ctx, out, title=a.title, version=__version__,
                                per_sample=not a.no_per_sample)
    print("")
    print(f"wrote {cohort}")
    if not a.no_per_sample:
        print(f"      {out}/reports/samples/   {len(ctx.samples)} per-sample page(s)")
    print(f"      {out}/report.json   every number the pages show, machine-readable")
    # A README beside the output, always. The directory is not self-describing and the reader
    # who arrives next has one question - which of these is the answer - that a listing cannot
    # answer for them.
    from .readme import write as _write_readme
    try:
        # The worked example is READ FROM THE DATA, never written into the module. A hardcoded
        # `Immune/Myeloid/Macrophage` fits one taxonomy, and a reader whose tissue has no such
        # lineage is shown an example that cannot appear in their own file. Take the deepest real
        # path present, which is also the most informative one to show.
        _ex = None
        try:
            _paths = [p for p in ctx.P[ctx.path_key].astype(str).unique()
                      if p and "/" in p]
            _ex = max(_paths, key=lambda p: p.count("/")) if _paths else None
        except Exception:                                                     # noqa: BLE001
            _ex = None
        _write_readme(out, chosen_resolution=a.resolution, path_key=ctx.path_key,
                      obs_columns=list(ctx.objects[0][1].obs.columns), path_example=_ex,
                      n_cells=int(ctx.n), n_samples=len(ctx.samples),
                      taxonomy_depth=int(ctx.depth), version=__version__,
                      inputs=", ".join(str(x) for x in a.h5ad[:3]) + (" ..." if len(a.h5ad) > 3 else ""),
                      withheld=int(ctx.P["flag"].sum()) if ctx.has_flag else None,
                      limits=["No label here is established as correct: there is no truth set, "
                              "and this tool reports what the corpus supports, not what is true.",
                              "The embedding is NOT integrated unless a separate step did so.",
                              "Any nucleus withheld upstream is labelled EXCLUDED and was never "
                              "annotated; its identity is not recoverable from this output."])
        print(f"      {out}/README.md   which file to use, and which not to")
    except Exception as e:                                                # noqa: BLE001
        print(f"  README not written: {type(e).__name__}: {e}")

    n_absent = len(payload["figures_not_drawn"])
    if n_absent:
        print(f"  {n_absent} figure(s) could not be drawn; each is NAMED on the page with the "
              f"input it needed:")
        for r in payload["figures_not_drawn"][:6]:
            print(f"    {r['name']}: {r['reason']}")
    return 0



def _embed(a):
    """ONE embedding over every sample together. Not integration - see scanno/embed.py."""
    try:
        import anndata as ad
    except ImportError:
        print("scanno: embed needs anndata and scanpy.  pip install -e '.[run]'",
              file=sys.stderr)
        return 1
    from .embed import build
    from .emit import classic_string_encoding, plain_string_labels

    # OLD=NEW pairs, parsed before anything is read so a typo costs nothing. A bare column name
    # with no `=` keeps its own name, so `--label-obs scanno_path_scope` still does the obvious
    # thing rather than refusing on punctuation.
    label_map = {}
    for spec in (a.label_obs or []):
        old, _, new = str(spec).partition("=")
        if not old:
            print(f"scanno embed: REFUSE - cannot parse --label-obs {spec!r}", file=sys.stderr)
            return 2
        label_map[old] = new or old

    objs = [(Path(src).stem.replace("_annotated", ""), ad.read_h5ad(src)) for src in a.h5ad]
    print(f"{len(objs)} object(s)")
    J = build(objs, sample_key=a.sample_key, label_map=label_map or None,
              n_hvg=a.n_hvg, n_pcs=a.n_pcs,
              n_neighbors=a.n_neighbors, min_dist=a.min_dist, seed=a.seed,
              gene_key=a.gene_key)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    plain_string_labels(J)
    with classic_string_encoding():
        J.write_h5ad(a.out)
    print("")
    print(f"wrote {a.out}   {J.n_obs:,} cells x {J.n_vars:,} genes")
    print("      NOT integrated: computed over the pooled cells with no batch correction.")
    print("      Whether integration is needed is a separate decision with its own evidence.")
    return 0


def _lab(a):
    """Audit .h5ad files for a browser viewer, and optionally rewrite them so they open.

    The failure this exists for is INVISIBLE IN PYTHON: anndata reads a nullable-string index
    and a classic one into the same pandas object, so a file a viewer cannot open looks
    identical in a notebook to one it can. The user sees `Cannot read properties of undefined
    (reading 'map')`, which names nothing and points nowhere.
    """
    from .emit import audit_file, rewrite_for_viewer

    mark = {"ok": "  ok  ", " warn": " warn ", "warn": " warn ", "missing": "MISSING"}
    worst = 0
    for src in a.h5ad:
        print(f"{src}")
        try:
            rows = audit_file(src)
        except Exception as e:                                            # noqa: BLE001
            print(f"  cannot read: {type(e).__name__}: {e}")
            worst = max(worst, 2)
            continue
        for level, _code, msg in rows:
            print(f"  [{mark.get(level, level)}] {msg}")
            worst = max(worst, {"ok": 0, "warn": 1, "missing": 2}.get(level, 0))
        if a.fix:
            out = Path(a.fix)
            out.mkdir(parents=True, exist_ok=True)
            dst = out / Path(src).name
            A, rep = rewrite_for_viewer(src, dst, path_key=a.path_key,
                                        level_prefix=a.level_prefix, slim=a.slim,
                                        keep=a.keep_obs or ())
            print(f"  rewrote -> {dst}")
            print("           classic string encoding on both indices and every string column")
            if rep["levels"]:
                print(f"           +{len(rep['levels'])} level column(s): "
                      f"{', '.join(rep['levels'])}   (taxonomy depth {rep['depth']})")
            if rep["uns_dropped"]:
                print(f"           dropped {len(rep['uns_dropped'])} scratch uns key(s): "
                      f"{', '.join(sorted(rep['uns_dropped'])[:6])}"
                      + (" ..." if len(rep["uns_dropped"]) > 6 else ""))
            if rep["obs_dropped"]:
                print(f"           obs {rep['obs_before']} -> {rep['obs_after']} columns "
                      f"(dropped {len(rep['obs_dropped'])}, added "
                      f"{len(rep['levels'])}), ALL drops named here:")
                for i in range(0, len(rep["obs_dropped"]), 6):
                    print("             " + ", ".join(rep["obs_dropped"][i:i + 6]))
                print(f"           the source object is unmodified, so the full sweep is still "
                      f"on disk at {src}")
            after = [r for r in audit_file(dst) if r[0] != "ok"]
            print(f"           re-audit: {len(after)} remaining issue(s)"
                  + ("" if not after else ": " + "; ".join(r[2][:60] for r in after)))
        print("")
    if worst == 2 and not a.fix:
        print("Some objects will NOT open in a browser viewer. Rewrite them:")
        print("  scanno lab --h5ad <files> --fix <output-directory>")
    return 0


def _readme(a):
    """Write the README that says WHICH FILE TO USE, by inspecting what is on disk."""
    from .readme import write
    from . import __version__
    obs, n_cells, n_samples, depth, withheld = [], None, None, None, None
    src = sorted(Path(a.dir).glob("annotated/*_annotated.h5ad")) or \
          sorted(Path(a.dir).glob("*.h5ad"))
    if src:
        try:
            import anndata as ad
            o = ad.read_h5ad(src[0], backed="r").obs
            obs = list(o.columns)
            n_samples = len(src)
            n_cells = sum(ad.read_h5ad(f, backed="r").n_obs for f in src)
            pk = a.path_key or next((c for c in obs if c.startswith(("scanno_path", "scAnno_path"))), None)
            if pk and pk in o:
                paths = o[pk].astype(str)
                depth = max((len(p.split("/")) for p in paths
                             if p not in ("EXCLUDED", "UNRESOLVED")), default=None)
        except Exception as e:                                            # noqa: BLE001
            print(f"  could not read {src[0].name}: {type(e).__name__}: {e}")
    p = write(a.dir, chosen_resolution=a.resolution, path_key=a.path_key,
              obs_columns=obs, n_cells=n_cells, n_samples=n_samples,
              taxonomy_depth=depth, version=__version__, command=a.command,
              inputs=a.inputs, withheld=withheld,
              limits=[l for l in (a.limit or [])])
    print(f"wrote {p}")
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


#: Options removed in 0.3.0, with what to do instead. They are checked BEFORE parsing so an old
#: command line gets the reason rather than argparse's "unrecognized arguments" - a generic error
#: invites the reader to look for a typo, and the answer here is that the behaviour is gone.
RETIRED_OPTIONS = {
    "--exclude-mode": (
        "There is one exclusion path now and it is the flag itself. `--exclude-mode cluster` "
        "widened the flag to whole clusters, which excluded nuclei upstream QC had PASSED "
        "(measured: 783 of 2,680, 29.2%, on the cohort it was written for) and made the "
        "excluded set depend on your resolution (42 nuclei at 0.25, 4,080 at 2.0 from one "
        "unchanged flag). Drop the option: `--exclude-flag COLUMN` alone now does what "
        "`--exclude-mode cell` did."),
    "--exclude-share": (
        "A share threshold is a QC decision, and scAnno does not make QC decisions. It existed "
        "only for the removed `--exclude-mode cluster`. If you want a cluster-level exclusion, "
        "compute it upstream where it can be assessed, and hand scAnno the resulting per-cell "
        "column."),
}


def _refuse_retired(argv) -> int | None:
    """Refuse a retired option by name, with the measurement that retired it."""
    for opt, why in RETIRED_OPTIONS.items():
        if any(t == opt or t.startswith(opt + "=") for t in (argv or [])):
            print(f"scanno: REFUSE - {opt} was removed in 0.3.0.\n\n        {why}\n",
                  file=sys.stderr)
            return REFUSE
    return None


def main(argv=None):
    retired = _refuse_retired(argv if argv is not None else sys.argv[1:])
    if retired is not None:
        return retired
    p = argparse.ArgumentParser(prog="scanno", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="run the adversarial suite")
    s.set_defaults(fn=_selftest)

    s = sub.add_parser("cluster", help="step 1: cluster at every resolution, and select nothing")
    s.add_argument("--h5ad", required=True, type=Path, nargs="+", metavar="H5AD",
                   help="one object, or every library of a cohort. Several inputs are each "
                        "clustered INDEPENDENTLY and written to --out-dir named from their own "
                        "stems, so a whole cohort is one command rather than a shell loop")
    s.add_argument("--out", type=Path, help="output object (one input, no --split-by)")
    s.add_argument("--split-by", metavar="OBS_COLUMN",
                   help="cluster each level of this column INDEPENDENTLY - no shared variable "
                        "genes, no joint embedding, no batch key. Whether a cluster present in "
                        "one sample and absent from another is real is a question about "
                        "identity, and a pooled clustering has already decided it")
    s.add_argument("--out-dir", type=Path, help="where the per-group objects go (--split-by)")
    s.add_argument("--resolutions", metavar="SPEC", default=None,
                   help="`0.25,0.5,1.0` or `start:stop:step`. Default 0.25:2.0:0.25. EVERY one "
                        "is kept: a sweep that discarded the evidence for its own stopping "
                        "point would be unfalsifiable")
    s.add_argument("--n-top-genes", type=int, default=None, help="variable genes (default 2000)")
    s.add_argument("--n-pcs", type=int, default=None, help="principal components (default 50)")
    s.add_argument("--n-neighbors", type=int, default=None, help="graph neighbours (default 15)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_cluster)

    s = sub.add_parser("background",
                       help="build ONE gene background from a cohort's own clusters, and save it")
    s.add_argument("--h5ad", required=True, type=Path, nargs="+", metavar="H5AD",
                   help="every library of the cohort. Pooled, so a cluster's score does not "
                        "depend on which library it sat in")
    s.add_argument("--cluster-key", required=True,
                   help="the clustering to profile, e.g. leiden_1p0")
    s.add_argument("--out", required=True, type=Path, help="store.npz")
    s.add_argument("--species", required=True)
    s.add_argument("--tissue", required=True)
    s.add_argument("--assay", default="sc", choices=["sc", "sn"])
    s.add_argument("--gene-key", metavar="VAR_COLUMN", default=None,
                   help="var column holding the gene names, defaulting to var['gene_symbol'] "
                        "when present. Must match what `annotate` uses or the background "
                        "matches nothing")
    s.set_defaults(fn=_background)

    s = sub.add_parser("annotate", help="label the clusters of one object")
    s.add_argument("--h5ad", required=True, type=Path)
    s.add_argument("--cluster-key", required=True, action="append", metavar="OBS_COLUMN",
                   help="the clustering to label. REPEATABLE: give it once per resolution and "
                        "the object is walked at each, against the SAME tree, the SAME scope "
                        "and the SAME gene background - one command, one store digest, one "
                        "answer per resolution. The FIRST is the run in every other respect: "
                        "it alone writes the label columns, the table, the report and the "
                        "independent L1, so adding resolutions changes nothing that was "
                        "already written. Each further key adds ONE column, "
                        "`<prefix>_resolved_path<suffix>_r<tag>`, and two more are voted from "
                        "the family: `<prefix>_sweep_agreement<suffix>`, the "
                        "share of the sweep whose label matches the delivered one, per cell. "
                        "A population too rare to form a cluster at one granularity forms one "
                        "at another, so a call that survives the sweep is a different claim "
                        "from one that does not, and a single resolution cannot state either. "
                        "REPORTED ONLY: no label is voted, chosen or changed by the sweep")
    s.add_argument("--tree", required=True, type=Path)
    s.add_argument("--resolve", action="store_true",
                   help="ALSO write a label column with no holes in it. Every walked cell gets a "
                        "LEAF in `<prefix>_resolved`, `<prefix>_resolved_path` and "
                        "`<prefix>_resolved_origin`; the ordinary label columns are untouched, so "
                        "the decision not to guess is still in the object beside it. A cell the "
                        "walk left UNRESOLVED is pushed from the ROOT down to a leaf, and one "
                        "stranded on an internal node is pushed from there - by the same descent "
                        "the FORCE pass uses, over the argmax the walk ALREADY recorded, so "
                        "nothing is scored that was not scored anyway and nothing is invented. "
                        "Where a chain cannot reach a leaf the cell stays UNRESOLVED and the "
                        "reason is recorded. EXCLUDED cells are never resolved: they were "
                        "withheld before the walk, so there is no trace to descend. "
                        "`<prefix>_resolved_origin` is the column that matters - it says per "
                        "cell whether the leaf was REACHED (`walk`) or ASSIGNED (`forced`, "
                        "`root_forced`), and a root-forced margin is below the gap bar by "
                        "construction, which is why the walk stopped there in the first place.")

    s.add_argument("--scope", type=Path, metavar="JSON",
                   help="the DECIDED scope - `scanno scope --out`. Drives the whole annotation "
                        "from the vote instead of from a hand-sealed tree: every SEAL is applied "
                        "to --tree here (idempotently, so an already-sealed tree is still valid "
                        "input) and every FORCE node is honoured, meaning no cluster may "
                        "TERMINATE on an internal node the cohort agreed to split. A stranded "
                        "cluster is reassigned to the most similar child the UNCHANGED walk "
                        "already recorded, and if that child is itself internal the push REPEATS "
                        "- scoring each further node with the same weights over the same data - "
                        "until a leaf is reached, however deep the tree is. classify() is not "
                        "called again and not modified. Because a forced call lands below the "
                        "gap bar while other samples' cells cleared it, the two are never pooled "
                        "silently: every cell carries `<prefix>_assignment` (gap / forced / "
                        "EXCLUDED) and `<prefix>_force_depth` (how many steps outside the walk "
                        "produced its label), `<prefix>_gap` is the first step's margin, and "
                        "uns['<prefix>_assignment_provenance'] holds the node, the leaf and "
                        "every step's margin per cluster. Without it nothing changes")
    s.add_argument("--l1-tree", type=Path, metavar="JSON",
                   help="a DEPTH-1 tree - `scanno scope --out-l1-tree`. With it the UNCHANGED "
                        "walk runs a SECOND time against that tree and its result becomes the "
                        "`scAnno_L1` column, replacing the one derived by truncating this run's "
                        "path. One command, one object, two label columns: the independent L1 "
                        "and the label from --tree. Without it nothing changes and L1 stays "
                        "derived. An independent L1 cannot be moved by any seal below the root; "
                        "it does NOT rescue a cluster the root itself declined, because both "
                        "walks face the same root child set - see scope.truncate_tree. The "
                        "column is marked in uns['scAnno_L1_provenance'] so a reader can tell "
                        "the two apart. REFUSES a tree deeper than one level")
    s.add_argument("--species", required=True)
    s.add_argument("--tissue", required=True)
    s.add_argument("--assay", default="sc", choices=["sc", "sn"])
    s.add_argument("--db", type=Path, help="marker corpus (corpus weight path)")
    s.add_argument("--store", type=Path, help="store.npz from `scanno calibrate`")
    s.add_argument("--background-from-clusters", action="store_true",
                   help="derive the gene background from this object; reported as REVIEW")
    s.add_argument("--exclude-flag", metavar="OBS_COLUMN",
                   help="obs column of per-NUCLEUS booleans marking what upstream QC flagged. "
                        "EXACTLY those nuclei are withheld: they contribute to no cluster "
                        "profile and each is labelled EXCLUDED. Nothing is deleted, nothing "
                        "unflagged is touched, and the excluded set does not depend on this "
                        "run's clustering. scAnno does not decide which nuclei are technical "
                        "and cannot widen this set. Overrides an upstream declaration")
    s.add_argument("--no-exclude", action="store_true",
                   help="annotate everything, including what upstream QC flagged. Turns off an "
                        "exclusion this object's own declaration would otherwise arm; the "
                        "resulting labels include labels for nuclei upstream QC rejected")
    s.add_argument("--gap-min", type=float, default=None,
                   help="override the descent threshold (0.30 corpus, 0.15 profiles)")
    s.add_argument("--min-tier", type=int, default=4)
    s.add_argument("--out-gene-key", metavar="VAR_COLUMN", default="gene_symbol",
                   help="var column to key the WRITTEN object by (default gene_symbol). "
                        "Accessions are the right index to compute on because symbols are not "
                        "unique, but a reader looks a gene up by symbol. Duplicated symbols keep "
                        "every row and are disambiguated; the accession is preserved in "
                        "var['gene_id']. Pass an empty string to leave var_names alone")
    s.add_argument("--gene-key", metavar="VAR_COLUMN", default=None,
                   help="var column holding the names the CORPUS is keyed on, usually symbols. "
                        "Defaults to var['gene_symbol'] when present, else var_names - because "
                        "an object is often keyed by accession with symbols beside it, and "
                        "matching a symbol corpus against accessions silently returns "
                        "UNRESOLVED for everything")
    s.add_argument("--use-raw", action="store_true")
    s.add_argument("--out", type=Path, help="per-CLUSTER table, as TSV")
    s.add_argument("--out-h5ad", type=Path, metavar="PATH",
                   help="write the annotated object: the input with the label added per CELL, "
                        "as <prefix>_cell_type plus the evidence behind each call. X, var and "
                        "obsm are the input's, untouched. This is the file a viewer opens")
    s.add_argument("--report", type=Path, metavar="PATH",
                   help="write the annotation report: one self-contained HTML file plus a "
                        "report.json carrying every number in it. Composition, the labels on "
                        "the embedding, reliability by tree depth, what was withheld and how "
                        "unevenly, every cluster call, and what none of it can show")
    s.add_argument("--sample-key", metavar="OBS_COLUMN",
                   help="obs column naming the sample/animal each cell came from. Optional: "
                        "with it the report shows composition and exclusion PER SAMPLE, which "
                        "is where unevenness is visible and a cohort bar hides it")
    s.add_argument("--condition-key", metavar="OBS_COLUMN",
                   help="obs column naming the experimental group. Optional: with it the report "
                        "reports the exclusion rate per arm and the widest ratio between arms")
    s.add_argument("--label-suffix", default="", metavar="SUFFIX",
                   help="appended to every obs column written by --out-h5ad. Annotating one "
                        "object at several resolutions wants `--label-suffix _r1p0`, giving "
                        "`scanno_path_r1p0` - the family `scanno resolution` reads. A SUFFIXED "
                        "run names its label column `<prefix>_label<suffix>`, not `cell_type`: "
                        "only the unsuffixed run - the chosen answer - carries the name a viewer "
                        "guesses by, or a sweep gives it eight candidates and it picks the "
                        "finest")
    s.add_argument("--label-prefix", default="scanno", metavar="STEM",
                   help="stem for the obs columns written by --out-h5ad (default: scanno, "
                        "giving scanno_cell_type). The default is chosen so a reader guessing "
                        "which column holds the annotation finds it without being told")
    s.set_defaults(fn=_annotate)

    s = sub.add_parser("rescue",
                       help="a rare cell type missing from one unit, looked for in that unit "
                            "alone, and renamed only where a cluster comes back as it")
    s.add_argument("--h5ad", required=True, nargs="+", type=Path, metavar="H5AD",
                   help="one object per unit, each already annotated at SEVERAL resolutions "
                        "(`scanno annotate` with --cluster-key given more than once)")
    s.add_argument("--label-key", required=True, metavar="OBS_COLUMN",
                   help="the delivered label being corrected - typically the FORCED column, "
                        "which has no holes for a zero to hide in")
    s.add_argument("--sweep-prefix", default="scanno_resolved_path_r", metavar="PREFIX",
                   help="prefix of the per-resolution label columns; the rest of the name is "
                        "the resolution (default scanno_resolved_path_r)")
    s.add_argument("--cluster-prefix", default="leiden_", metavar="PREFIX",
                   help="prefix of the cluster columns, matched rung for rung against "
                        "--sweep-prefix (default leiden_)")
    s.add_argument("--unit-key", default=None, metavar="OBS_COLUMN",
                   help="obs column naming the unit - sample, donor, library. Must be CONSTANT "
                        "within each file. Without it the file stem is used")
    s.add_argument("--from-resolution", type=float, default=None, metavar="R",
                   help="start the search here, normally the resolution the delivered "
                        "annotation used. Default: the coarsest rung every object carries")
    s.add_argument("--to-resolution", type=float, default=None, metavar="R",
                   help="stop here. THE CEILING IS PART OF THE CLAIM: past a granularity a "
                        "study would report at, an appearance stops being evidence that a "
                        "population was merged and becomes evidence the partition was "
                        "shattered. Default: the finest rung every object carries")
    s.add_argument("--tree", type=Path, metavar="JSON",
                   help="the taxonomy. With it only LEAVES are eligible targets: a cell resting "
                        "on an internal node carries a compartment name, and a compartment is "
                        "not a population that can be missing from a unit")
    s.add_argument("--out-key", default=None, metavar="OBS_COLUMN",
                   help="name of the rescued column (default <label-key>_rescued). "
                        "`<name>_origin` says per cell whether it was kept or rescued")
    s.add_argument("--out-dir", type=Path, help="write each object with the rescued columns")
    s.add_argument("--out", type=Path, metavar="JSON", help="every search and its result")
    s.add_argument("--out-table", type=Path, metavar="CSV",
                   help="per unit x label: counts and percentages before and after")
    s.add_argument("--out-report", type=Path, metavar="HTML", help="the document")
    s.set_defaults(fn=_rescue)

    s = sub.add_parser("compare",
                       help="two annotated objects: how far the labels agree, and whether the "
                            "second route is strong enough to be worth comparing against")
    s.add_argument("--a", required=True, type=Path, help="annotated object, route A")
    s.add_argument("--b", required=True, type=Path, help="annotated object, route B")
    s.add_argument("--path-key", default="scanno_path", help="route A's label column")
    s.add_argument("--path-key-b", default=None, metavar="OBS_COLUMN",
                   help="route B's label column, when it differs from --path-key. It usually "
                        "does: two routes are normally annotated under different "
                        "--label-suffix, which is what stops them colliding, and that leaves "
                        "them with different column names. Defaults to --path-key")
    s.add_argument("--sample-key", metavar="OBS_COLUMN",
                   help="with --cluster-key, measures how much of each route-B cluster is one "
                        "sample. A joint clustering of an un-integrated cohort can group cells "
                        "by library rather than by cell type, and then disagreement indicts B. "
                        "It also turns on the per-cluster crosstab of route A's labels BY "
                        "SAMPLE, and the merge candidates read off it")
    s.add_argument("--cluster-key", metavar="OBS_COLUMN", help="route B's cluster column")
    s.add_argument("--agreement-key", metavar="OBS_COLUMN",
                   help="route B's per-cell sweep agreement - `<prefix>_sweep_agreement<suffix>`, written by `scanno annotate` with more than one "
                        "--cluster-key. With it every candidate carries `pct_sweep_agrees`: how "
                        "much of route B's OWN resolution sweep carries the label it is "
                        "offering, over exactly the cells that would move. A population too "
                        "rare to cluster at one granularity clusters at another, so a call the "
                        "whole sweep makes and a call one resolution makes are different "
                        "evidence, and a single-resolution comparison cannot tell them apart. "
                        "REPORTED, never acted on - the candidate set is identical with and "
                        "without it")
    s.add_argument("--group-key", metavar="OBS_COLUMN",
                   help="obs column naming the experimental group. With it, the cells a "
                        "candidate would move are tabulated across that column's levels - rule "
                        "one's third question, is the change differential across the design. It "
                        "is REPORTED and takes no part in deciding what is a candidate")
    s.add_argument("--out", type=Path, help="write the comparison as JSON")
    s.add_argument("--out-impact", type=Path, metavar="CSV",
                   help="write the PER-SAMPLE composition impact: one row per sample x label "
                        "with the count and the share before and after adopting every "
                        "candidate, and the percentage-point change. The denominator is every "
                        "nucleus of that sample and adoption does not change it. Needs "
                        "--sample-key and --cluster-key")
    s.add_argument("--out-h5ad", type=Path, metavar="PATH",
                   help="write route A's object with a THIRD label column added: the "
                        "--path-key label with every merge candidate applied, plus "
                        "<--out-key>_origin naming per cell whether it was kept or corrected. "
                        "The corrected column sits BESIDE the one it corrects and never "
                        "replaces it, so reverting is a column drop. Needs --sample-key and "
                        "--cluster-key")
    s.add_argument("--out-key", metavar="OBS_COLUMN", default=None,
                   help="name for the joint-route column (default <--path-key>_joint)")
    s.add_argument("--verdicts", type=Path, metavar="JSON",
                   help="verdicts to render into --out-report, from `scanno joint-review`. "
                        "Where the file does not exist yet it is WRITTEN, recording every "
                        "candidate as ungraded - a review that has not happened is a result and "
                        "a candidate with no verdict is never assumed to be approved")
    s.add_argument("--out-report", type=Path, metavar="HTML",
                   help="write the joint-route document: the three annotations, every cluster "
                        "the joint route changed with its credibility, what it cost per label "
                        "and per sample, and the populations the joint clustering ABSORBED")
    s.add_argument("--out-table", type=Path, metavar="CSV",
                   help="write the merge candidates as CSV: one row per cluster/label pair "
                        "where a label some samples carry is absent from other samples in the "
                        "same cluster ENTIRELY. Needs --sample-key and --cluster-key")
    s.set_defaults(fn=_compare)

    s = sub.add_parser("joint-review",
                       help="record a graded verdict against each joint-route candidate and "
                            "re-render the document. Reads only what the run already wrote")
    s.add_argument("--payload", required=True, type=Path, metavar="JSON",
                   help="`joint_route.json`, written beside the document by `scanno compare "
                        "--out-report`. Everything the page was built from, so recording a "
                        "verdict costs no clustering, no annotation and no comparison")
    s.add_argument("--verdict", action="append", default=[], metavar="CLUSTER=GRADE:REASON",
                   help="e.g. --verdict '20=refuse:every corrected cell falls in one level of a "
                        "confounded factor'. GRADE is adopt, refuse or undecided; the reason is "
                        "required and recorded verbatim. Repeatable")
    s.add_argument("--reviewer", default="", metavar="NAME",
                   help="who or what graded these, recorded with them")
    s.add_argument("--out", type=Path, metavar="JSON", help="write the verdicts")
    s.add_argument("--out-report", type=Path, metavar="HTML",
                   help="re-render the document with the verdicts in it")
    s.set_defaults(fn=_joint_review)

    s = sub.add_parser("report",
                       help="the report: one cohort document, plus one page per sample")
    s.add_argument("--h5ad", required=True, type=Path, nargs="+", metavar="H5AD",
                   help="every annotated object of the cohort")
    s.add_argument("--out", required=True, type=Path, metavar="DIR",
                   help="output DIRECTORY. Written under it: reports/cohort.html (the "
                        "document), reports/samples/<name>.html (one comprehensive page per "
                        "sample), figures/, tables/, and report.json carrying every number the "
                        "pages show")
    s.add_argument("--joint", type=Path, metavar="H5AD",
                   help="a joint clustering of the whole cohort, annotated the same way. With "
                        "it the report adds the two-route agreement, the label-against-library "
                        "figure and the feature plots, which need one embedding for the cohort")
    s.add_argument("--group-order", nargs="+", default=None, metavar="GROUP",
                   help="the order the experimental groups should be READ in, given as your "
                        "own level names. Alphabetical order interleaves the factors of a "
                        "crossed design so that no two adjacent rows are a comparison. A group "
                        "not named here is appended and reported, never dropped")
    s.add_argument("--forced-key", default=None, metavar="OBS_COLUMN",
                   help="the FORCED scope column - `scanno annotate --resolve` writes "
                        "`<prefix>_resolved_path<suffix>`. With it the report shows the scope "
                        "annotation twice: as the walk delivered it, and with every walked "
                        "nucleus pushed to a leaf, plus what moved between them. Absent, the "
                        "forced block is a NAMED absence rather than the unresolved share "
                        "redistributed by the document, which would be this report inventing a "
                        "measurement the walk declined to make")
    s.add_argument("--joint-route-key", default=None, metavar="OBS_COLUMN",
                   dest="joint_route_key",
                   help="the joint-route column from `scanno compare --out-h5ad --out-key`: the "
                        "forced annotation with a joint clustering's corrections applied. With "
                        "it the composition section carries a THIRD block drawn from that "
                        "column, beside the two it descends from")
    s.add_argument("--forced-l1-key", default=None, metavar="OBS_COLUMN",
                   help="the FORCED L1 column, `scAnno_L1_resolved<suffix>`. Same treatment for "
                        "the independent depth-1 walk")
    s.add_argument("--joint-key", default=None, metavar="OBS_COLUMN",
                   help="the JOINT route's own label column. Required for the two-route "
                        "agreement: a joint object assembled from the per-sample annotations "
                        "carries those columns too, so the default key would compare the "
                        "per-sample labels with themselves and report ~100%")
    s.add_argument("--panels", metavar="JSON|auto",
                   help="marker panels, {node: [genes]} or {depth: {node: [genes]}}. Pass "
                        "'auto' with --db and --tree to build them from the SAME corpus the "
                        "classifier scored on, per node, at every level the taxonomy has. "
                        "Without either, the marker section is a NAMED ABSENCE rather than a "
                        "silently missing one")
    s.add_argument("--db", type=Path, metavar="SQLITE",
                   help="the marker corpus, for --panels auto")
    s.add_argument("--gene-key", default=None, metavar="VAR_COLUMN",
                   help="the var column holding gene SYMBOLS. Detected from gene_symbol, "
                        "symbol, feature_name or gene_name; name it here when the object uses "
                        "something else. A symbol panel matched against Ensembl var_names "
                        "finds nothing and reports every gene as absent")
    s.add_argument("--min-tier", type=int, default=4,
                   help="corpus tier ceiling for --panels auto (default 4)")
    s.add_argument("--palette", type=Path, metavar="JSON",
                   help="pin colours: {\"label or path\": \"#RRGGBB\"}. Anything unpinned is "
                        "assigned automatically - descendants keep their ancestor's hue at a "
                        "different lightness, at whatever depth the taxonomy has")
    s.add_argument("--flag-key", default=None, metavar="OBS_COLUMN",
                   help="the upstream flag naming nuclei withheld before annotation. Read from "
                        "the scQC declaration when the object carries one")
    s.add_argument("--factor", action="append", default=None, metavar="OBS_COLUMN",
                   help="a design factor, repeatable. Rule one's Q3 is computed per factor. "
                        "Without any, low-cardinality obs columns are auto-detected and "
                        "LABELLED as auto-detected wherever they are used")
    s.add_argument("--resolution", default=None, metavar="R",
                   help="the chosen clustering resolution, named on the pages")
    s.add_argument("--sweep", type=Path, metavar="JSON",
                   help="`scanno resolution --out` result; adds the resolution section")
    s.add_argument("--scope", type=Path, metavar="JSON",
                   help="`scanno scope --out` result. Adds the common-scope section: the rule "
                        "the scope was voted under, the sealed tree pass 2 walked, the per-node "
                        "vote, and every label a seal removed with its nuclei count. Without it "
                        "the section is a NAMED ABSENCE, because a reader cannot otherwise tell "
                        "a subtype absent from the tissue from one the scope removed everywhere")
    s.add_argument("--l1-key", default=None, metavar="OBS_COLUMN",
                   help="the INDEPENDENT level-1 label column, as written by "
                        "`scanno annotate --l1-tree`. With it the report adds the delivered "
                        "cell-type tree with L1 integrated: both delivered columns in one "
                        "picture, and the measured concordance between them. Do NOT point this "
                        "at the deep walk's own level-1 prefix - that column is DERIVED from "
                        "the path and agrees with it by construction, so the section would "
                        "report 100% agreement between one column and itself")
    s.add_argument("--no-per-sample", action="store_true",
                   help="write only the cohort document. The per-sample pages are the detail "
                        "behind its rows and are cheap; this is for a cohort of hundreds")
    s.add_argument("--title", default="Annotation", metavar="TEXT")
    s.add_argument("--label-key", default="scanno_cell_type")
    s.add_argument("--path-key", default=None,
                   help="defaults to the label key with _cell_type replaced by _path")
    s.add_argument("--sample-key", default="sample", metavar="OBS_COLUMN",
                   help="the biological unit. Without it every object counts as one sample")
    s.add_argument("--condition-key", default=None, metavar="OBS_COLUMN",
                   help="the experimental group. With it the report shows one point per sample "
                        "grouped by arm, and the within-group spread against the between-group "
                        "range - the comparison that decides whether a compositional claim is "
                        "available at all")
    s.add_argument("--compare", nargs="*", type=Path, metavar="JSON",
                   help="`scanno compare --out` results, summarised into the report")
    s.add_argument("--tree", type=Path)
    s.add_argument("--species", default="")
    s.add_argument("--tissue", default="")
    s.set_defaults(fn=_report)

    s = sub.add_parser("embed",
                       help="compute ONE embedding over every sample together")
    s.add_argument("--h5ad", required=True, type=Path, nargs="+", metavar="H5AD",
                   help="every object of the cohort. Raw counts are taken from "
                        "layers['counts'] where present, else from .X")
    s.add_argument("--out", required=True, type=Path, metavar="H5AD",
                   help="the joint object, for `scanno report --joint`")
    s.add_argument("--sample-key", default="sample", metavar="OBS_COLUMN")
    s.add_argument("--label-obs", dest="label_obs", nargs="*", default=None, metavar="OLD=NEW",
                   help="which ANNOTATION columns survive into the joint object, and what to call "
                        "them there, e.g. `scanno_path_scope=cell_type "
                        "scAnno_L1_scope=cell_compartment`. Every column scAnno did NOT write is "
                        "kept untouched - the design factors, the group, and the QC statistics the "
                        "cells arrived with all travel, because this tool did not write them and "
                        "has no business discarding them. Of the columns it DID write it keeps "
                        "only these, because annotation emits a statistic per label suffix per "
                        "resolution - label, path, depth, gap, survival, support, assignment, "
                        "force_depth, one per level - and a viewer offered twenty of them cannot "
                        "tell which is the answer. Absent, every column travels. A column that "
                        "does not exist is REFUSED by name, because an absent column and an empty "
                        "one produce the same slim object and only one is a mistake.")
    s.add_argument("--gene-key", default=None, metavar="VAR_COLUMN",
                   help="the var column holding gene symbols, used as the shared gene axis. "
                        "Two objects indexed differently concatenate to whatever they happen "
                        "to share, silently")
    s.add_argument("--n-hvg", type=int, default=2000,
                   help="highly-variable genes, selected over ALL genes (default 2000). No "
                        "gene class is excluded; notable ones are counted and reported")
    s.add_argument("--n-pcs", type=int, default=50)
    s.add_argument("--n-neighbors", type=int, default=15)
    s.add_argument("--min-dist", type=float, default=0.2, metavar="D",
                   help="UMAP min_dist (default 0.2). Lower packs each population tighter and "
                        "separates them more clearly, which is what a cohort manifold is read "
                        "for; it is a LAYOUT parameter and changes no label, no count and no "
                        "metric. Recorded in uns['scanno_embed']")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_embed)

    s = sub.add_parser("lab",
                       help="audit .h5ad files for a browser viewer, and rewrite them to open")
    s.add_argument("--h5ad", required=True, type=Path, nargs="+", metavar="H5AD")
    s.add_argument("--fix", type=Path, metavar="DIR",
                   help="write a corrected copy of each object into DIR: classic string "
                        "encoding on both indices and every string column, and scratch uns "
                        "keys dropped. Nothing in the DATA is changed, and what was dropped "
                        "is named")
    s.add_argument("--path-key", default=None, metavar="OBS_COLUMN",
                   help="the annotation path column. With it, --fix writes one column per "
                        "LEVEL of the taxonomy (scAnno_L1, scAnno_L2, ...) beside the full "
                        "path. A viewer groups by a categorical, and handed full paths it "
                        "offers one category per path - the truncations are what a reader "
                        "actually switches between")
    s.add_argument("--level-prefix", default="scAnno_L", metavar="PREFIX",
                   help="the level columns' prefix (default scAnno_L, giving scAnno_L1 ...)")
    s.add_argument("--slim", action="store_true",
                   help="also drop the per-resolution sweep columns. An object swept over "
                        "eight resolutions carries eight of everything and a viewer's column "
                        "list becomes unusable. This IS a removal: every dropped column is "
                        "named, the chosen resolution's columns are kept, nothing matching a "
                        "design, identity or QC name is touched, and the source object is not "
                        "modified so the sweep stays on disk in full")
    s.add_argument("--keep-obs", nargs="*", default=None, metavar="COL",
                   help="extra obs columns --slim must keep")
    s.set_defaults(fn=_lab)

    s = sub.add_parser("scope",
                       help="step 2 of 3: vote the ten scouting walks into ONE common scope")
    s.add_argument("--h5ad", required=True, type=Path, nargs="+", metavar="H5AD",
                   help="every PASS 1 object — the independent per-sample walks")
    s.add_argument("--tree", required=True, type=Path, metavar="JSON",
                   help="the DECLARED taxonomy pass 1 walked")
    s.add_argument("--path-key", default="scanno_path", metavar="OBS_COLUMN",
                   help="the pass-1 path column (default scanno_path; a sweep suffixes it, "
                        "e.g. scanno_path_r1p0)")
    s.add_argument("--sample-key", default="sample", metavar="OBS_COLUMN",
                   help="the biological unit. One vote per sample, not per object")
    s.add_argument("--min-support", type=float, default=1.0, metavar="FRAC",
                   help="seal a node unless at least this fraction of the samples that REACHED "
                        "it descended below it. Default 1.0 — unanimity: a split one animal "
                        "would not make is a split the cohort cannot be compared across. "
                        "Loosening it lets residual per-sample depth disagreement through, "
                        "which is the whole problem this step exists to remove")
    s.add_argument("--min-reach", type=int, default=2, metavar="N",
                   help="a node reached by fewer than N samples is UNVOTABLE and is left OPEN, "
                        "never sealed. Sealing on one animal's evidence is a removal with no "
                        "quorum behind it (default 2)")
    s.add_argument("--descend-rule", choices=("any", "majority"), default="any",
                   help="what counts as one sample having descended. 'any' — at least one of "
                        "its cells went below. 'majority' — more than half the cells arriving "
                        "at the node did. They differ because a single sample can do both: "
                        "different clusters of one lineage truncating differently inside one "
                        "animal is common (default any)")
    s.add_argument("--out", type=Path, metavar="JSON",
                   help="the vote, every node's evidence, and the labels each seal removes")
    s.add_argument("--out-tree", type=Path, metavar="JSON",
                   help="the SEALED taxonomy. Pass 2 reads this as --tree; the walk itself is "
                        "unchanged, only the tree it walks is smaller")
    s.add_argument("--out-l1-tree", type=Path, metavar="JSON",
                   help="the DEPTH-1 taxonomy, for an INDEPENDENT L1 annotation. Running the "
                        "unchanged walk against this gives an L1 column no seal at any depth "
                        "can move — independent by construction, not by convention. Deriving "
                        "L1 as path[:1] instead makes it inherit the deep walk's failures: a "
                        "nucleus the walk sent to UNRESOLVED has no path to truncate and so no "
                        "L1 at all")
    s.set_defaults(fn=_scope)

    s = sub.add_parser("readme",
                       help="write the README that says WHICH FILE TO USE, from what is on disk")
    s.add_argument("--dir", required=True, type=Path, help="a scAnno output directory")
    s.add_argument("--resolution", default=None, help="the chosen clustering resolution")
    s.add_argument("--path-key", default=None, help="the obs column carrying the full label path")
    s.add_argument("--command", default=None, help="the command that produced it, for the record")
    s.add_argument("--inputs", default=None, help="what it was produced from")
    s.add_argument("--limit", action="append", default=None,
                   help="a limit of this output, repeatable. Written under 'What this cannot "
                        "tell you' - the section most often omitted and most worth having")
    s.set_defaults(fn=_readme)

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
