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

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
