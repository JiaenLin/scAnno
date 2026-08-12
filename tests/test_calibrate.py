"""Calibration, on synthetic data — runs anywhere, needs no atlas and no corpus.

The properties checked here are the ones that make a learned weight safe to ship: it is
bounded away from zero, it is shrunk when support is thin, promotion needs label-clean
sources, and the reordering it emits is a table someone can read.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scanno.calibrate import L_MAX, L_MIN, POOL_K, calibrate, load_store, save
from scanno.store import build_store

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


GENES = np.array([f"G{i:03d}" for i in range(60)])
TYPES = {"alpha": 0, "beta": 20, "gamma": 40}       # each type's marker block


def synth(seed, n=200):
    """Three cell types, each with a private block of 20 informative genes."""
    rng = np.random.default_rng(seed)
    X, lab = [], []
    for ct, off in TYPES.items():
        b = rng.gamma(0.4, 0.5, size=(n, len(GENES)))
        b[:, off:off + 20] += rng.gamma(3.0, 0.9, size=(n, 20))
        X.append(b)
        lab += [ct] * n
    return np.vstack(X), np.array(lab)


TREE = {
    "children": {"root": ["alpha", "beta", "gamma"]},
    "members": {"alpha": ["alpha"], "beta": ["beta"], "gamma": ["gamma"]},
    "patterns": {"alpha": ["alpha"], "beta": ["beta"], "gamma": ["gamma"]},
}
CTX = {"species": "Synth", "tissue": "Synth", "assay": "sc"}


def fake_corpus():
    """A corpus right about 18 markers per type and wrong about 3 — the wrong ones cited
    HARDER than the right ones.

    That is how the contamination actually looks: a gene heavily cited in the literature,
    genuinely important, and belonging to a neighbouring lineage. It has to sit in the
    citation top for `DEMOTED` to be the right verdict, because that verdict means "the
    literature rates this highly and the data disagrees".
    """
    asr = {}
    for ct, off in TYPES.items():
        d = {f"G{i:03d}": 40.0 for i in range(off, off + 18)}
        wrong = (off + 25) % len(GENES)                # a neighbour's marker block
        for i in range(wrong, wrong + 3):
            d[f"G{i % len(GENES):03d}"] = 55.0         # cited above the true markers
        asr[ct] = d
    return asr


def planted_wrong():
    out = set()
    for ct, off in TYPES.items():
        w = (off + 25) % len(GENES)
        for i in range(w, w + 3):
            out.add((ct, f"G{i % len(GENES):03d}"))
    return out


def main():
    one = build_store([("srcA", GENES, *synth(0), "sorted")], CTX)
    many = build_store([(f"src{k}", GENES, *synth(k), "sorted") for k in range(6)], CTX)
    asr = fake_corpus()

    print("\nbounds — a learned weight is never zero and never unbounded")
    cal = calibrate(many, asr, TREE, CTX)
    Ls = np.concatenate([v for v in cal.L.values()])
    check("L stays inside [L_MIN, L_MAX]", Ls.min() >= L_MIN - 1e-9 and Ls.max() <= L_MAX + 1e-9,
          f"observed {Ls.min():.2f} .. {Ls.max():.2f}, bounds [{L_MIN}, {L_MAX}]")
    check("no weight is driven to zero", Ls.min() > 0, f"min {Ls.min():.3f}")

    print("\nshrinkage — thin support must not assert a strong node-specific weight")
    c1 = calibrate(one, asr, TREE, CTX)
    s1 = float(np.std(np.concatenate([v for v in c1.L.values()])))
    s6 = float(np.std(Ls))
    check("weights spread more with more independent sources", s6 > s1,
          f"sd {s1:.3f} at 1 source -> {s6:.3f} at 6  (POOL_K={POOL_K})")

    print("\nprovenance — promotion requires label-clean sources")
    dirty = build_store([(f"src{k}", GENES, *synth(k), "marker_derived") for k in range(6)],
                        CTX)
    gclean = [many.grade(i) for i in range(len(many.celltypes))]
    gdirty = [dirty.grade(i) for i in range(len(dirty.celltypes))]
    check("C1 reachable with clean labels", "C1" in gclean, f"grades {gclean}")
    check("C1 withheld when every label is marker-derived", "C1" not in gdirty,
          f"grades {gdirty} — 6 sources but 0 clean")

    print("\nreordering — the panel is emitted, and the contamination is demoted")
    dem = {(r["node"], r["gene"]) for r in cal.rows if r["verdict"] == "DEMOTED"}
    wrong_claims = planted_wrong()
    hit = len(dem & wrong_claims)
    check("every planted cross-lineage claim is demoted", hit == len(wrong_claims),
          f"{hit}/{len(wrong_claims)} demoted; {len(dem) - hit} other demotions")
    true_claims = {(ct, f"G{i:03d}") for ct, off in TYPES.items()
                   for i in range(off, off + 18)}
    check("no true marker is demoted", not (dem & true_claims),
          f"{len(dem & true_claims)} false demotions")
    rho = {r["node"]: r["panel_rho"] for r in cal.rows}
    check("panels actually move", all(v < 0.999 for v in rho.values()),
          ", ".join(f"{k} rho={v}" for k, v in sorted(rho.items())))

    print("\nround trip — a store survives being written and read")
    with tempfile.TemporaryDirectory() as td:
        save(cal, many, td)
        for f in ("store.npz", "reliability.tsv", "panels.tsv", "calibration.json"):
            check(f"{f} written", (Path(td) / f).exists())
        back = load_store(Path(td) / "store.npz")
        check("digest is stable across the round trip", back.digest == many.digest,
              f"{many.digest} -> {back.digest}")
        check("profiles survive unchanged", np.allclose(back.mean, many.mean))

    print(f"\n{'=' * 60}")
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        return 1
    print("all calibration checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
