"""`scanno embed` names each library from what the object and the manifest record, never a file's name.

WHY THIS FILE EXISTS

When the objects did not carry `--sample-key`, embed filled the joint object's library column with
each input's file stem. Through a pipeline the per-library files carry the stage that wrote them
(`<library>.ladder.h5ad`, `<library>.ladder.rescued.h5ad`), so the column read `<library>.ladder`
and a later stage keyed on the cohort's own names found no library - with exit 0 here.

Everything below runs the real CLI (`python -m scanno.cli embed`) on the suite's synthetic cohort,
written under suffixed file names. At 9fb561c: the suffixed files come out with their stems as
names, every refusal below exits 0 or fails at the write after the embedding, and the joint object
records no source for any name.
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

fails = []
def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: fails.append(name)

try:
    import anndata as ad
    import numpy as np
except ImportError:
    print("SKIP: needs anndata/scanpy.  pip install -e '.[run]'"); raise SystemExit(0)

from _cohort import make_object, SAMPLES  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="embed_names_"))


def write(lib, fname, *, key="sample", recorded=None, sub=""):
    """One library's object under `fname`; its library column called `key` (None: no column)."""
    A = make_object(lib, n_per_type=12)
    col = A.obs.pop("sample")
    if key is not None:
        A.obs[key] = col
    if recorded is not None:                 # scQC's declaration, as it stamps per-library objects
        A.uns["scqc"] = {"schema": "scqc/provenance@1", "tool": "scQC", "sample": recorded,
                         "n_obs": int(A.n_obs), "flag_column": ""}
    d = TMP / sub if sub else TMP
    d.mkdir(parents=True, exist_ok=True)
    p = d / fname
    A.write_h5ad(p)
    return p


def embed(paths, *extra, out="joint.h5ad"):
    o = TMP / out
    if o.exists():
        o.unlink()
    r = subprocess.run([sys.executable, "-m", "scanno.cli", "embed", "--h5ad", *map(str, paths),
                        "--out", str(o), "--n-hvg", "50", "--n-pcs", "5", "--n-neighbors", "5",
                        *extra], cwd=str(ROOT), capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr, (ad.read_h5ad(o) if o.exists() else None)


def names(J, key="sample"):
    return sorted(map(str, J.obs[key].unique())) if J is not None and key in J.obs else None


def refused_early(rc, text, J):
    """Exit 2, nothing written, and no step of the embedding reached (ATTACKS 17)."""
    return rc == 2 and J is None and "preflight" not in text and "UMAP" not in text


libs = list(SAMPLES)

print("1. suffixed files without the key: the library's name the upstream stage recorded")
up = [write(s, f"{s}.ladder.rescued.h5ad", key="library_id", recorded=s, sub="up") for s in libs]
rc, text, J = embed(up)
check("exit 0", rc == 0, text.strip().splitlines()[-1:] if rc else "")
check("the joint library column holds the libraries' own names, not the file stems",
      names(J) == libs, f"got {names(J)}")
rec = dict(J.uns["scanno_embed"].get("library_names", {})) if J is not None else {}
check("the record names the upstream declaration as each name's source",
      list(rec.get("source", [])) == ["uns['scqc']['sample']"] * 3, f"got {rec}")
check("the record keeps each file's base name, never a folder",
      list(rec.get("file", [])) == [p.name for p in up], f"got {list(rec.get('file', []))}")

print("2. suffixed files with no record: a manifest, relative paths from its folder")
bare = [write(s, f"{s}.ladder.h5ad", key=None, sub="bare") for s in libs]
man = TMP / "bare" / "names.tsv"
man.write_text("# libraries\npath\tsample\n" + "".join(f"{p.name}\t{s}\n" for p, s in zip(bare, libs)),
               encoding="utf-8")
rc, text, J = embed(bare, "--manifest", str(man))
check("exit 0 with --manifest", rc == 0, text.strip().splitlines()[-1:] if rc else "")
check("the joint library column holds the manifest's names", names(J) == libs, f"got {names(J)}")
rec = dict(J.uns["scanno_embed"].get("library_names", {})) if J is not None else {}
check("the record names the manifest", list(rec.get("source", [])) == ["manifest"] * 3, f"got {rec}")

print("3. the refusals, each by file and before anything is computed")
rc, text, J = embed(bare)
check("R1 no column, no record, no manifest: refused by file, naming what to pass",
      refused_early(rc, text, J) and bare[0].name in text and "--manifest" in text,
      f"exit {rc}, joint {names(J)}")
dis = TMP / "up" / "disagree.tsv"
dis.write_text("path\tsample\n" + "".join(f"{p.name}\t{s}\n" for p, s in zip(up, ["S9"] + libs[1:])),
               encoding="utf-8")
rc, text, J = embed(up, "--manifest", str(dis))
check("R2 the manifest and the upstream record disagree: refused, naming both values",
      refused_early(rc, text, J) and "'S9'" in text and "'S1'" in text, f"exit {rc}")
same = [write("S1", f"copy{i}.h5ad", key=None, recorded="S1", sub="same") for i in (1, 2)]
rc, text, J = embed(same)
check("R3 one recorded name for two files: refused",
      refused_early(rc, text, J) and "copy1.h5ad" in text and "copy2.h5ad" in text, f"exit {rc}")
mixed = [write("S1", "carries.h5ad", key="sample", sub="mixed"),
         write("S2", "fills.h5ad", key=None, recorded="S1", sub="mixed")]
rc, text, J = embed(mixed)
check("R3 a filled name another object carries in its column: refused",
      refused_early(rc, text, J) and "fills.h5ad" in text, f"exit {rc}")
part = TMP / "bare" / "part.tsv"
part.write_text(f"path\tsample\n{bare[0].name}\tS1\n", encoding="utf-8")
rc, text, J = embed(bare, "--manifest", str(part))
check("R4 a manifest missing an object without the column: refused by file",
      refused_early(rc, text, J) and bare[1].name in text, f"exit {rc}")
nopath = TMP / "bare" / "nopath.tsv"
nopath.write_text("file\tsample\n" + "".join(f"{p.name}\t{s}\n" for p, s in zip(bare, libs)),
                  encoding="utf-8")
rc, text, J = embed(bare, "--manifest", str(nopath))
check("R4 a manifest without a `path` column: refused", refused_early(rc, text, J) and "'path'" in text,
      f"exit {rc}")
rc, text, J = embed(up, "--sample-key", "lib/id")
check("R5 a library column the format forbids ('/'): refused before the embedding, not at the write",
      refused_early(rc, text, J) and "'/'" in text, f"exit {rc}")
rc, text, J = embed(up, "--sample-key", "_index")
check("R5 the reserved column name '_index': refused before the embedding",
      refused_early(rc, text, J), f"exit {rc}")
rc, text, J = embed(up, "--label-obs", "cluster=joint/type")
check("R5 a renamed label column the format forbids: refused before the embedding",
      refused_early(rc, text, J) and "joint/type" in text, f"exit {rc}")
nul = TMP / "bare" / "nul.tsv"
nul.write_text("path\tsample\n" + "".join(f"{p.name}\t{s}\n" for p, s in zip(bare, ["S\x001"] + libs[1:])),
               encoding="utf-8")
rc, text, J = embed(bare, "--manifest", str(nul))
check("R5 a library name with an embedded NUL: refused before the embedding",
      refused_early(rc, text, J) and "NUL" in text, f"exit {rc}")

print("4. objects that carry the key: cell for cell as before, and the record says so")
own = [write(s, f"{s}.ladder.h5ad", key="sample", sub="own") for s in libs]
rc, text, J = embed(own)
want = np.concatenate([np.asarray(ad.read_h5ad(p).obs["sample"].astype(str)) for p in own])
check("exit 0", rc == 0)
check("the joint library column is the objects' own, cell for cell",
      J is not None and np.array_equal(np.asarray(J.obs["sample"].astype(str)), want))
rec = dict(J.uns["scanno_embed"].get("library_names", {})) if J is not None else {}
check("the record names the objects' own column", list(rec.get("source", [])) == ["obs['sample']"] * 3,
      f"got {rec}")
agree = TMP / "own" / "agree.csv"
agree.write_text("path,sample\n" + "".join(f"{p.name},{s}\n" for p, s in zip(own, libs)), encoding="utf-8")
rc, text, J = embed(own, "--manifest", str(agree))
check("a CSV manifest that agrees with the objects' own column: accepted", rc == 0 and names(J) == libs,
      f"exit {rc}")
two = write("S1", "two.h5ad", key="sample", sub="two")
B = ad.read_h5ad(two)
B.obs["sample"] = np.where(np.arange(B.n_obs) % 2 == 0, "S1", "S2")
B.write_h5ad(two)
one = TMP / "two" / "one.tsv"
one.write_text("path\tsample\ntwo.h5ad\tS1\n", encoding="utf-8")
rc, text, J = embed([two], "--manifest", str(one))
check("R2 a manifest naming one library for a file whose column holds two: refused",
      refused_early(rc, text, J) and "2 libraries" in text, f"exit {rc}")
rc, text, J = embed([two])
check("the same file without a manifest: its two libraries kept cell for cell, as before",
      rc == 0 and names(J) == ["S1", "S2"], f"exit {rc}")

print("")
if fails:
    print(f"{len(fails)} FAILED: {', '.join(fails)}"); raise SystemExit(1)
print("embed library names OK - from the object or the manifest, refused by name, recorded")
