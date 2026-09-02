"""One context, assembled once, that every figure and every table reads from.

WHY THIS EXISTS RATHER THAN EACH FIGURE READING WHAT IT NEEDS

A report whose figures and tables compute their own numbers can disagree with itself, and nothing
on the page says which half is right. So the numbers are derived HERE, once, and the figures are
pure functions of the result. A figure cannot contradict the table beside it because they are the
same array.

It reads `obs` for everything it can, and touches a matrix only for the marker and signature
figures - composition, reliability, per-animal spread and the exclusion are all obs quantities,
and opening ten matrices to compute them would make a report cost what the annotation cost.

DEPTH IS READ FROM THE TAXONOMY, NEVER ASSUMED

There is no level 1 and level 2 here. `ctx.depth` is however deep the deepest label actually goes,
`ctx.levels` enumerates 1..depth, and every quantity - composition, ordering, colour, marker
panel, neighbourhood agreement - is computed for each of them. The previous version hardcoded two,
which silently discarded a third level that the annotator had already produced and paid for.

ORDERING IS RECURSIVE

At every depth, labels are ordered by abundance WITHIN their parent, and parents keep the order
they had one level up. So each level's figures are the previous level's subdivided, in the same
left-to-right order, and a reader tracks one population down the tree by staying in the same
place. The sentinels sort last at every depth.
"""
from __future__ import annotations

import numpy as np

EXCLUDED = "EXCLUDED"
UNRESOLVED = "UNRESOLVED"
SENTINELS = (EXCLUDED, UNRESOLVED)
MIN_FLAGGED_PER_ANIMAL = 30      # below this a per-animal flagged-vs-kept comparison is noise
DETECT_FLOOR = 0.05              # a gene detected in under 5% of nuclei is not ranked

#: obs columns that are QC measurements rather than design or identity. Carried through so the
#: exclusion figures can use whichever of them a given object happens to have.
QC_COLUMNS = ("total_counts", "n_genes", "n_genes_by_counts", "pct_counts_mt",
              "doublet_score", "nn_agreement")

#: Columns never offered as a design factor however few levels they have.
NOT_A_FACTOR = {"sample", "barcode", "cell_id", "path", "flag", "_obj", "group"}


def sweep_stem(path_key):
    """`scanno_path_r1p0` -> `scanno_path`; anything else unchanged.

    The split happens only when what follows `_r` PARSES AS A RESOLUTION. Splitting on the bare
    substring is what shipped, and `_r` occurs inside ordinary column names: it turned
    `scanno_resolved_path_scope` into `scanno`, so every sweep column written beside that key
    was looked for under the wrong stem, none matched, and the per-resolution figures reported
    an object that had never been swept. A heuristic that reads a name is a hypothesis about
    it; this one now checks.
    """
    if "_r" in str(path_key):
        stem, tag = str(path_key).rsplit("_r", 1)
        try:
            float(tag.replace("p", "."))
            return stem
        except ValueError:
            pass
    return str(path_key)


class Context:
    """Everything the reports need. `objects` is [(name, AnnData), ...] read in backed mode
    where possible; the matrix is opened only if a figure that needs it is requested."""

    def __init__(self, objects, *, label_key="scanno_cell_type", path_key=None,
                 sample_key="sample", group_key=None, joint=None, panels=None,
                 chosen_resolution=None, resolutions=(), sweep=None, tolerance=None,
                 sweep_pick=None, sweep_reason=None, flag_column=None, declaration=None,
                 version="", tree_path="", corpus_path="", species="", tissue="",
                 factors=None, pinned_colours=None, tree=None, gene_key=None,
                 joint_key=None, group_order=None, scope=None, l1_key=None,
                 forced_key=None, forced_l1_key=None, joint_route_key=None,
                 rescue_key=None):
        import pandas as pd

        self.objects = list(objects)
        self.samples = [n for n, _ in self.objects]
        self.label_key = label_key
        self.path_key = path_key or label_key.replace("_cell_type", "_path")
        self.sample_key, self.group_key = sample_key, group_key
        self.joint = joint
        self.chosen_resolution = chosen_resolution
        self.resolutions = list(resolutions)
        self._sweep = sweep or {}
        self._tolerance = tolerance
        self.sweep_pick, self.sweep_reason = sweep_pick, sweep_reason
        self.flag_column = flag_column
        self.declaration = declaration or {}
        self.version, self.tree_path, self.corpus_path = version, tree_path, corpus_path
        self.species, self.tissue = species, tissue
        self.tree = tree or {}
        self.gene_key, self.gene_column = gene_key, None
        self.joint_key = joint_key
        # The common scope, as `scanno scope --out` wrote it. Carried verbatim and never
        # recomputed: the report must show the decision pass 2 was GIVEN, and anything derived
        # from the annotated objects would instead describe what pass 2 produced.
        self.scope = scope or None
        # The INDEPENDENT L1 column. It is not `L1` truncated from the path: that one is DERIVED
        # from the deep walk and agrees with it by construction, so a report showing it beside
        # the deep label would be showing one column twice. This is a second walk's own answer,
        # and whether the two agree is a MEASUREMENT — `l1_concordance()` — not a guarantee.
        self.l1_key = l1_key or None
        # The FORCED columns: the same two annotations with every walked nucleus pushed to a
        # leaf. Optional, and absent unless `scanno annotate --resolve` wrote them.
        self.forced_key = forced_key or None
        self.forced_l1_key = forced_l1_key or None
        # The JOINT ROUTE column: the forced annotation with a second, joint clustering's
        # corrections applied. A THIRD reading of the same nuclei, not a better one - the joint
        # partition is the coarser of the two and absorbs populations as well as recovering them.
        self.joint_route_key = joint_route_key or None
        self.rescue_key = rescue_key or None
        self._group_order = [str(g) for g in (group_order or [])]
        #: Every name searched for a per-call statistic, so an absence can say what it looked for
        #: rather than guessing the names a second time and guessing differently.
        self.stat_keys_tried = set()

        frames = []
        for name, A in self.objects:
            obs = A.obs
            d = pd.DataFrame(index=obs.index.astype(str))
            d["_obj"] = name
            d["path"] = (obs[self.path_key].astype(str) if self.path_key in obs
                         else np.full(A.n_obs, UNRESOLVED))
            d["sample"] = (obs[sample_key].astype(str) if sample_key in obs else name)
            # Read as a MISSING VALUE where the column is absent, never silently back-filled from
            # the path. An object annotated without the independent walk has no L1 answer, and
            # filling it from the deep path would manufacture perfect agreement for exactly the
            # objects that never measured any.
            if self.l1_key:
                d["l1"] = (obs[self.l1_key].astype(str) if self.l1_key in obs
                           else np.full(A.n_obs, ""))
            # Empty where absent, exactly as `l1` is, and for the same reason: a column that was
            # NAMED and not found must not read as "nothing needed forcing".
            if self.forced_key:
                d["forced"] = (obs[self.forced_key].astype(str) if self.forced_key in obs
                               else np.full(A.n_obs, ""))
            if self.forced_l1_key:
                d["forced_l1"] = (obs[self.forced_l1_key].astype(str)
                                  if self.forced_l1_key in obs else np.full(A.n_obs, ""))
            # Empty where absent, like the two above and for the same reason: a column that was
            # NAMED and not found must not read as "the joint route changed nothing".
            if self.joint_route_key:
                d["joint_route"] = (obs[self.joint_route_key].astype(str)
                                    if self.joint_route_key in obs else np.full(A.n_obs, ""))
            if self.rescue_key:
                d["rescue"] = (obs[self.rescue_key].astype(str)
                               if self.rescue_key in obs else np.full(A.n_obs, ""))
            if group_key and group_key in obs:
                d["group"] = obs[group_key].astype(str)
            # Derived from the PATH KEY, not the label key: an annotation swept over several
            # resolutions writes its statistics per resolution too (`scanno_gap_r1p0`), so a key
            # built from the label name finds `scanno_gap`, finds nothing, and the reliability
            # section disappears with no error - which reads as a run that had no statistics
            # rather than one that looked in the wrong place.
            # TWO GUARDS, and the run that needed them was reporting over a delivered object
            # whose columns are named `cell_type`, `cell_type_forced`, `cell_type_joint_route`.
            #
            #   1. A no-op replace yields the KEY ITSELF. `"cell_type".replace("_path", "_gap")`
            #      is `"cell_type"`, which is in obs, so the LABEL column was read as every
            #      statistic - `could not convert string to float`, from a line that looks like
            #      it is reading a number.
            #   2. A name is not a type. Even a changed name can land on a column holding
            #      something else, and this package's own naming makes that likely: a label
            #      column and its statistics share a stem by design. So the column is COERCED
            #      and accepted only if something numeric survives.
            for stat in ("depth", "gap", "support", "survival"):
                for k in (self.path_key.replace("_path", f"_{stat}"),
                          self.label_key.replace("_cell_type", f"_{stat}")):
                    # RECORD WHAT WAS LOOKED FOR, so an absence can name it. The message that
                    # reports one derived the names by the same replace, which is a no-op on a
                    # key holding neither substring - it told the reader obs carried none of
                    # `cell_type`, `cell_type`, `cell_type` or `cell_type`.
                    if k not in (self.path_key, self.label_key):
                        self.stat_keys_tried.add(k)
                    if k in (self.path_key, self.label_key) or k not in obs:
                        continue
                    _v = pd.to_numeric(pd.Series(np.asarray(obs[k])), errors="coerce")
                    if _v.notna().any():
                        d[stat] = np.asarray(_v, dtype=float)
                        break
            if flag_column and flag_column in obs:
                d["flag"] = np.asarray(pd.Series(obs[flag_column]).astype("boolean")
                                       .fillna(False).to_numpy(dtype=bool))
            for q in QC_COLUMNS:
                if q in obs:
                    d[q] = np.asarray(obs[q], dtype=float)
            # Design factors: whatever the caller named, plus anything low-cardinality that is
            # not identity. Auto-detection is offered because rule one's Q3 must be computable
            # on a project that never told us its design - but an auto-detected factor is
            # LABELLED as such wherever it is used, never presented as a declared one.
            for c in (factors or []):
                if c in obs:
                    d[c] = obs[c].astype(str)
            frames.append(d)
        self.P = pd.concat(frames)
        self.n = len(self.P)

        self.declared_factors = [c for c in (factors or []) if c in self.P]
        self.auto_factors = [] if factors else self._detect_factors()
        self.factors = self.declared_factors or self.auto_factors

        # ---- the taxonomy, at whatever depth it happens to have ---------------------------
        self.depth = max((len(str(p).split("/")) for p in self.P["path"]
                          if str(p) not in SENTINELS), default=1)
        self.levels = list(range(1, self.depth + 1))
        for dpt in self.levels:
            self.P[f"L{dpt}"] = self._trunc(self.P["path"], dpt)

        self.has_flag = "flag" in self.P and bool(self.P["flag"].any())
        # An l1 column of empty strings is a column that was NAMED and not found in any object,
        # which must not read as "the independent walk agreed everywhere".
        self.has_l1 = "l1" in self.P and bool((self.P["l1"].astype(str) != "").any())
        self.has_forced = "forced" in self.P and bool((self.P["forced"].astype(str) != "").any())
        self.has_forced_l1 = ("forced_l1" in self.P
                              and bool((self.P["forced_l1"].astype(str) != "").any()))
        self.has_joint_route = ("joint_route" in self.P
                                and bool((self.P["joint_route"].astype(str) != "").any()))
        self.has_rescue = ("rescue" in self.P
                           and bool((self.P["rescue"].astype(str) != "").any()))
        self._order = {}
        for dpt in self.levels:
            self._order[dpt] = self._order_for(dpt)

        # Marker panels, keyed by depth. A caller may pass {depth: {node: [genes]}} or, for
        # compatibility, a flat {node: [genes]} which is taken as depth 1.
        panels = panels or {}
        if panels and not all(isinstance(k, int) for k in panels):
            panels = {1: panels}
        self._panels = {int(k): v for k, v in panels.items()}

        from .palette import Palette
        # LEVEL-1 ORDER FIRST. The palette takes root order from the order roots first appear,
        # so passing the deepest order first hands out hues in the order the deepest labels
        # happen to fall - which put the two largest populations on adjacent reds in the figure
        # everyone reads. Level 1 first means hue rank == abundance rank at level 1, and the
        # deeper paths that follow only add descendants to roots already placed.
        self.palette = Palette(self._order[1] + self._order[self.depth],
                               pinned=pinned_colours)

    # ------------------------------------------------------------------ labels and ordering
    @staticmethod
    def _trunc(paths, depth):
        return np.array([p if p in SENTINELS else "/".join(str(p).split("/")[:depth])
                         for p in paths], dtype=object)

    def parent_of(self, label):
        """The label one level up. Generalised - `split('/')[0]` was the depth-2 special case."""
        s = str(label)
        return s if s in SENTINELS or "/" not in s else s.rsplit("/", 1)[0]

    def leaf_of(self, label):
        """The last component, for an axis tick that must not carry the whole path."""
        s = str(label)
        return s if s in SENTINELS else s.rsplit("/", 1)[-1]

    def _order_for(self, depth):
        """Abundance order within each parent, parents in the order they took one level up.

        The recursion is what makes every level read as the previous one subdivided. Computed
        bottom-up from `self._order[depth-1]`, which is already final by the time this runs.
        """
        col = self.P[f"L{depth}"]
        counts = {}
        for v in col:
            counts[v] = counts.get(v, 0) + 1
        real = [k for k in counts if k not in SENTINELS]
        if depth == 1:
            real.sort(key=lambda k: -counts[k])
        else:
            up = self._order[depth - 1]
            pos = {p: i for i, p in enumerate(up)}
            real.sort(key=lambda k: (pos.get(self.parent_of(k), len(up)), -counts[k], k))
        return real + [s for s in SENTINELS if s in counts]

    def _rows_over(self, col):
        """(label, nuclei, share, samples, depth) over a column of DELIVERED labels.

        Sentinels sort last: EXCLUDED and UNRESOLVED are not cell types and must not head a
        composition table by size.
        """
        vals = self.P[col].astype(str)
        samp = self.P["sample"].astype(str)
        out = []
        for label in sorted(set(vals)):
            m = (vals == label).values
            n = int(m.sum())
            if not n or label == "":
                continue
            out.append({
                "label": label, "nuclei": n,
                "share": 100.0 * n / max(self.n, 1),
                "samples": int(samp[m].nunique()),
                "depth": 0 if label in SENTINELS
                         else len([x for x in label.split("/") if x])})
        return sorted(out, key=lambda r: (r["label"] in SENTINELS, -r["nuclei"]))

    def scope_rows(self):
        """THE SCOPE ANNOTATION — the delivered terminals, each at the depth it stops at.

        Read from the PATH column itself, never from a truncation. `L{depth}` coincides with the
        delivered set only when `depth` happens to be the declared tree's maximum, so a report
        built on the deepest level index is right by accident and silently wrong on a deeper tree.
        """
        return self._rows_over("path")

    def forced_scope_rows(self):
        """THE SCOPE ANNOTATION, FORCED — every walked nucleus pushed to a leaf.

        The same rows as `scope_rows`, with the UNRESOLVED share redistributed onto the leaves the
        walk declined to choose between. Read the two together: the difference IS the set of calls
        that were made for the reader rather than by the walk.
        """
        return self._rows_over("forced") if self.has_forced else None

    def joint_route_rows(self):
        """THE JOINT ROUTE — the forced annotation with a joint clustering's corrections applied.

        A third reading, not a correction of the second in the sense of being righter: the joint
        partition recovers populations the per-sample one merged AND merges populations it
        recovered. Read the two together; the difference is the set of cells whose label depends
        on which clustering scored them.
        """
        return self._rows_over("joint_route") if self.has_joint_route else None

    def rescue_rows(self):
        """THE RESCUE — the delivered annotation with ONLY located clusters relabelled.

        Not a re-annotation and not a coarser reading: a rare cell type a unit lacks and another
        unit carries is looked for in the units that lack it, by clustering those units more
        finely, and where a cluster comes back as the target THAT CLUSTER'S CELLS take the
        label. Every other cell keeps what it was delivered with, so the difference between this
        block and the one it corrects is exactly the rescued set and nothing else.
        """
        return self._rows_over("rescue") if self.has_rescue else None

    def forced_l1_rows(self):
        """THE L1 ANNOTATION, FORCED — the independent depth-1 walk with nothing left unresolved."""
        return self._rows_over("forced_l1") if self.has_forced_l1 else None

    def forced_moved(self, col="forced", against="path"):
        """How many nuclei the forcing moved, and where to. The number the block is really about."""
        if col not in self.P or against not in self.P:
            return {"n": 0, "moves": {}}
        a = self.P[against].astype(str)
        b = self.P[col].astype(str)
        m = (a != b) & (b != "")
        moves = {}
        for x, y in zip(a[m], b[m]):
            moves[f"{x} -> {y}"] = moves.get(f"{x} -> {y}", 0) + 1
        return {"n": int(m.sum()),
                "moves": dict(sorted(moves.items(), key=lambda kv: -kv[1]))}

    def l1_rows(self):
        """THE L1 ANNOTATION — the INDEPENDENT depth-1 walk, or None where it was never written.

        NOT `path[:1]`. The independent walk is a separate column produced by walking a depth-1
        tree, so no seal at any depth can move it; a truncation of the scope path inherits every
        edit the scope made. Rendering a truncation under this name would manufacture perfect
        agreement between the two delivered columns for exactly the objects that measured none —
        which is the reason this class refuses to back-fill `l1` from the path in the first place.
        """
        return self._rows_over("l1") if self.has_l1 else None

    def label_order(self, depth):
        return self._order[min(max(int(depth), 1), self.depth)]

    def colour(self, label):
        return self.palette.of(label)

    def colours(self, depth):
        return {l: self.palette.of(l) for l in self.label_order(depth)}

    def count(self, label, depth):
        return int((self.P[f"L{depth}"] == label).sum())

    def share(self, label, depth):
        return 100.0 * self.count(label, depth) / max(self.n, 1)

    def mask_label(self, label, depth):
        return np.asarray(self.P[f"L{depth}"] == label)

    def is_truncated(self, label, depth):
        """A label shorter than its level is a call the walk STOPPED on, not a subtype it chose.
        Distinguishing the two is why `truncated` and `unresolved` are separate columns."""
        s = str(label)
        return s not in SENTINELS and len(s.split("/")) < depth

    # ------------------------------------------------------------------ composition
    def label_order_for(self, col):
        """Abundance order over an arbitrary label column, sentinels last.

        The depth-keyed `label_order` cannot serve a forced column: those hold full paths at mixed
        depths and are not a level of the taxonomy.
        """
        if col not in self.P:
            return []
        v = self.P[col].astype(str)
        counts = {}
        for lab in v:
            if lab != "":
                counts[lab] = counts.get(lab, 0) + 1
        return sorted(counts, key=lambda l: (l in SENTINELS, -counts[l], l))

    def colours_for(self, col):
        """One colour per label of `col`, from the SAME palette the depth figures use — so a cell
        type keeps its colour between a figure and its forced twin, which is the only way the
        pair can be read against each other."""
        return {l: self.palette.of(l) for l in self.label_order_for(col)}

    def composition_rows(self, depth, by="group", col=None, order=None):
        """One row per group or per sample, as percentages WITHIN that row.

        Computed per sample and averaged for a group row, never pooled before division - pooling
        lets the largest library set the group's composition.

        `col`/`order` override the depth-derived label column, which is what lets the forced
        columns reuse this rather than growing a second, drifting copy of the averaging rule.
        """
        lab_col = col or f"L{depth}"
        lab_order = order if order is not None else self.label_order(depth)
        key = "group" if by == "group" else "sample"
        if key not in self.P or lab_col not in self.P:
            return []
        rows = []
        for name in self._levels(key):
            m = self.P[key] == name
            if by == "group":
                pcts = {}
                sams = sorted(set(self.P.loc[m, "sample"]))
                for lab in lab_order:
                    vals = [100.0 * float(((self.P["sample"] == s) & m &
                                           (self.P[lab_col] == lab)).sum())
                            / max(int((self.P["sample"] == s).sum()), 1) for s in sams]
                    pcts[lab] = float(np.mean(vals)) if vals else 0.0
            else:
                tot = max(int(m.sum()), 1)
                pcts = {lab: 100.0 * float((m & (self.P[lab_col] == lab)).sum()) / tot
                        for lab in lab_order}
            rows.append({"name": str(name), "n": int(m.sum()), "pct": pcts})
        return rows

    def _levels(self, key):
        """The levels of a column, in the order they should be READ.

        For the experimental group that is the order the caller declared - a 2x2 is read
        young/aged x chow/HFD, and alphabetical order interleaves the two factors so that
        neighbouring rows differ in both at once, which is the one arrangement in which no
        pair of adjacent rows is a comparison. A level the caller did not name is APPENDED
        and reported, never dropped: silently omitting an arm turns a missing declaration
        into a missing population.
        """
        if key not in self.P:
            return []
        have = set(str(v) for v in self.P[key])
        if key == "group" and self._group_order:
            named = [g for g in self._group_order if g in have]
            self.group_order_unnamed = sorted(have - set(named))
            return named + self.group_order_unnamed
        return sorted(have)

    def _detect_factors(self):
        """Low-cardinality obs columns that could be a design factor. Reported as auto-detected
        wherever used: guessing the design is worth doing and never worth hiding."""
        out = []
        for c in self.P.columns:
            if c in NOT_A_FACTOR or c.startswith("L") and c[1:].isdigit():
                continue
            if c in QC_COLUMNS or c in ("depth", "gap", "support", "survival"):
                continue
            try:
                lv = set(self.P[c].astype(str))
            except Exception:                                             # noqa: BLE001
                continue
            if 2 <= len(lv) <= 6 and len(lv) < max(len(self.samples), 2):
                out.append(c)
        return out

    def per_animal_points(self, depth, col=None, order=None):
        """{(group, label): [one % per animal]} and the group order. The points of F141/F143.

        `col`/`order` override the depth-derived label column, so a forced column gets the same
        per-animal figure rather than a block with one figure fewer than the block above it.
        """
        if "group" not in self.P:
            return {}, []
        lab_col = col or f"L{depth}"
        lab_order = order if order is not None else self.label_order(depth)
        if lab_col not in self.P:
            return {}, []
        pts, groups = {}, self._levels("group")
        for s in self._levels("sample"):
            ms = self.P["sample"] == s
            g = str(self.P.loc[ms, "group"].iloc[0])
            tot = max(int(ms.sum()), 1)
            for lab in lab_order:
                v = 100.0 * float((ms & (self.P[lab_col] == lab)).sum()) / tot
                pts.setdefault((g, lab), []).append(v)
        return pts, groups

    def animals_with(self, label, depth):
        """How many samples the label appears in. A label seen in one animal of ten is not a
        population a downstream comparison can use."""
        return sum(1 for s in self._levels("sample")
                   if ((self.P["sample"] == s) & (self.P[f"L{depth}"] == label)).any())

    # ------------------------------------- the delivered annotation: both columns, together
    def l1_concordance(self):
        """Do the two DELIVERED columns agree about the compartment? Measured, never assumed.

        The independent L1 is a second walk. Nothing constrains it to return the same root the
        deep walk returned, and the value of the comparison is exactly that nothing does — a
        run where the two agree everywhere has demonstrated the root decision is stable, and a
        run where they differ has found the cells whose compartment is not settled.

        Sentinels are scored where BOTH routes emit them and dropped where only one does: a
        nucleus one route withheld and the other annotated is a guaranteed mismatch that
        measures the exclusion rather than the agreement.

        Returns None when there is no independent column at all — which is not "they agreed".
        """
        if not self.has_l1:
            return None
        own = self.P["L1"].astype(str)
        ind = self.P["l1"].astype(str)
        scored = (ind != "")
        n = int(scored.sum())
        if not n:
            return None
        a, b = own[scored], ind[scored]
        agree = (a == b)
        pairs = {}
        for x, y in zip(a[~agree], b[~agree]):
            pairs[(x, y)] = pairs.get((x, y), 0) + 1
        return {"n_scored": n, "n_agree": int(agree.sum()),
                "n_disagree": int((~agree).sum()),
                "pct": 100.0 * float(agree.mean()),
                "pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
                "column": self.l1_key}

    def l1_bands(self):
        """The delivered annotation, grouped by the INDEPENDENT L1 column.

        Returns one band per value of that column, ordered by size, each carrying the rows of the
        scope-based taxonomy those nuclei fall in. A row is a node of the OBSERVED taxonomy —
        built from the paths actually delivered, never from the declared tree — so the section
        describes what was produced rather than what was asked for.

        Rows come out in a depth-first walk of the observed prefix tree with siblings ordered by
        size, which is what preserves the taxonomy's shape in a flat list. Each row carries:

          depth     how deep it sits, for indentation. NOT assumed to be 1, 2 or 3.
          n_here    nuclei whose delivered label is EXACTLY this path — a terminal count
          n_below   nuclei at or below it — a subtotal
          samples   how many samples have any nucleus terminating here

        A row with `n_here == 0` is a guide: an intermediate node nothing terminates at, kept
        because deleting it would flatten a three-level branch into a list of unrelated names.
        """
        if not self.has_l1:
            return []
        import pandas as pd

        d = pd.DataFrame({"l1": self.P["l1"].astype(str),
                          "path": self.P["path"].astype(str),
                          "sample": self.P["sample"].astype(str)})
        d = d[d["l1"] != ""]
        bands = []
        for band, sub in sorted(d.groupby("l1"), key=lambda kv: (-len(kv[1]), kv[0])):
            here = sub.groupby("path").size().to_dict()
            samp = sub.groupby("path")["sample"].nunique().to_dict()
            # Every prefix of every delivered path, so an intermediate node with no terminal
            # cells of its own still appears and the branch keeps its shape.
            below = {}
            for p, n in here.items():
                parts = str(p).split("/")
                for i in range(1, len(parts) + 1):
                    below["/".join(parts[:i])] = below.get("/".join(parts[:i]), 0) + n
            rows = []

            def walk(prefix, depth):
                kids = sorted({k for k in below
                               if k.startswith(prefix) and k.count("/") == depth - 1},
                              key=lambda k: (-below[k], k))
                for k in kids:
                    rows.append({"path": k, "label": k.rsplit("/", 1)[-1], "depth": depth,
                                 "n_here": int(here.get(k, 0)), "n_below": int(below[k]),
                                 "samples": int(samp.get(k, 0))})
                    walk(k + "/", depth + 1)

            walk("", 1)
            bands.append({"l1": band, "n": int(len(sub)),
                          "samples": int(sub["sample"].nunique()), "rows": rows})
        return bands

    # ------------------------------------------------------------------ reliability
    def calls_at_chosen(self):
        """One row per (sample, cluster) at the chosen resolution, with its statistics."""
        import pandas as pd
        rows = []
        has_depth = "depth" in self.P
        for (obj, path), sub in self.P.groupby(["_obj", "path"], observed=True):
            # Depth from the PATH when no column carries it. A sweep writes gap, support and
            # survival per resolution but commonly not depth, and requiring the column removed
            # the whole reliability section from a run that had every statistic it needed.
            dv = sub["depth"].iloc[0] if has_depth else float("nan")
            if dv != dv:
                dv = (0 if str(path) in SENTINELS else len(str(path).split("/")))
            rows.append({"sample": obj, "path": path,
                         "depth": int(dv),
                         "n_cells": int(len(sub)),
                         "gap": float(sub["gap"].iloc[0]) if "gap" in sub else np.nan,
                         "support": float(sub["support"].iloc[0]) if "support" in sub else np.nan,
                         "survival": (float(sub["survival"].iloc[0]) if "survival" in sub
                                      else np.nan)})
        return pd.DataFrame(rows)

    def reliability_rows(self):
        c = self.calls_at_chosen()
        if c is None or not len(c) or not any(
                c[k].notna().any() for k in ("gap", "support", "survival") if k in c):
            return []
        out = []
        for d in sorted(set(c["depth"])):
            m = c["depth"] == d
            cells = int(c.loc[m, "n_cells"].sum())
            sup = c.loc[m, "support"]
            out.append({
                "depth": int(d), "calls": int(m.sum()), "nuclei": cells,
                "pct": round(100.0 * cells / max(self.n, 1), 1),
                "median_gap": _r(c.loc[m, "gap"].median(), 2),
                "median_support": _r(sup.median(), 0),
                "median_survival": _r(c.loc[m, "survival"].median(), 2),
                "pct_thin": (None if sup.isna().all()
                             else round(100.0 * float((sup < 10).mean()), 0))})
        return out

    def worst_evidence(self, n=6):
        """Calls won on the most depleted panels - a high gap on low survival is a confident
        call on evidence better-cited neighbours had already taken."""
        c = self.calls_at_chosen()
        if c is None or "survival" not in c or c["survival"].isna().all():
            return []
        c = c[~c["survival"].isna()].sort_values(["survival", "support"]).head(n)
        return [{"sample": r["sample"], "path": r["path"], "gap": _r(r["gap"], 2),
                 "support": _r(r["support"], 0), "survival": _r(r["survival"], 2)}
                for _, r in c.iterrows()]

    def neighbourhood(self):
        return (np.asarray(self.P["nn_agreement"], dtype=float)
                if "nn_agreement" in self.P else None)

    def neighbourhood_by_label(self, depth, floor=50):
        """Per-nucleus kNN agreement grouped by label. Labels under `floor` nuclei are omitted -
        a box drawn from nine cells is a shape, not a distribution."""
        nn = self.neighbourhood()
        if nn is None:
            return [], []
        keys, data = [], []
        for l in self.label_order(depth):
            m = np.asarray(self.P[f"L{depth}"] == l)
            if m.sum() < floor:
                continue
            v = nn[m]
            v = v[~np.isnan(v)]
            if v.size:
                keys.append(l)
                data.append(v * 100.0)
        return keys, data

    def manifold_rows(self, n=6):
        """Where the annotation leaves the manifold: the lowest-agreement clusters."""
        nn = self.neighbourhood()
        if nn is None:
            return []
        rows = []
        for (obj, path), sub in self.P.groupby(["_obj", "path"], observed=True):
            v = np.asarray(sub["nn_agreement"], dtype=float)
            rows.append({"cluster": f"{obj}", "label": str(path), "nuclei": int(len(sub)),
                         "agreement": round(100 * float(np.nanmean(v)), 1)})
        return sorted(rows, key=lambda r: r["agreement"])[:n]

    # ------------------------------------------------------------------ resolution
    def resolution_sweep(self):
        return self._sweep

    def resolution_band(self, depth, key):
        rows = self._sweep.get(depth) or []
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals or self._tolerance is None or self.chosen_resolution is None:
            return None
        best = max(vals)
        return (best - float(self._tolerance), best)

    @property
    def tolerance(self):
        return self._tolerance

    # ------------------------------------------------------------------ expression
    def _matrix(self, name):
        for n, A in self.objects:
            if n == name:
                return A
        return None

    def has_gene(self, g):
        return str(g).upper() in self._gene_index()

    def gene_names(self, A=None):
        """The names a marker panel is written in: SYMBOLS, not accessions.

        A corpus is curated in symbols and an object is commonly indexed by Ensembl accession.
        Matching a symbol panel against `var_names` there finds nothing, and the failure is
        quiet in both directions - the panel reports every gene as absent from the object, and
        an annotator reports every cluster as UNRESOLVED, both with exit status zero. So the
        symbol column is preferred wherever one exists, and which column was used is reported.
        """
        A = A if A is not None else self.objects[0][1]
        col = self.gene_key
        if col is None:
            for cand in ("gene_symbol", "gene_symbols", "symbol", "feature_name", "gene_name"):
                if cand in A.var:
                    col = cand
                    break
        self.gene_column = col
        src = A.var[col] if col and col in A.var else A.var_names
        return [str(v).upper() for v in src]

    def _gene_index(self):
        if not hasattr(self, "_gi"):
            self._gi = {}
            for i, g in enumerate(self.gene_names()):
                self._gi.setdefault(g, i)
        return self._gi

    def panels(self, depth):
        return self._panels.get(int(depth)) or {}

    def panels_for(self, labels, sep="/"):
        """{label: [genes]} taking each label's panel from the depth it TERMINATES at.

        WHY THIS IS NOT `panels(depth)`. The marker panels are keyed by depth, and the scope
        annotation is MIXED across depths by construction - a sealed node terminates at 2 while
        its cousins reach 3. Asking for the panels of the deepest level therefore returns nothing
        for every label that stops short, and those labels appear in the dotplot as rows with no
        gene columns: the figure looks complete and silently omits the evidence for exactly the
        labels the cohort's vote created.

        So the panel is looked up per LABEL, at its own depth, falling back to any depth that
        declares it. A label the corpus has no panel for is simply absent from the result, and
        the caller reports it rather than drawing an empty block.
        """
        out = {}
        for l in labels:
            if l in SENTINELS:
                continue
            d = len([x for x in str(l).split(sep) if x])
            g = (self._panels.get(d) or {}).get(l)
            if not g:
                for dd in sorted(self._panels):
                    if l in (self._panels[dd] or {}):
                        g = self._panels[dd][l]
                        break
            if g:
                out[l] = list(g)
        return out

    def panel_depths(self):
        return sorted(d for d in self._panels if self._panels[d])

    def obs_column_for(self, key):
        """Internal frame key -> the column name on the ACTUAL objects.

        `self.P` uses short internal names - `path`, `l1`, `forced`, `forced_l1` - while `A.obs`
        carries whatever the annotation wrote: `scanno_path_scope`, `scAnno_L1_resolved_scope`
        and so on. Anything that reads the objects rather than `P` must translate, and a lookup
        that fails silently produces an all-zero matrix and a figure that renders EMPTY.
        """
        return {"path": self.path_key,
                "l1": self.l1_key,
                "forced": self.forced_key,
                "forced_l1": self.forced_l1_key}.get(key, key)

    def expression_by_label(self, genes, depth, labels, col=None):
        """(fraction detecting, mean among all) per label x gene, pooled over objects.

        Values are log1p(CP10K) computed here from counts if a counts layer is present, so a
        dotplot never mixes raw and normalised objects - which reads as a real difference
        between labels and is a difference between how two files were saved.
        """
        import scipy.sparse as sp
        gi = self._gene_index()
        cols = [gi[str(g).upper()] for g in genes]
        missing_in = []
        frac = np.zeros((len(labels), len(genes)))
        mean = np.zeros((len(labels), len(genes)))
        tot = np.zeros(len(labels))
        for name, A in self.objects:
            # `col` reads a DELIVERED column verbatim; without it the path is truncated to
            # `depth`, which is right for a level figure and wrong for the scope, whose labels
            # live at mixed depths and are not a truncation of anything.
            if col:
                ocol = self.obs_column_for(col)
                if ocol not in A.obs:
                    missing_in.append(name)
                    continue
                lab = A.obs[ocol].astype(str).values
            else:
                lab = self._trunc(A.obs[self.path_key].astype(str), depth)
            X = A.layers["counts"] if "counts" in getattr(A, "layers", {}) else A.X
            for li, l in enumerate(labels):
                m = lab == l
                if not m.any():
                    continue
                rows = np.flatnonzero(m)
                sub = X[rows][:, cols]
                sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
                sub = lognorm(sub, _row_totals(X, rows))
                frac[li] += (sub > 0).sum(axis=0)
                mean[li] += sub.sum(axis=0)
                tot[li] += m.sum()
        if col and len(missing_in) == len(self.objects):
            # EVERY object lacked the column. Returning zeros here renders a complete-looking
            # dotplot with no dots in it, which is worse than an error: it reads as "these genes
            # are not expressed" rather than "this figure asked for a column that is not there".
            raise KeyError(
                f"no object carries the obs column {self.obs_column_for(col)!r} (internal key "
                f"{col!r}); the first object has: {list(self.objects[0][1].obs.columns)[:12]}")
        nz = np.where(tot == 0, 1, tot)[:, None]
        return frac / nz, mean / nz

    def marker_breadth(self, depth=1):
        """Detection within a gene's own label against the best OTHER label. Measured here,
        because a corpus's own specificity cannot tell promiscuity from several names for one
        lineage."""
        panels = self.panels(depth)
        picks = [(l, g) for l in self.label_order(depth) for g in (panels.get(l) or [])
                 if self.has_gene(g)]
        if not picks:
            return []
        genes = list(dict.fromkeys(g for _, g in picks))
        labels = [l for l in self.label_order(depth) if l not in SENTINELS]
        frac, _ = self.expression_by_label(genes, depth, labels)
        out = []
        for own, g in picks:
            j = genes.index(g)
            if own not in labels:
                continue
            i = labels.index(own)
            others = [frac[k, j] for k in range(len(labels)) if k != i]
            w = [self.count(l, depth) for l in labels]
            overall = float(np.average(frac[:, j], weights=w)) if sum(w) else 0.0
            out.append({"gene": g, "plotted_for": own, "own": float(frac[i, j]),
                        "best_other": float(max(others) if others else 0.0),
                        "overall": overall,
                        "labels_over_25": int(sum(1 for k in range(len(labels))
                                                  if frac[k, j] >= 0.25))})
        return sorted(out, key=lambda r: (-r["labels_over_25"], -r["overall"]))

    # ------------------------------------------------------------------ the joint route
    def embedding_is_stitched(self, tol=1e-3):
        """Is the joint embedding actually the per-sample embeddings, moved apart?

        A cohort object assembled by concatenating per-sample objects inherits their `X_umap`
        unchanged, and some assemblers translate each sample's block so the clouds do not
        overlap. The result LOOKS like a joint embedding and is not one: every sample occupies
        its own territory because it was PUT there, so any figure asking whether structure
        follows libraries answers yes by construction. That is the exact reading F132 exists
        to support, and on a stitched embedding it is a tautology dressed as a finding.

        The test is decisive rather than statistical: take each sample's rows out of the joint
        object by BARCODE and subtract that sample's own coordinates. If every sample's
        difference is constant - a rigid translation - the coordinates were never recomputed.
        A genuine joint embedding of the same cells differs everywhere.

        Returns None when it cannot be tested, else a dict naming what was found.
        """
        if self.joint is None or self.joint_embedding() is None:
            return None
        import numpy as np
        xy = self.joint_embedding()
        jb = np.asarray(self.joint.obs_names.astype(str))
        js = (np.asarray(self.joint.obs[self.sample_key].astype(str))
              if self.sample_key in self.joint.obs else None)
        if js is None:
            return None
        rigid, tested = [], []
        for name, A in self.objects:
            own = self.embedding(name)
            if own is None:
                continue
            idx = {b: i for i, b in enumerate(np.asarray(A.obs_names.astype(str)))}
            m = js == str(name)
            if m.sum() < 10:
                continue
            bs = jb[m]
            keep = np.array([b in idx for b in bs])
            if keep.sum() < 10:
                continue
            take = np.array([idx[b] for b in bs[keep]])
            d = xy[m][keep] - own[take]
            tested.append(name)
            rigid.append(bool(max(float(d[:, 0].std()), float(d[:, 1].std())) < tol))
        if not tested:
            return None
        return {"stitched": all(rigid) and len(rigid) > 1,
                "samples_tested": tested, "samples_rigid": int(sum(rigid))}

    def joint_embedding(self):
        if self.joint is None:
            return None
        for k in self.joint.obsm:
            if any(h in k.lower() for h in ("umap", "tsne", "draw_graph")):
                return np.asarray(self.joint.obsm[k])[:, :2]
        return None

    def joint_labels(self, depth=1):
        if self.joint is None:
            return None
        col = self.joint_key or self.path_key
        if col not in self.joint.obs:
            return None
        return self._trunc(self.joint.obs[col].astype(str), depth)

    def joint_samples(self):
        o = self.joint.obs
        return (np.asarray(o[self.sample_key].astype(str)) if self.sample_key in o
                else np.full(self.joint.n_obs, "all"))

    def joint_agreement(self, depth):
        """Share of nuclei the two routes give the same label at this depth.

        ALIGNED BY BARCODE, never by position. A joint object is a different clustering of the
        same cells and there is no guarantee it was written in the concatenation order of the
        per-sample objects; comparing positionally then measures a shuffle.

        REFUSES when the joint column and the per-sample column are the SAME COLUMN. A joint
        object assembled from the per-sample annotations usually carries those columns too, so
        the default key finds the per-sample labels sitting in the joint object and the routes
        agree with themselves - which reports as 100% and reads as a spectacular result. Name
        the joint route's own column with `joint_key`.

        Scored over the nuclei BOTH routes annotated. Leaving the withheld nuclei in makes them
        agree with themselves and inflates the number by their share of the cohort; when only
        one route withholds them, every one is a guaranteed mismatch. Either way the statistic
        measures the exclusion rather than the agreement it exists to measure.
        """
        if self.joint is None:
            return None
        col = self.joint_key or self.path_key
        if col not in self.joint.obs:
            return {"error": f"the joint object has no obs column {col!r}"}
        if not self.joint_key:
            return {"error": "no --joint-key given, so the joint route's own label column is "
                             "unknown. Comparing the default key would compare the per-sample "
                             "labels with themselves and report ~100%"}
        # The name check above stops being enough once the joint object RENAMES its label columns.
        # A column called `cell_type` that is a copy of `scanno_path_scope` passes every check
        # keyed on the name and then agrees with itself at ~100%. `scanno embed` records the
        # rename in uns, so read the provenance rather than trusting the name.
        _renamed = dict((self.joint.uns.get("scanno_embed") or {}).get("label_map") or {})
        if _renamed.get(col) == self.path_key:
            return {"error": f"the joint object's {col!r} is a RENAME of {self.path_key!r} - the "
                             f"same per-sample labels under another name, recorded in "
                             f"uns['scanno_embed']['label_map']. Comparing them would report "
                             f"~100% agreement of the labels with themselves. A two-route "
                             f"agreement needs the joint object annotated INDEPENDENTLY."}
        jl = self.joint_labels(depth)
        if jl is None:
            return None
        import pandas as pd
        js = pd.Series(jl, index=self.joint.obs_names.astype(str))
        js = js[~js.index.duplicated()]
        own = pd.Series(np.asarray(self.P[f"L{depth}"]), index=self.P.index.astype(str))
        shared = own.index.intersection(js.index)
        if not len(shared):
            return {"error": "the two routes share no barcodes; they are not the same cells"}
        a, b = own.loc[shared].astype(str), js.loc[shared].astype(str)
        scored = ~(a.str.startswith(EXCLUDED) | b.str.startswith(EXCLUDED))
        if not scored.any():
            return None
        return {"pct": 100.0 * float((a[scored] == b[scored]).mean()),
                "n_scored": int(scored.sum()), "n_matched": int(len(shared)),
                "n_total": int(self.n), "joint_key": col}

    def joint_expression(self, gene):
        import scipy.sparse as sp
        j = self._gene_index()[str(gene).upper()]
        v = self.joint.X[:, j]
        return np.asarray(v.todense()).ravel() if sp.issparse(v) else np.asarray(v).ravel()

    # ------------------------------------------------------------------ per sample
    def embedding(self, sample):
        A = self._matrix(sample)
        if A is None:
            return None
        for k in getattr(A, "obsm", {}) or {}:
            if any(h in k.lower() for h in ("umap", "tsne", "draw_graph")):
                return np.asarray(A.obsm[k])[:, :2]
        return None

    def sweep_keys(self, sample):
        """The per-resolution label columns this object actually carries, in numeric order."""
        A = self._matrix(sample)
        if A is None:
            return []
        stem = sweep_stem(self.path_key)
        out = []
        for c in A.obs.columns:
            if c.startswith(stem + "_r"):
                tag = c[len(stem) + 2:]
                try:
                    out.append((float(tag.replace("p", ".")), c, tag))
                except ValueError:
                    continue
        return sorted(out)

    def labels(self, sample, depth, resolution=None):
        A = self._matrix(sample)
        if A is None:
            return None
        key = self.path_key
        if resolution is not None:
            stem = sweep_stem(self.path_key)
            tag = str(resolution).replace(".", "p")
            for cand in (f"{stem}_r{tag}", f"scanno_path_r{tag}"):
                if cand in A.obs:
                    key = cand
                    break
        if key not in A.obs:
            return None
        return self._trunc(A.obs[key].astype(str), depth)

    def sample_depth(self, sample):
        """The deepest label this one sample reaches. Panels are titled with it, so a sample
        whose tree stops shallower than the cohort's says so rather than showing empty axes."""
        A = self._matrix(sample)
        if A is None or self.path_key not in A.obs:
            return 1
        return max((len(str(p).split("/")) for p in A.obs[self.path_key].astype(str)
                    if str(p) not in SENTINELS), default=1)

    def clusters(self, sample, resolution=None):
        A = self._matrix(sample)
        if A is None:
            return None
        stem = sweep_stem(self.path_key)
        raw = str(resolution if resolution is not None else self.chosen_resolution or "")
        # A resolution appears in a column name spelled at least three ways - `leiden_1.0`,
        # `leiden_1p0`, `leiden_r1p0` - depending on which tool wrote it, and `1.0` and `1`
        # are the same number written two ways. Missing the spelling loses the whole figure
        # and reports it as "no cluster column", which reads as an object that was never
        # clustered rather than one whose column we failed to name.
        tags = {raw, raw.replace(".", "p"), raw.rstrip("0").rstrip("."), }
        try:
            f = float(raw)
            tags |= {f"{f:g}", f"{f:g}".replace(".", "p"), f"{f}", f"{f}".replace(".", "p")}
        except ValueError:
            pass
        cands = []
        for t in sorted(tags):
            if not t:
                continue
            cands += [f"leiden_r{t}", f"leiden_{t}", f"{stem}_cluster_r{t}",
                      f"{stem}_cluster_{t}", f"cluster_r{t}", f"cluster_{t}"]
        cands += ["leiden", "cluster", "clusters"]
        for cand in cands:
            if cand in A.obs:
                return np.asarray(A.obs[cand].astype(str))
        return None

    def depths(self, sample):
        A = self._matrix(sample)
        k = self.label_key.replace("_cell_type", "_depth")
        return np.asarray(A.obs[k], dtype=float) if k in A.obs else None

    def flag(self, sample):
        import pandas as pd
        A = self._matrix(sample)
        if not self.flag_column or self.flag_column not in A.obs:
            return np.zeros(A.n_obs, dtype=bool)
        return np.asarray(pd.Series(A.obs[self.flag_column]).astype("boolean")
                          .fillna(False).to_numpy(dtype=bool))

    def sample_rows(self, sample):
        """The pooled frame restricted to one sample, for the per-sample document.

        Matched on `_obj` FIRST — the object's own name, which is what every per-sample figure
        is called with. `obs["sample"]` is a different string whenever the file is named for
        anything but its sample: `<sample>.filtered.h5ad` gives an object named `<sample>.filtered`
        while its obs says `<sample>`, and matching on obs alone returned an EMPTY frame. The
        figures then reported "no QC columns in obs" for ten samples whose objects carried all
        four — a named absence that was a lookup failure wearing a finding's clothes.
        """
        by_obj = self.P[self.P["_obj"] == str(sample)]
        if len(by_obj):
            return by_obj
        by_sample = self.P[self.P["sample"] == str(sample)]
        if len(by_sample):
            return by_sample
        if str(sample) in {str(x) for x in self.samples}:
            # The object EXISTS and we failed to find its rows. That is a defect in this code,
            # not a property of the data, and it must never reach a figure as a named absence -
            # the page would then report a statement about the cohort that is really a statement
            # about a string comparison.
            raise LookupError(
                f"sample_rows({sample!r}) found no rows, but {sample!r} is one of this "
                f"context's objects. Object keys: {sorted({str(x) for x in self.P['_obj']})[:4]}; "
                f"obs sample values: {sorted({str(x) for x in self.P['sample']})[:4]}. "
                f"This is a bug in scAnno, not a gap in the data.")
        return by_sample

    # ------------------------------------------------------------------ the flagged nuclei
    def flag_per_animal(self):
        if not self.has_flag:
            return []
        out = []
        for s in self._levels("sample"):
            m = self.P["sample"] == s
            f = int(self.P.loc[m, "flag"].sum())
            g = str(self.P.loc[m, "group"].iloc[0]) if "group" in self.P else ""
            out.append({"animal": s, "arm": g, "nuclei": int(m.sum()), "flagged": f,
                        "rate": round(100.0 * f / max(int(m.sum()), 1), 2),
                        "in_comparison": f >= MIN_FLAGGED_PER_ANIMAL})
        return sorted(out, key=lambda r: (r["arm"], r["animal"]))

    def flag_by_factor(self, factors=None):
        """Rule one Q3, per design factor: is the removal differential across the design?"""
        if not self.has_flag:
            return []
        use = list(factors) if factors else list(self.factors)
        if "group" in self.P and "group" not in use:
            use.append("group")
        out = []
        for fac in use:
            if fac not in self.P:
                continue
            rates = {}
            for lev in sorted(set(self.P[fac].astype(str))):
                m = self.P[fac].astype(str) == lev
                rates[str(lev)] = 100.0 * float(self.P.loc[m, "flag"].mean())
            if not rates:
                continue
            nz = [v for v in rates.values() if v > 0]
            out.append({"factor": fac, "rates": {k: round(v, 2) for k, v in rates.items()},
                        "lo": round(min(rates.values()), 2), "hi": round(max(rates.values()), 2),
                        "auto": fac in self.auto_factors,
                        "ratio": round(max(rates.values()) / min(nz), 2) if nz else float("inf")})
        return sorted(out, key=lambda r: -r["ratio"])

    @property
    def n_compare_animals(self):
        return sum(1 for r in self.flag_per_animal() if r["in_comparison"])

    @property
    def n_genes_ranked(self):
        return getattr(self, "_n_ranked", 0)

    def flag_identity(self, depth=1):
        """What the flagged nuclei WOULD have been called, if a label survives beside the flag.

        Measured rather than inferred: without it a reader has to guess whether an exclusion fell
        on one population, and guessing is how a cardiomyocyte-selective filter goes unnoticed.
        """
        if not self.has_flag:
            return []
        m = np.asarray(self.P["flag"])
        lab = np.asarray(self.P[f"L{depth}"])
        tot = {}
        for l in self.label_order(depth):
            tot[l] = int((lab == l).sum())
        rows = []
        for l in self.label_order(depth):
            k = int((m & (lab == l)).sum())
            if k:
                rows.append({"label": l, "flagged": k, "of_label": tot[l],
                             "pct_of_label": round(100.0 * k / max(tot[l], 1), 2),
                             "pct_of_flagged": round(100.0 * k / max(int(m.sum()), 1), 2)})
        return sorted(rows, key=lambda r: -r["flagged"])

    def exclusion_signature(self, floor=DETECT_FLOOR):
        """Genes ranked by Δ mean expression, flagged − kept. Effect size, not a p-value: at
        these group sizes every gene is significant and a p-value ranks genes by abundance."""
        if hasattr(self, "_sig_cache"):
            return self._sig_cache
        if not self.has_flag:
            self._sig_cache = []
            return []
        import scipy.sparse as sp
        # POSITIONAL, not the deduplicated index. `_gene_index` maps a symbol to its FIRST
        # column, so its length is the number of unique symbols - 34,247 where the matrix has
        # 34,290 columns, because symbols repeat. Sizing the accumulators from it makes every
        # `+=` a broadcast error, and would have silently mislabelled every gene past the first
        # duplicate if the shapes had happened to match.
        genes = self.gene_names()
        nF = nK = 0
        sF = np.zeros(len(genes)); sK = np.zeros(len(genes))
        dF = np.zeros(len(genes)); dK = np.zeros(len(genes))
        per_animal = {}
        for name, A in self.objects:
            f = self.flag(name)
            if not f.any():
                continue
            X = A.layers["counts"] if "counts" in getattr(A, "layers", {}) else A.X
            rf, rk = np.flatnonzero(f), np.flatnonzero(~f)
            sub_f, sub_k = X[rf], X[rk]
            sub_f = sub_f.toarray() if sp.issparse(sub_f) else np.asarray(sub_f)
            sub_k = sub_k.toarray() if sp.issparse(sub_k) else np.asarray(sub_k)
            # Every gene is present here, so the block's own row sums ARE the row totals.
            sub_f = lognorm(sub_f, sub_f.sum(axis=1))
            sub_k = lognorm(sub_k, sub_k.sum(axis=1))
            sF += sub_f.sum(axis=0); dF += (sub_f > 0).sum(axis=0); nF += sub_f.shape[0]
            sK += sub_k.sum(axis=0); dK += (sub_k > 0).sum(axis=0); nK += sub_k.shape[0]
            if f.sum() >= MIN_FLAGGED_PER_ANIMAL:
                per_animal[name] = (sub_f.mean(axis=0) - sub_k.mean(axis=0),
                                    (sub_f > 0).mean(axis=0), (sub_k > 0).mean(axis=0))
            del sub_f, sub_k
        if nF == 0:
            self._sig_cache = []
            return []
        keep = ((dF / max(nF, 1)) >= floor) | ((dK / max(nK, 1)) >= floor)
        self._n_ranked = int(keep.sum())
        d_mean = sF / max(nF, 1) - sK / max(nK, 1)
        d_det = 100.0 * (dF / max(nF, 1) - dK / max(nK, 1))
        rows = []
        for j in np.flatnonzero(keep):
            agree = sum(1 for v in per_animal.values()
                        if d_mean[j] != 0 and np.sign(v[0][j]) == np.sign(d_mean[j]))
            rows.append({"gene": genes[j], "d_mean": float(d_mean[j]),
                         "d_detect": float(d_det[j]),
                         "det_flagged": float(dF[j] / max(nF, 1)),
                         "det_kept": float(dK[j] / max(nK, 1)),
                         "animals_agree": int(agree),
                         "animals_compared": len(per_animal)})
        rows.sort(key=lambda r: -r["d_mean"])
        self._sig_cache = rows
        self._per_animal_sig = per_animal
        return rows

    def signature_per_animal(self, top=12):
        sig = self.exclusion_signature()
        if not sig or len(getattr(self, "_per_animal_sig", {})) < 2:
            return None
        import scipy.sparse as sp
        gi = self._gene_index()
        genes = [g for g in dict.fromkeys([r["gene"] for r in sig[:top]]
                                          + [r["gene"] for r in sig[-max(top // 2, 1):]])
                 if g in gi]
        if not genes:
            return None
        cols_idx = [gi[g] for g in genes]
        animals = sorted(self._per_animal_sig)
        cols, frac, mean = [], [], []
        for a in animals:
            A = self._matrix(a)
            f = self.flag(a)
            X = A.layers["counts"] if "counts" in getattr(A, "layers", {}) else A.X
            for which, m in (("flagged", f), ("kept", ~f)):
                rows = np.flatnonzero(m)
                sub = X[rows][:, cols_idx]
                sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
                sub = lognorm(sub, _row_totals(X, rows))
                cols.append((a, which))
                frac.append((sub > 0).mean(axis=0))
                mean.append(sub.mean(axis=0))
        F = np.array(frac).T
        M = np.array(mean).T
        rng_ = M.max(axis=1) - M.min(axis=1)
        M = (M - M.min(axis=1)[:, None]) / np.where(rng_ == 0, 1, rng_)[:, None]
        return genes, cols, F, M

    def exclusion_qc(self):
        if not self.has_flag:
            return []
        out = []
        for col, name, log in (("doublet_score", "doublet score", False),
                               ("pct_counts_mt", "% mitochondrial", False),
                               ("total_counts", "UMI per nucleus", True),
                               ("n_genes", "genes per nucleus", True),
                               ("n_genes_by_counts", "genes per nucleus", True)):
            if col not in self.P or any(o[1] == name for o in out):
                continue
            f = np.asarray(self.P.loc[self.P["flag"], col], dtype=float)
            k = np.asarray(self.P.loc[~self.P["flag"], col], dtype=float)
            out.append((col, name, f[~np.isnan(f)], k[~np.isnan(k)], log))
        return out


def is_counts(block):
    """Whether a block looks like integer counts rather than something already normalised.

    A matrix normalised twice produces a dotplot that looks exactly like a correct one, so the
    check is made explicitly instead of assumed either way. Reported by the caller, never silent.
    """
    if block.size == 0:
        return False
    mx = float(block.max())
    return mx > 20 and float(np.abs(block - np.round(block)).max()) < 1e-6


def lognorm(block, row_totals):
    """log1p(counts per 10,000), row-wise.

    `row_totals` MUST be the sum over EVERY gene of each row, not over the columns in `block`.
    Normalising a gene-subset by its own row sums makes each nucleus's selected genes sum to
    10,000 - so a dotplot column reports a gene's share of the panel rather than its expression,
    and every panel is renormalised to look equally strong. The bug is invisible on the page:
    both versions produce a plausible grid.
    """
    if block.size == 0 or not is_counts(block):
        return block
    tot = np.asarray(row_totals, dtype=float).reshape(-1, 1)
    tot = np.where(tot <= 0, 1.0, tot)
    return np.log1p(block / tot * 1e4)


def _row_totals(X, rows):
    """Sum over all genes for the given rows, without densifying the whole matrix."""
    import scipy.sparse as sp
    sub = X[rows]
    return (np.asarray(sub.sum(axis=1)).ravel() if sp.issparse(sub)
            else np.asarray(sub).sum(axis=1))


def _r(v, nd):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return round(float(v), nd) if nd else int(round(float(v)))
    except Exception:                                                     # noqa: BLE001
        return None
