# Directions

Designs that have been thought through and **not built**. Nothing here is decided, nothing here
is implemented, and no number in this file was produced by a run.

A direction earns a place here by surviving a discussion, not by sounding good. Each one records
what it would answer, the objection that nearly killed it, and what has to be settled before any
code is written — because the expensive mistake is not building the wrong thing, it is building
the right thing before its open questions have been answered.

---

## Direction 1 — rare cell type ABSENCE quality control

**Status:** noted 2026-09-02, discussed, **not designed and not built**. Named by the PI.

### The question it answers

Multi-sample studies routinely deliver a cell type in some samples and zero in others. That zero
has two causes and they are indistinguishable in the output:

- the population is genuinely absent from that animal, or
- it was present and the per-sample clustering could not separate it, so its cells were absorbed
  into a larger neighbour and took that neighbour's label.

**A composition claim cannot be made across samples until those two are told apart**, and nothing
in the pipeline currently tells them apart.

### What it is, and what it deliberately is not

For a label `L` that some samples carry and others do not: train a detector on the samples that
have it (`S_have`), apply it to the samples that do not (`S_lack`), and ask whether `L`'s
signature is present there.

**It produces a FINDING, not a correction.** It touches no delivered label and proposes no
relabelling. This is the whole reason it is worth doing: the joint route conflates *"is this zero
real?"* with *"should these cells be relabelled?"*, and the first is the question the biology
needs, is far weaker, and is answerable without editing anything.

It is a **confirmer, not a discoverer.** It can only look for labels that already exist somewhere
in the annotation. A cell type no sample resolved is invisible to it, by construction.

### Why it is worth building when the earlier ideas were not

Every mechanism proposed in this area so far has run into `PRINCIPLES.md` §3 — *no statistic
gates an output until its agreement with correctness has been shown.* The resolution sweep could
not show it. The consensus vote certainly could not, and was withdrawn for exactly that reason.

**This design can show it, and the demonstration is free.** `L` exists in more than one sample,
so hold one of them out: train on the rest, predict the held-out sample, and measure recall for a
label that is *known* to be there. That is a per-label, per-dataset answer to "does cross-sample
transfer work on this data at all", computed on the data in hand rather than assumed from
elsewhere.

It is also the answer to **"what happens on a project with a large batch effect?"** — the failure
mode of every clustering-based route. A batch effect does not corrupt this one silently. It shows
up as held-out recall collapsing, which is a result you can act on rather than a corruption you
cannot see. **The mechanism reports its own failure instead of producing confident nonsense.**

### The boundary it draws by itself

A label present in exactly **one** sample cannot be held out, so its transfer cannot be validated
at all. That is not a threshold anyone chose; it falls out of the design, and the honest output
for such a label is a NAMED refusal rather than a number.

Measured on the cohort this was discussed against, and the awkwardness is the point: the label the
whole discussion circled — the one the joint route absorbs — is present in **one** sample and
would be declined. Neural (4 samples), Adipocyte (5), Smooth muscle (7) and Lymphoid (8) are all
testable. A tool that says *"cannot be validated for this label, and here is why"* is worth more
than one that returns a number for everything.

### The objection that has no clean answer

**There is no known-negative sample.** Recall on held-out positives is measurable; the
false-positive rate is not, because no sample is certified to lack `L`. Without it, "the detector
fired on N cells in sample S" has no null to be read against.

Two partial answers, neither sufficient, both to be printed rather than relied on:

- **permutation null** — retrain on shuffled labels and see what the detector does by chance.
  A floor, not a calibrated rate.
- **cross-label null** — run detectors for similarly abundant labels that the annotation *did*
  resolve everywhere, and use their behaviour in `S_lack` as a reference point.

This is a permanent limitation of the design, not a gap to be closed with a threshold.

### Per-cell hard calls are the wrong output

A nucleus with 1–3k detected genes, for a type whose markers are lowly expressed, is thin
evidence — and a linear classifier treats an undetected gene as absence, where `store.safe_scale`
deliberately does not. scAnno scores clusters for that reason.

So the output should be a **presence test on the score distribution**, not a per-cell assignment:
score every cell in `S_lack` and ask whether the upper tail departs from the null. That concludes
*"the signature is present in this sample"* without committing to which nuclei carry it — better
behaved on sparse data, and a closer match to the claim being made. Cells can be produced if
wanted; the finding must not depend on them.

### What it may claim

Not *"confirms the presence of L"*. What the evidence supports is: **the absence of `L` in sample
S is, or is not, consistent with `L` being genuinely absent, given how `L` looks in the samples
that have it.** It establishes no identity — there is no truth set for this tissue and scAnno has
never been validated on snRNA.

### Open before anything is designed

1. **Train `L` against what?** Every other cell, or only its confusable siblings inside the
   compartment? The second is far better conditioned and matches the tree, but it presumes the
   compartment call is right.
2. **How few `S_have` samples is too few?** Four samples give four folds; two give a coin flip.
   The floor should be decided before the numbers are seen.
3. **What does a middling recall mean?** 0.6 is neither a green light nor a refusal. Decide in
   advance, or the number will decide for us — which is how the withdrawn consensus vote came to
   have its rule chosen after the fact.

### Where it would NOT go

Not inside `compare`, which opens no matrix and must keep that property. Not inside `classify`,
which is the pure function everything rests on. A fitted step belongs where `calibrate` already
puts one: its own command, its own declared and digested artifact, consumed by name.

---

## Direction 2 — resolution-induced absence

**Status: BUILT, 2026-09-02, as `scanno rescue` (0.12.0).** The design below is the PI's; what is
recorded here is what the discussion got wrong on the way, because all of it was mine.

**The design, as stated:** for a rare cell type a unit lacks and another unit carries, cluster
*that unit* more finely step by step, annotate each step in the ordinary unbiased way, and on the
first step where a cluster comes back as the target, **rename that cluster's cells and nothing
else**. The finer clustering LOCATES; it is never adopted.

**Three errors before it was implemented, in order:**

1. **A grid is not a search.** The first attempt ran a fixed eight-rung grid on every sample and
   reported "did not appear" for populations the grid could never have separated.
2. **When the instrument does not reach, the finding is that it does not reach.** The second
   attempt answered the first by extending the ladder to resolution 32 — 442 to 628 clusters of
   14 to 24 nuclei per sample. That is dissolving a partition, not refining one, and choosing the
   range by what makes the answer appear is choosing the analysis on its outcome.
3. **Adopting the finer annotation is a different claim.** Both attempts measured the whole
   sample's churn at the located rung — `UNRESOLVED` growth, compartment names taking
   macrophages, fibroblasts collapsing — and reported it as the cost of the rescue. None of it
   is a cost, because none of it is adopted. **Only the located cluster moves.** The impact is
   the renamed cells and the labels they came from, and nothing else.

Everything below was written before the design was understood and is kept for the record.

### The question it answers

The same question as Direction 1 — *is this zero real, or a detection failure?* — by a completely
different route, using no fitted model at all.

Where a sample lacks a label `L` that other samples carry, **raise that sample's clustering
resolution gradually and measure what it costs to make `L` appear.** If `L` appears cheaply, the
imbalance was induced by the granularity that happened to be chosen. If it appears only at great
cost, or never, the absence survives.

### Why the cost, and not the appearance, is the measurement

*"Search the sweep until `L` appears"* is a search that cannot fail: at fine enough resolution
every population fragments into something, and a small coincidentally-pure fragment can score as
a rare type. A search that always succeeds carries no information.

Measuring the **cost** converts that into a graded, falsifiable quantity, and it reverses the
burden of proof in the right direction: an absence must now SURVIVE a perturbation to be
believed. That is the correct default here, because per-sample clustering is already known to
under-resolve rare types — a zero is a suspect, not a witness.

The nearest established analogue is the fragility index: a trial result that flips when one
patient's outcome changes is fragile whatever its p-value. An absence that dissolves after one
resolution step was never an absence.

### The cost axis is Δ CLUSTERS, not Δ resolution

Leiden resolution is a request, not a result, and it is nonlinear — one cohort went 14 → 18
clusters over 0.25→0.5 and 32 → 36 over 1.75→2.0. Worse, "a small resolution difference" is an
artefact of the sweep's own step: a finer sweep would make every label appear at a small one.

**How many more clusters had to be admitted before `L` appeared** is step-independent, directly
interpretable, and measured rather than requested — the same correction `cluster_weights` needed.

### It supplies its own null, which Direction 1 cannot

Direction 1's unfixable gap is that no sample is certified to lack `L`, so no false-positive rate
is measurable. **This design has an internal reference.**

Ask the CARRYING samples the same question: at what cost does `L` first appear in a sample that
demonstrably has it? That yields a distribution of appearance costs for a label known to be
present. A lacking sample's cost is then read against it:

- inside that distribution → **resolution-induced imbalance**
- far outside it, or never → the absence survives

No threshold is chosen. The label's own behaviour where it is known to exist sets the scale.

### Run it PER SAMPLE, and the consequence is larger than it looks

Each sample is clustered and annotated across its own sweep, alone. **No clustering ever spans
samples, so a between-sample batch effect cannot corrupt it by construction** — which is the
failure mode that motivated this whole line of thought and that every joint-clustering route is
exposed to. It needs no integration, no fitted model, and stays inside *nothing is fitted at
runtime*.

### Report the PATTERN, not the first crossing

Leiden partitions across a sweep are not nested, so a label can appear at 1.0, vanish at 1.25 and
return at 1.75. The shape separates three findings a first-crossing number would merge:

| pattern | reading |
|---|---|
| appears at *r* and at every finer resolution | a detection floor — the strong case |
| appears at exactly one resolution | a fragment; noise wearing the label |
| flickers in and out | the evidence is about the algorithm, not the cells |

### THE CHURN QC — a gate, not a footnote

**Monitoring what the triggered resolution change does to every OTHER label is required, not
optional** (PI, 2026-09-02). Three distinct failures, and the third is disqualifying:

1. **A partition is indivisible.** You cannot take `L` from a finer resolution and leave the rest
   coarse. Claiming 40 `L` cells at 1.75 also claims that sample's Macrophage, Fibroblast and
   Pericyte counts at 1.75.
2. **The collateral can dwarf the finding** — 40 cells gained while 3,000 move between other
   labels makes the discovery a rounding error inside a re-annotation.
3. **The search can MANUFACTURE a new absence.** Measured on a real sweep: one resolution lost
   `Adipocyte`, another lost `Smooth muscle`. Chasing `L`'s absence in sample S can create `M`'s
   absence in sample S — using a rare-cell-type QC to fabricate the exact artefact it exists to
   detect.

**Failure 3 is rule one.** A resolution change that deletes a label is a removal, and removals
require the list of what was removed BEFORE the fact, not a count afterwards. The churn QC is
therefore the gate this direction has to pass to be run at all.

**Churn has structure and the tree already knows it.** `compare.level()` truncates paths to a
depth, splitting churn into two very different quantities for free:

- **within-compartment** (`Macrophage ↔ Dendritic` inside `Immune/Myeloid`) — refinement, and
  what you want to see when hunting a myeloid subtype
- **cross-compartment** (`Cardiomyocyte ↔ Stromal`) — not refinement but a different partition,
  and `L`'s appearance inside it is one outcome of a reshuffle rather than a resolved merge

**Read the appearance against that sample's OWN churn curve.** Churn is not uniform along a
sweep — some ranges sit on a granularity boundary, others are plateaus — and baseline stability
differs per sample. A label appearing exactly where that sample's churn spikes is suspect; one
appearing on a plateau, with the rest of the annotation barely moving, is the clean case. This
turns *"is the churn small?"*, which needs a threshold, into *"is the churn unusual HERE?"*,
which does not.

### Two modes that must never be conflated

**Evidence mode** — the finding is *"S's zero for `L` is resolution-induced"*. The delivered
annotation never changes and the finer partition is never shipped; churn is then a question about
whether the evidence is contaminated, and the stakes are low.

**Adoption mode** — cells actually move. Rule one applies in full and the labels-lost check
becomes a hard refusal rather than a caveat.

**Default to evidence mode, and probably never leave it.** The question the biology asks is *is
this zero real*, and answering it requires relabelling nothing.

### What must travel with each (sample, label) record

- appearance cost in clusters admitted, against the carrying samples' distribution
- the appearance pattern across the sweep
- **the gap at the appearing resolution against that label's cohort-median gap** — fine clusters
  are small and their profiles noisier, so gaps shrink; a fine-resolution appearance WITH a
  healthy gap is a free internal control
- where the new `L` cells came from, by previous label — emerging from one plausible neighbour is
  a coherent story, scattering out of six unrelated labels is not
- **the range searched, named, when nothing appears.** "Not found up to 36 clusters" is a bounded
  claim; "not found" is not
- the churn record above, stepwise as well as endpoint

### Most of it already exists

`resolution.sweep_stability` already returns per resolution: `n_labels` (a drop is label loss),
`smallest` (the rarest population's size — the label-loss early warning), `neighbour` (stability
against BOTH adjacent resolutions, i.e. churn), `truncated` and `unresolved`. `sweep_agreement`
gives per-cell churn's complement. `compare.level()` gives the tree split. `annotate
--cluster-key` is already repeatable.

**This is assembly of things already in the tool, not new statistics** — and it therefore
inherits their conventions instead of inventing parallel ones.

### Where the QC itself can still mislead

- **a lost label of 5 cells and one of 500 are not the same event.** Presence/absence is the hard
  gate; the count must be printed beside it or the gate is absurd on trivial populations.
- **churn between two resolutions is not churn along the path between them.** Comparing 1.0 to
  1.75 directly misses intermediate thrashing; stepwise churn matters as much as the endpoint.
- **it cannot tell you the appearance is CORRECT** — only that it was not bought by wrecking
  everything else. A fine, pure, mislabelled fragment passes every churn check cleanly.

### Relation to Direction 1

Direction 2 is the cheaper and safer instrument and should run first: no fitting, no cross-sample
clustering, existing machinery, its own internal null. Direction 1 is a genuinely independent
line of evidence — expression transfer rather than granularity perturbation — and agreement
between two mechanisms sharing no assumptions would be worth a great deal.

They also help each other: **if the per-sample sweep resolves a label in four samples where the
delivered resolution found one, Direction 1's hardest boundary dissolves** — it gains folds and
becomes validatable for exactly the labels it would otherwise have to decline.

---

## Direction 3 — resolution-induced PRESENCE, and when absorption is right

**Status:** noted 2026-09-02, **explicitly lower priority — the PI has said this can be left.**
Recorded so it is not rediscovered from scratch, not because it is queued.

### The mirror of Direction 2

Direction 2 asks whether a **zero** is resolution-induced. Direction 3 asks whether a small
**positive** is: a rare label that exists in the delivered annotation only because of the
granularity that happened to be chosen, and that a coarser or different partition absorbs.

The two are the same question with the sign flipped, and they should share machinery and
vocabulary if they are ever both built.

### The case it came from

The joint route ABSORBS a rare myeloid label — the per-sample route delivers 47 nuclei, the joint
partition at the reconciled resolution has no cluster of its own for them, and its cells take
their clusters' calls. **The PI's judgement is that this absorption is correct**, and it is: at
that resolution the joint route genuinely says those cells group with macrophages. The label is a
cluster's modal call only at the two finest resolutions of eight.

So absorption is not a defect to be fixed. What is missing is a principled account of **when it
is right**, symmetric to Direction 2's account of when an absence is real.

### The open question

Absorption currently fires on a structural condition — route B delivers the label nowhere — with
no evidence about whether the population is real. Direction 2's apparatus would supply exactly
that evidence, applied in the other direction: at what cost does the label appear, does it appear
in several samples, does it survive its own sweep. A label that appears only at one extreme of
one partition is a fragment and absorbing it is right; one that appears robustly across a range
and is absorbed only at the reconciled resolution is a different matter.

### Why it is lower priority

The absorbing direction is conservative — it moves cells back onto the majority call, which is
the safe error. Direction 2's failures are the expensive ones: an absence wrongly believed
becomes a composition claim across the design.
