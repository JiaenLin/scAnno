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
