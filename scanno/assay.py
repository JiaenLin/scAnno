"""The assay: whole cells or nuclei. One table, read by every rule and every word that depends on it.

Single cells are not single nuclei (single-cell-harness rule 2). A nuclear transcriptome is not a
whole-cell one with fewer reads: it is missing most of the cytoplasm's mRNA and carries far more
unspliced and nuclear-retained RNA. So whatever scAnno learned from one assay is not lent to the
other, and nothing written about one assay is printed about the other.

WHAT DEPENDS ON THE ASSAY HERE, AND WHAT DOES NOT, is `RULES` below. It is printed by
`scanno describe` and set out in docs/USER_GUIDE.md. Three things depend on it:
  - the gene background. A store records the assay it was measured on, and `annotate` refuses a
    store of the other assay, or one that records none;
  - the words. Every print and every document a command writes names cells for `scrna` and
    nuclei for `snrna`, or names no unit. background, calibrate, annotate and report use the
    declared assay; rescue, scope, compare, joint-review, embed and readme use the assay their
    annotated objects recorded. cluster, lab and resolution, objects that record none, and all
    help text (printed before any input is read) use `EITHER`: "cells or nuclei". Strings
    RECORDED in a data product (rescue.json, compare's and the joint route's records) keep their
    words, because reproductions compare them; KNOWN_ISSUES.md lists them;
  - what the agent is told it is annotating.
Everything else is one rule for both assays. Each such rule says where its numbers came from,
because "the same for both" is a claim about calibration as well as about code.

SPELLINGS. `scrna` and `snrna` are the family's names: scQC's, scProfile's, and the harness's
rules. `sc` and `sn` are this tool's original spellings. Every store built before this table, and
every job written for one, carries them, so they are accepted and mean exactly the same two
assays. A value is RECORDED AS DECLARED, so a store's bytes and its digest do not change. It is
COMPARED CANONICALLY: `sn` and `snrna` are one assay.

NO DEFAULT. An absent or unrecognised assay raises. It is never read as whichever assay the first
cohort happened to be. That is the defect this module exists to end in both directions: scQC read
an absent assay as nuclei, and this tool read one as cells.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The two assays and the words each one is named by.
ASSAYS = {
    "scrna": {"name": "single-cell", "unit": "cell", "units": "cells"},
    "snrna": {"name": "single-nucleus", "unit": "nucleus", "units": "nuclei"},
}

#: Every accepted spelling, mapped to its assay. The family's names first.
SPELLINGS = {"scrna": "scrna", "snrna": "snrna", "sc": "scrna", "sn": "snrna"}

#: What `--assay` accepts, everywhere it is taken.
CHOICES = tuple(SPELLINGS)

HELP = ("scrna (whole cells) or snrna (nuclei); sc and sn are the same two assays under this "
        "tool's original spelling. A fact of the experiment, so there is no default")


def canonical(value) -> str:
    """`scrna` or `snrna` for any accepted spelling. Absent or unknown raises ValueError."""
    v = "" if value is None else str(value).strip().lower()
    if v not in SPELLINGS:
        raise ValueError(
            f"assay {value!r} is neither scrna (whole cells) nor snrna (nuclei)"
            + ("; an absent assay is not read as either" if not v else "")
            + f". Accepted: {', '.join(CHOICES)}")
    return SPELLINGS[v]


def recorded(context) -> str | None:
    """The canonical assay a record states - a store's context, an object's `scanno_context` -
    or None when it states none. An unrecognised value raises, like any other."""
    v = (context or {}).get("assay")
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    return canonical(v)


@dataclass(frozen=True)
class Words:
    """The words one assay is written in. `fill` puts them into a sentence the tool wrote.

    `fill` is for the tool's OWN prose and never for data: a cell type can be called anything,
    and a data value passed through it could be rewritten.
    """
    assay: str
    name: str
    unit: str
    units: str

    @property
    def Unit(self) -> str:              # noqa: N802 - sentence case, read as the word it prints
        return self.unit[:1].upper() + self.unit[1:]

    @property
    def Units(self) -> str:             # noqa: N802
        return self.units[:1].upper() + self.units[1:]

    def fill(self, text: str) -> str:
        """Replace {unit} {units} {Unit} {Units} {assay_name} - and nothing else, so a brace
        that is not one of these five passes through untouched."""
        for k, v in (("{unit}", self.unit), ("{units}", self.units), ("{Unit}", self.Unit),
                     ("{Units}", self.Units), ("{assay_name}", self.name)):
            text = text.replace(k, v)
        return text


def words(value) -> Words:
    a = canonical(value)
    return Words(a, **ASSAYS[a])


#: The words for an observation whose assay is NOT KNOWN: help text, which is printed before any
#: input is read, and a command whose inputs record no assay. They name both units, so neither
#: assay becomes the default by being the word that was printed. `assay` is not an assay's name,
#: and `canonical` refuses it.
EITHER = Words("unrecorded", "single-cell or single-nucleus", "cell or nucleus", "cells or nuclei")


def words_or_either(value) -> Words:
    """The assay's words when one is known - declared, or recorded by an object - and EITHER's
    when none is. An UNRECOGNISED value still raises: an assay nobody stated is not the same as
    one stated wrongly."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return EITHER
    return words(value)


#: Every rule that decides a label, a reference or a word, against the assay. `depends` says
#: whether scrna and snrna are treated differently. Where they are not, `basis` says where the
#: rule's numbers came from - and says so only as far as the record goes.
#: Docs/USER_GUIDE.md names every `id`, and tests/test_assay_rules.py holds the two to each other.
RULES = [
    {"id": "background", "rule": "the gene background: per-gene mean and spread across cell types "
                                 "(a store from `background` or `calibrate`)",
     "scrna": "built from whole cells", "snrna": "built from nuclei", "depends": True,
     "basis": "measured on each cohort's own data. A store records the assay it was built from, "
              "and `annotate --store` refuses a store of the other assay or one that records none. "
              "`--background-from-clusters` builds it from the object being annotated, so it is "
              "that object's assay by construction"},
    {"id": "words", "rule": "the unit every print and every document names: the two reports, "
                            "annotate's summaries, and the prints and documents of every other "
                            "command",
     "scrna": "cell, cells", "snrna": "nucleus, nuclei", "depends": True,
     "basis": "the declared assay (background, calibrate, annotate, report), or the assay the "
              "annotated objects recorded (rescue, scope, compare, joint-review, embed, readme). "
              "cluster, lab, resolution, objects that record none, and help text say 'cells or "
              "nuclei'. Strings recorded in data products keep theirs, because reproductions "
              "compare them, and are listed in KNOWN_ISSUES.md"},
    {"id": "agent_prompt", "rule": "what `scanno agent` tells the model it is annotating",
     "scrna": "single-cell RNA-seq", "snrna": "single-nucleus RNA-seq", "depends": True,
     "basis": "the declared assay; a prompt with none is refused"},
    {"id": "corpus_weights", "rule": "marker weights from the CellMarker-schema corpus "
                                     "(evidence tier x log publications)",
     "scrna": "the same", "snrna": "the same", "depends": False,
     "basis": "the corpus records no assay. Its assertions come mostly from whole-cell and tissue "
              "studies, so on nuclei a marker whose mRNA sits in the cytoplasm can be under-"
              "detected. Nothing here corrects for that (KNOWN_ISSUES.md)"},
    {"id": "gap_min", "rule": "the descent bar: 0.30 on the corpus path, 0.15 on profiles",
     "scrna": "the same", "snrna": "the same", "depends": False,
     "basis": "declared, and chosen on human blood (PBMC), which is whole cells. It has not been "
              "calibrated on nuclei (KNOWN_ISSUES.md)"},
    {"id": "detect_floor", "rule": "a gene is scored for a cluster only where at least 1% of the "
                                   "cluster detects it",
     "scrna": "the same", "snrna": "the same", "depends": False,
     "basis": "declared in the first release. No calibration on either assay is recorded"},
    {"id": "ood_floor", "rule": "the background must cover 30% of the genes the object expresses",
     "scrna": "the same", "snrna": "the same", "depends": False,
     "basis": "a check that two gene lists overlap. It measures neither transcriptome"},
    {"id": "walk", "rule": "the rooted walk: sibling contrast, truncation, the scope vote "
                           "(SEAL, KEEP, FORCE) and --resolve",
     "scrna": "the same", "snrna": "the same", "depends": False,
     "basis": "it compares siblings' scores with each other. Its only number is gap_min, above"},
    {"id": "scope_vote", "rule": "a node is sealed unless every sample that reached it descended, "
                                 "and is voted on only when two or more samples reached it",
     "scrna": "the same", "snrna": "the same", "depends": False,
     "basis": "declared principles, unanimity and a quorum. They were not fitted to data"},
    {"id": "normalisation", "rule": "counts to 10,000 per observation, then log1p; `cluster`'s "
                                    "2,000 variable genes, 50 components, 15 neighbours, Leiden",
     "scrna": "the same", "snrna": "the same", "depends": False,
     "basis": "scanpy's conventional defaults. They were not calibrated on either assay"},
    {"id": "store_sizes", "rule": "a cell type enters a store at 10 observations and is graded "
                                  "at 50",
     "scrna": "the same", "snrna": "the same", "depends": False,
     "basis": "set on human blood (PBMC), which is whole cells: at 50 the entry floor excluded "
              "the two rarest PBMC types (docs/CLASSIFIER.md)"},
    {"id": "reporting_floors", "rule": "numbers that decide what a report shows and never a label: "
                                       "30 withheld per sample before a per-sample comparison, a "
                                       "3x ratio between arms, 80% of a cluster from one sample, "
                                       "a 0.1-point floor on the sweep's tolerance",
     "scrna": "the same", "snrna": "the same", "depends": False,
     "basis": "declared. Each is reported and none changes a label"},
    {"id": "exclusion", "rule": "withholding exactly the observations upstream QC flagged",
     "scrna": "the same", "snrna": "the same", "depends": False,
     "basis": "scAnno reads the flag and decides nothing. The QC that set it is upstream's, and "
              "upstream's rules are per assay"},
    {"id": "species_check", "rule": "the declared species against the gene identifiers",
     "scrna": "the same", "snrna": "the same", "depends": False,
     "basis": "identifiers name a species, not an assay"},
]
