"""The sentinels: label values meaning *the annotator declined to call this nucleus*.

Not cell types. Never a population in a composition, never a denominator, never dropped. They
were defined in five modules; this is the one definition, and every other module imports it, so
a consumer asking `scanno describe` gets the same words the code writes.

    EXCLUDED     withheld from annotation upstream (a QC flag the object declares)
    UNRESOLVED   walked, and could not be pushed to a leaf without inventing evidence

`ALIASES` names what the same idea is called by the harness this tool is orchestrated by, so an
adapter maps by declaration rather than by guessing from the word.
"""
EXCLUDED = "EXCLUDED"
UNRESOLVED = "UNRESOLVED"
SENTINELS = (EXCLUDED, UNRESOLVED)
ALIASES = {"unassigned": UNRESOLVED, "excluded": EXCLUDED}


def is_sentinel(label) -> bool:
    return str(label).split("/", 1)[0] in SENTINELS or str(label) in SENTINELS
