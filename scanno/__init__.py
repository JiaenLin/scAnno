"""scAnno - hierarchical cell-type annotation that truncates rather than guesses.

    from scanno import build_store, cluster_profile, standardise, classify

The classifier is a pure function of (query, store-or-corpus, declared tree). It fits
nothing at runtime, so a result is reproducible from the store digest and the tree.

scAnno annotates. It does not perform quality control: it computes no QC metric, applies no
threshold, and cannot decide that a nucleus is technical. Where upstream QC has already made that
decision, `--exclude-flag` withholds EXACTLY the nuclei it names - see `scanno/exclude.py`.
"""
from .classify import (GAP_CORPUS, GAP_PROFILE, classify, gate_auc, missing_nodes,
                       node_profiles, profile_weights)
from .corpus import (GeneSpaceMismatch, TIER_W, check_gene_space, load_assertions,
                     node_weights)
from .cluster import cluster, parse_resolutions, res_tag
from .compare import compare as compare_routes
from .emit import annotate_obs, format_readiness, lab_readiness, per_cell
from .exclude import (EXCLUDED, ExclusionMismatch, as_mask, exclusion_record_cells,
                      flag_digest, unprofilable)
from .neighbours import cluster_neighbourhood, label_flow
from .query import DETECT_FLOOR, cluster_profile, standardise
from .resolution import (derived_tolerance, format_report, pick_resolution,
                         sweep_stability)
from .store import NORM, ProfileStore, build_store, safe_scale

#: Kept in step with the VERSION file, which is the one a reader checks. They disagreed between
#: 0.1.0 and 0.2.0 - the package reported a version it had not been for two releases - which is
#: the same class of defect as a run citing a commit hash that does not exist.
__version__ = "0.9.0"

__all__ = [
    "build_store", "ProfileStore", "safe_scale", "NORM",
    "cluster_profile", "standardise", "DETECT_FLOOR",
    "load_assertions", "node_weights", "TIER_W", "check_gene_space",
    "GeneSpaceMismatch",
    "classify", "node_profiles", "profile_weights", "missing_nodes", "gate_auc",
    "GAP_PROFILE", "GAP_CORPUS",
    "sweep_stability", "pick_resolution", "derived_tolerance", "format_report",
    "cluster_neighbourhood", "label_flow",
    # Exclusion. `cluster_flags`, `exclusion_record`, `FLAG_SHARE`, `CELL`, `CLUSTER` and
    # `MODES` were REMOVED in 0.3.0 and are deliberately not re-exported under any name: they
    # implemented a cluster-share exclusion, which is a QC decision and not scAnno's to make.
    # An importer of those names now fails loudly, which is the intended behaviour.
    "as_mask", "EXCLUDED", "ExclusionMismatch", "unprofilable", "exclusion_record_cells",
    "flag_digest",
    # Emitting the annotation per CELL. classify() returns one row per CLUSTER, and until
    # 0.3.1 the join back onto the object was left to every caller - so the object scAnno had
    # just annotated still carried no annotation, and nothing downstream could open it.
    "per_cell", "annotate_obs", "lab_readiness", "format_readiness",
    # Step 1, and the two-route check. Neither selects anything: `cluster` keeps every
    # resolution it computes and `compare_routes` changes no call.
    "cluster", "parse_resolutions", "res_tag", "compare_routes",
    "__version__",
]
