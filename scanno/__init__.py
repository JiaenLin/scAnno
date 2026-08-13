"""scAnno 鈥?hierarchical cell-type annotation that truncates rather than guesses.

    from scanno import build_store, cluster_profile, standardise, classify

The classifier is a pure function of (query, store-or-corpus, declared tree). It fits
nothing at runtime, so a result is reproducible from the store digest and the tree.
"""
from .classify import (GAP_CORPUS, GAP_PROFILE, classify, gate_auc, missing_nodes,
                       node_profiles, profile_weights)
from .corpus import (GeneSpaceMismatch, TIER_W, check_gene_space, load_assertions,
                     node_weights)
from .neighbours import cluster_neighbourhood, label_flow
from .query import DETECT_FLOOR, cluster_profile, standardise
from .resolution import (derived_tolerance, format_report, pick_resolution,
                         sweep_stability)
from .store import NORM, ProfileStore, build_store, safe_scale

__version__ = "0.1.0"

__all__ = [
    "build_store", "ProfileStore", "safe_scale", "NORM",
    "cluster_profile", "standardise", "DETECT_FLOOR",
    "load_assertions", "node_weights", "TIER_W", "check_gene_space",
    "GeneSpaceMismatch",
    "classify", "node_profiles", "profile_weights", "missing_nodes", "gate_auc",
    "GAP_PROFILE", "GAP_CORPUS",
    "sweep_stability", "pick_resolution", "derived_tolerance", "format_report",
    "cluster_neighbourhood", "label_flow",
    "__version__",
]

