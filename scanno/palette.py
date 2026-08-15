"""Colours for a taxonomy of ANY depth, assigned automatically or pinned by the caller.

FOUR RULES, AND EACH ONE IS A THING THAT WENT WRONG WITHOUT IT

1. A DESCENDANT KEEPS ITS ANCESTOR'S HUE. A level-2 figure is the level-1 one SUBDIVIDED, not a
   different partition of the same cells, and giving subtypes fresh hues makes a reader compare
   two pictures that are the same picture. So the hue comes from the top-level ancestor and only
   the lightness moves - which also means a reader who has learnt the level-1 colours can read
   any deeper figure without relearning them.

2. THE SENTINELS ARE GREY, AND THEY ARE TWO DIFFERENT GREYS. `UNRESOLVED` means the evidence ran
   out; `EXCLUDED` means we chose not to look. Conflating them hides the more consequential one,
   and giving either a hue makes it read as one more cell type in the legend. `EXCLUDED` is the
   darker of the two because it is the one a reader must not mistake for an absence of cells.

3. EVERY LOOKUP HAS A DEFAULT. A palette lookup for a categorical that can gain categories
   returned `None` for a label that did not exist when the palette was written, and matplotlib
   raised on `c=[None]` after the annotation had already been rewritten. A hope is not a default.

4. NO TWO LABELS THAT CAN APPEAR TOGETHER MAY SHARE A COLOUR. A truncating annotator emits labels
   at MIXED depths in one figure - a cell whose evidence ran out at level 1 keeps its level-1
   label beside another cell called at level 4 - so distinctness has to hold across the whole
   subtree, not merely among siblings.

HOW LIGHTNESS IS ASSIGNED, AND THE TWO WAYS IT WAS DONE WRONG FIRST

Every node in a root's subtree is enumerated in depth-first order and given an evenly spaced
lightness slot in the legible band. Depth-first means a parent sits beside its own children, so
the family structure is still visible, and an even spread over distinct slots means rule 4 holds
by construction at any depth.

  - The first attempt COMPOUNDED a fixed step per level. A depth-3 node under a dark root landed
    at #080b10 - black. It spent lightness it did not have, because nothing checked how much was
    left to spend.
  - The second normalised the accumulated offset into the band, which fixed the black but not
    rule 4: with the per-level step decaying by 0.62, the steps below a level sum to more than
    the level itself, so paths interleave and a depth-6 tree collided on 5 of 32 leaves. Sibling
    -relative arithmetic cannot promise global distinctness however carefully it is tuned; rank
    over the whole subtree can, and is simpler.

DEPTH IS NOT ASSUMED

Nothing here knows how deep the tree is. A taxonomy of depth 2 and one of depth 6 both work.
"""
from __future__ import annotations

import colorsys
import json

#: Distinct hues for top-level labels. Deliberately not a continuous colormap: adjacent cell
#: types are not adjacent quantities. Roots are coloured in the order given, so a caller that
#: passes them most-abundant-first gets the strongest hues on the populations that carry the plot.
#: No two of the first eight share a hue. `#8FCFC0` was `#2FA88B` lightened, so two
#: unrelated roots read as a lineage pair - the one relationship shading reserves.
#: No two adjacent entries are near-duplicates. `#C0392B` sat at index 6 and is a darker
#: `#E04B3A`; on a seven-root taxonomy the first and seventh populations came out as two reds
#: side by side in the stacked bars, which is the figure a reader spends the longest on.
BASE_HUES = ["#E04B3A", "#3D5A8A", "#2FA88B", "#F2A07B", "#8B7FB8", "#6BA83F",
             "#946B2D", "#D94F9A", "#55A868", "#DD8452", "#B07AA1", "#64B5CD",
             "#7F7F7F", "#76B7B2", "#F28E2B", "#59A14F"]

UNRESOLVED = "UNRESOLVED"
EXCLUDED = "EXCLUDED"

#: Two greys, not one. See rule 2.
SENTINELS = {UNRESOLVED: "#B8B8B8", EXCLUDED: "#5A5A5A"}

#: Anything the palette has never seen. Never `None`. See rule 3.
FALLBACK = "#D9D9D9"

#: The legible band. Below the floor a colour is indistinguishable from the axis line; above the
#: ceiling it is indistinguishable from the page.
LIGHTNESS_MIN, LIGHTNESS_MAX = 0.30, 0.82

#: Saturation falls slightly with depth, so a deep subtype reads as a shade of its lineage rather
#: than as a competing primary colour.
SAT_DECAY = 0.94


def _to_hls(hex_colour):
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)


def _to_hex(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def shade(base_hex, frac, depth):
    """`base_hex` recoloured to band position `frac` in [0, 1], at `depth`.

    Hue is the ancestor's untouched; lightness is the band position; saturation decays with depth.
    """
    h, _, s = _to_hls(base_hex)
    l = LIGHTNESS_MIN + max(0.0, min(1.0, frac)) * (LIGHTNESS_MAX - LIGHTNESS_MIN)
    return _to_hex(h, l, s * (SAT_DECAY ** max(0, depth - 1)))


def _split(path):
    return [p for p in str(path).split("/") if p]


def depth_of(paths):
    """The deepest path in the taxonomy, so nothing downstream has to assume 2 or 3."""
    d = 0
    for p in paths:
        if str(p) in SENTINELS:
            continue
        d = max(d, len(_split(p)))
    return d


class Palette:
    """Colours for every label at every depth of a taxonomy.

    `paths` is any iterable of full label paths (`"Immune/Myeloid/Macrophage"`); intermediate
    nodes need not be listed separately, they are implied. Sentinels may appear at any depth and
    are passed through. `pinned` maps a label - a full path, or a bare name - to a colour and
    always wins; pinning an interior node recolours its whole subtree, which is what pinning a
    lineage means.
    """

    def __init__(self, paths, pinned=None):
        self.pinned = {str(k): str(v) for k, v in (pinned or {}).items()}
        self._cache = {}
        self._children = {}
        roots = []
        for p in paths:
            p = str(p)
            if p in SENTINELS or _split(p) and _split(p)[-1] in SENTINELS:
                continue
            parts = _split(p)
            for i in range(len(parts)):
                sibs = self._children.setdefault("/".join(parts[:i]), [])
                kid = "/".join(parts[:i + 1])
                if kid not in sibs:
                    sibs.append(kid)
            if parts and parts[0] not in roots:
                roots.append(parts[0])
        self.roots = roots
        self.depth = depth_of(paths)

        # Rank every node of every root's subtree, depth-first. This is what makes rule 4 hold at
        # any depth: distinct rank -> distinct slot -> distinct colour, whatever the tree shape.
        self._rank, self._subtree_n = {}, {}
        for r in roots:
            order = []
            self._walk(r, order)
            self._subtree_n[r] = len(order)
            for i, node in enumerate(order):
                self._rank[node] = i

    def _walk(self, node, out):
        for kid in self._children.get(node, []):
            out.append(kid)
            self._walk(kid, out)

    def _pinned_ancestor(self, parts):
        """The deepest pinned ancestor of a path, if any - so a pin on a lineage propagates."""
        for i in range(len(parts), 0, -1):
            key = "/".join(parts[:i])
            if key in self.pinned:
                return key, i
            if parts[i - 1] in self.pinned:
                return parts[i - 1], i
        return None, 0

    def of(self, label, depth=None):
        """The colour of a label. `label` may be a full path or a bare name. Never `None`."""
        key = str(label)
        if key in self._cache:
            return self._cache[key]
        parts = _split(key)
        bare = parts[-1] if parts else key

        if key in self.pinned:
            return self.pinned[key]
        if bare in self.pinned:
            return self.pinned[bare]
        if key in SENTINELS:
            return SENTINELS[key]
        if bare in SENTINELS:
            return SENTINELS[bare]
        if not parts:
            return FALLBACK

        anc, anc_depth = self._pinned_ancestor(parts)
        if anc is not None and anc_depth < len(parts):
            # A pinned lineage: recolour the subtree from the pinned hue, keeping this node's
            # band position so the internal structure of that lineage survives the pin.
            base, root = self.pinned[anc], parts[0]
        else:
            root = parts[0]
            if root not in self.roots:
                self._cache[key] = FALLBACK
                return FALLBACK
            base = BASE_HUES[self.roots.index(root) % len(BASE_HUES)]

        if len(parts) == 1:
            self._cache[key] = base
            return base

        n = self._subtree_n.get(root, 0)
        if key not in self._rank or n <= 1:
            colour = shade(base, 0.5, len(parts))
        else:
            colour = shade(base, self._rank[key] / (n - 1), len(parts))
        self._cache[key] = colour
        return colour

    def map(self, labels, depth=None):
        """{label: colour} for a list, in the order given."""
        return {str(l): self.of(l) for l in labels}

    def as_dict(self):
        """Every colour assigned so far, for the report's provenance table."""
        return dict(sorted(self._cache.items())) | dict(SENTINELS) | dict(self.pinned)

    def collisions(self, labels):
        """Labels that would be drawn in the same colour. A report states this rather than
        letting a reader discover it by misreading a legend."""
        seen, clash = {}, []
        for l in labels:
            c = self.of(l)
            if c in seen and seen[c] != str(l):
                clash.append((seen[c], str(l), c))
            seen.setdefault(c, str(l))
        return clash

    @staticmethod
    def load(path):
        """A user's pinned mapping: {"label or path": "#RRGGBB"}."""
        if not path:
            return {}
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if not isinstance(d, dict):
            raise ValueError("a palette file must be a JSON object of label -> colour")
        bad = [k for k, v in d.items()
               if not (isinstance(v, str) and v.startswith("#") and len(v) in (4, 7))]
        if bad:
            raise ValueError(f"palette: not #RRGGBB colours: {', '.join(map(str, bad[:6]))}")
        return d
