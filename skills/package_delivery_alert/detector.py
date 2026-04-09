"""
Package-delivery detection logic.

Scores a VLM response string by counting how many weighted keyword groups
match.  The caller compares the total score against a configurable threshold.

Scoring design
--------------
We use weighted groups rather than a flat keyword list so that a hit on a
highly-specific term (e.g. "fedex") counts more than a generic term (e.g.
"person").  A group score is counted at most once regardless of how many
keywords inside it match, which prevents "box box box" from inflating the
score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Keyword groups — (weight, [keywords])
# ---------------------------------------------------------------------------
_KEYWORD_GROUPS: list[tuple[int, list[str]]] = [
    # Carrier / courier brands — highly specific (weight 4)
    (4, ["fedex", "ups", "usps", "dhl", "amazon", "ontrac", "lasership",
         "purolator", "canada post", "royal mail"]),

    # Delivery-action verbs near context — specific (weight 3)
    (3, ["delivering", "delivered", "package delivery", "parcel delivery",
         "dropping off", "drop off", "leaving a package", "left a package",
         "placing a package", "placed a package"]),

    # Object: package / parcel (weight 2)
    (2, ["package", "parcel", "cardboard box", "cardboard boxes",
         "shipping box", "delivery box", "box on the porch", "boxes on",
         "package on", "parcel on"]),

    # Delivery person / uniform (weight 2)
    (2, ["delivery person", "delivery driver", "courier", "mail carrier",
         "postal worker", "in uniform", "delivery uniform"]),

    # Delivery vehicle (weight 2)
    (2, ["delivery truck", "delivery van", "mail truck", "ups truck",
         "fedex truck", "amazon van", "brown truck", "white van"]),

    # Location context — adds confidence when combined with other signals (weight 1)
    (1, ["front door", "front porch", "doorstep", "door step", "front step",
         "porch", "driveway", "walkway"]),

    # Generic person present (weight 1 — alone not enough)
    (1, ["person", "someone", "individual", "man", "woman", "he", "she",
         "they", "approaching the door", "walking to the door"]),
]


@dataclass
class DetectionResult:
    detected: bool
    score: int
    matched_groups: list[str]
    excerpt: str  # first 300 chars of the VLM response for notification context


def detect_package_delivery(vlm_text: str, threshold: int = 2) -> DetectionResult:
    """
    Return a DetectionResult for *vlm_text*.

    Parameters
    ----------
    vlm_text:   The assistant content string from the VLM completion.
    threshold:  Minimum score to set detected=True.
    """
    normalised = vlm_text.lower()
    score = 0
    matched_groups: list[str] = []

    for weight, keywords in _KEYWORD_GROUPS:
        for kw in keywords:
            # Use word-boundary matching for single words; substring for phrases.
            if " " in kw:
                if kw in normalised:
                    score += weight
                    matched_groups.append(kw)
                    break
            else:
                if re.search(rf"\b{re.escape(kw)}\b", normalised):
                    score += weight
                    matched_groups.append(kw)
                    break

    excerpt = vlm_text[:300].strip()
    if len(vlm_text) > 300:
        excerpt += "…"

    return DetectionResult(
        detected=score >= threshold,
        score=score,
        matched_groups=matched_groups,
        excerpt=excerpt,
    )
