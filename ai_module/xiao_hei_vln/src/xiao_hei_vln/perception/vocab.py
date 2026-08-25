"""Open-vocabulary class list management for the perception responder.

YOLOv8x-World v2 accepts a different class list per call — but
re-encoding the prompt embeddings takes ~50 ms on a 4090 for ~50
classes. We keep a stable hybrid list:

* A small **prior** of common indoor objects so the scene graph
  accumulates labels between questions (object referenced *next*
  tick).
* **Question-derived nouns** merged on top, so the detector sees the
  task-specific vocabulary too.

The class list is **dedup-stable** so the sidecar's prompt embedding
cache stays warm: the same set of classes always produces the same
ordered tuple, and the responder only calls ``/reload_classes`` when
the set actually changes.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

log = logging.getLogger(__name__)

# Pinned to the arabic_room ground-truth object labels (from the scene's
# object_list.txt) so the open-vocab class list matches what's actually
# in the scene — used to evaluate detector performance against GT
# without vocabulary mismatch. The "unknown" GT placeholder is dropped.
# The cross-scene prior: the 100 most generally-useful labels regenerated from
# all 15 released VLA-3D scenes, capped by cross-scene generality.
#
# Kept SEPARATE from DEFAULT_PRIOR on purpose. `DEFAULT_PRIOR` below is skewed
# toward arabic_room, which is narrow — but it is also what the `perception` and
# `scene_gemini` responders are currently scored with, and swapping the shared
# default would change their behaviour on no evidence. Only `scene_claude` opts
# in, by passing this explicitly:
#
#     Vocabulary(prior=CROSS_SCENE_PRIOR)
#
# It matters most there: object reference gates arrival on the *named* class
# appearing in the scene graph, and an open-vocabulary detector cannot find what
# it was never asked to look for. Half the released object-reference targets
# ("beer bottle", "bowl", "clock", "desk light") are absent from DEFAULT_PRIOR.
CROSS_SCENE_PRIOR: tuple[str, ...] = (
    "balcony door",
    "bed",
    "bedroom light",
    "beer bottle",
    "bench",
    "bird decoration",
    "blanket",
    "book",
    "bookcase",
    "bottle",
    "bowl",
    "box",
    "cabinet",
    "candle",
    "carpet",
    "ceiling",
    "ceiling lamp",
    "ceiling light",
    "chair",
    "chopsticks",
    "clock",
    "coffee cup",
    "coffee machine",
    "coffee table",
    "column",
    "computer monitor",
    "computer mouse",
    "couch",
    "cup",
    "curtain",
    "dining table",
    "dish",
    "door",
    "door frame",
    "drawer",
    "dvd",
    "eye glasses",
    "file",
    "fireplace",
    "floor",
    "flowers",
    "focus light",
    "folder",
    "glass",
    "handle",
    "hanger",
    "horse figurine",
    "kettle",
    "keyboard",
    "kitchen cabinet",
    "kitchen island",
    "lamp",
    "lantern",
    "laptop",
    "light switch",
    "magazine",
    "marker",
    "mattress",
    "microwave",
    "mirror",
    "newspaper",
    "night stand",
    "ottoman",
    "painting",
    "paper",
    "paper holder",
    "pen",
    "phone",
    "photo",
    "picture",
    "pillow",
    "potted plant",
    "quilt",
    "range hood",
    "sculpture",
    "shelf",
    "shower",
    "sink",
    "slipper",
    "sofa",
    "speaker",
    "stool",
    "table",
    "toilet",
    "toilet paper",
    "towel",
    "towel rack",
    "trash bin",
    "trash can",
    "tray",
    "tv",
    "tv cabinet",
    "tv remote",
    "vase",
    "wall",
    "wall lamp",
    "wardrobe",
    "window",
    "window frame",
    "wine bottle",
)

DEFAULT_PRIOR: tuple[str, ...] = (
    "focus light",
    "pillow",
    "potted plant",
    "wall",
    "window",
    "wall lamp",
    "door",
    "glass",
    "sofa",
    "picture",
    "floor",
    "column",
    "vase",
    "lantern",
    "table",
    "carpet",
    "stool",
    "door frame",
    "ceiling lamp",
    "shoes",
    "exterior walls",
    "arabic jar",
    "book",
    "coffee pot",
    "hookah",
    "hookah wire",
    "tray",
    "ceiling",
)

# Words to drop from a question before treating the rest as candidate
# noun labels. Kept conservative — YOLO-World ignores junk classes
# anyway (they just get low scores), so over-removing is fine but
# under-removing rarely hurts.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "from", "for", "with", "by", "and", "or",
    "this", "that", "these", "those", "it", "its", "there", "here",
    "you", "your", "yours", "i", "my", "mine", "we", "our", "ours",
    "how", "what", "which", "where", "when", "who", "whose", "why",
    "many", "much", "some", "any", "more", "most", "less", "least",
    "can", "could", "would", "should", "do", "does", "did", "have",
    "has", "had", "find", "go", "show", "tell", "near", "next",
    "between", "above", "below", "behind", "front", "left", "right",
})


class Vocabulary:
    """Hybrid scene-prior + question-derived class list.

    ``current_classes(question)`` is the only method callers need —
    it returns a stable, deduplicated tuple that the
    :class:`HTTPPerceptionClient` can compare against the last list it
    pushed to the sidecar.
    """

    def __init__(
        self,
        prior: Iterable[str] = DEFAULT_PRIOR,
        *,
        stopwords: frozenset[str] = _STOPWORDS,
    ) -> None:
        self._prior: tuple[str, ...] = tuple(_normalise(p) for p in prior)
        self._stopwords = stopwords

    @property
    def prior(self) -> tuple[str, ...]:
        return self._prior

    def current_classes(self, question_text: str | None) -> tuple[str, ...]:
        """Return the (prior ∪ question-derived) class list for the
        current tick. Order: prior first (preserving its order), then
        question-derived nouns in question order, deduped.
        """
        seen: dict[str, None] = {label: None for label in self._prior}
        if question_text:
            for label in self._extract_nouns(question_text):
                if label not in seen:
                    seen[label] = None
        return tuple(seen.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_nouns(self, text: str) -> list[str]:
        """Pull plausible object labels out of a question.

        v1 heuristic: lowercase, strip punctuation, drop stopwords,
        strip trailing plural ``s`` (so ``"chairs"`` becomes
        ``"chair"`` and matches the prior). Multi-word concepts can be
        added by the caller via ``Vocabulary(prior=...)``; the
        extractor itself emits single words only.
        """
        words = re.findall(r"[a-z]+", text.lower())
        out: list[str] = []
        for w in words:
            if w in self._stopwords or len(w) < 2:
                continue
            singular = _depluralise(w)
            if singular not in out:
                out.append(singular)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(label: str) -> str:
    """Lower-case and collapse whitespace. Keeps internal hyphens."""
    return re.sub(r"\s+", " ", label.strip().lower())


def _depluralise(word: str) -> str:
    """Strip a trailing ``s``/``es`` when the result is still ≥3 chars.

    Conservative on purpose — we only handle the common case
    (``chairs`` → ``chair``, ``boxes`` → ``boxe`` is undesirable so we
    leave words ending in -ses/-xes/-zes/-shes/-ches alone). YOLO-World
    forgives the rest.
    """
    if len(word) < 4:
        return word
    if word.endswith(("ses", "xes", "zes", "shes", "ches")):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word
