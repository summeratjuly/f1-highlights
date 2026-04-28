"""Clarity-of-the-moment helpers.

Classify each high-relevance region by *what kind of moment* it is, so we
can pad it to feel self-contained: an overtake needs build-up + outcome,
a crash needs cause + reaction, a pit stop needs in/out laps, a passing
mention needs almost no padding. Replays of a moment get glued back to
their parent so they don't appear as orphan flashbacks.

Most rules are pure keyword lookups so they run cheaply over the
transcript that already lives in memory. The only signal sourced from
OCR is the "REPLAY" graphic, detected from the cached ocr.json.
"""
from __future__ import annotations

import re

# Per-event padding (pre_roll, post_roll) in seconds. Tuned so each event
# type lands as a self-contained micro-story in the final reel.
EVENT_PADDING = {
    "overtake":       (4.0, 6.0),  # need the chase + the result
    "incident":       (3.0, 8.0),  # cause + reaction (often a replay follows)
    "pit":            (3.0, 5.0),  # in-lap context + out-lap commentary
    "qualifying_lap": (2.0, 4.0),  # sector commentary + final time
    "start":          (5.0, 6.0),  # lights, launch, first corner
    "finish":         (3.0, 8.0),  # last lap + checkered + reaction
    "mention":        (1.5, 2.5),  # passive name-check, low action
}

# Keyword tables. Order matters within a category — earlier = more
# specific. A region matched in multiple categories goes to the highest-
# priority one (per EVENT_PRIORITY).
EVENT_KEYWORDS: dict[str, list[str]] = {
    "incident": [
        "crash", "crashed", "crashes", "spin", "spun", "spinning",
        "into the wall", "into the barrier", "into the gravel",
        "into the grass", "off the track", "lost it",
        "incident", "contact", "tangle", "collision", "collided",
        "yellow flag", "double yellow", "red flag",
        "safety car", "virtual safety", "vsc",
        "puncture", "damage", "broken front wing", "broken wing",
        "stricken", "in the wall", "off he goes",
    ],
    "overtake": [
        "passes", "passed", "overtakes", "overtaken", "overtake",
        "alongside", "side by side", "wheel to wheel",
        "down the inside", "down the outside",
        "around the outside", "around the inside",
        "into the lead", "takes the lead", "snatches the lead",
        "moves ahead of", "moves into", "ahead of",
        "drs open", "drs is", "in the slipstream", "slipstreams",
        "lunge", "dive", "dived", "diving", "switchback", "cut back",
        "is through", "gets through", "makes the move",
    ],
    "pit": [
        "pits", "into the pits", "into the pit lane", "pit entry",
        "pit stop", "stationary", "boxes",
        "fresh tyres", "fresh tires", "tyre change", "tire change",
        "out of the pits", "rejoins", "rejoined", "rejoining",
        "in lap", "out lap", "undercut", "overcut",
    ],
    "qualifying_lap": [
        "purple", "purple sector", "personal best",
        "fastest lap", "fastest of all", "first sector",
        "second sector", "final sector",
        "going green", "green sector",
        "provisional pole", "pole position", "on pole",
        "improving", "improved", "lap time",
    ],
    "start": [
        "lights out", "off they go", "away cleanly",
        "the start of the", "off the line", "got a good start",
        "bogged down", "great start", "dreadful start",
        "into turn one", "into turn 1",
    ],
    "finish": [
        "chequered flag", "checkered flag", "crosses the line",
        "takes the win", "takes victory", "wins the",
        "first across", "victory for", "the win for",
    ],
}

# Higher priority wins when a region matches multiple categories.
# Incidents trump overtakes (a crash that passed someone is still a crash).
EVENT_PRIORITY = ["incident", "finish", "start", "overtake", "pit", "qualifying_lap"]

# Cues that signal a replay is being shown. Used both alone (in commentary)
# and together with the OCR "REPLAY" graphic.
REPLAY_CUES = [
    "let's see that again", "let's look at that again",
    "look at that again", "watch this again", "watch that again",
    "another look", "another angle", "different angle",
    "from a different angle", "from this angle",
    "we'll show you", "we will show you",
    "as we saw", "moments ago", "a moment ago",
    "rewind", "going back to", "back to that moment",
    "that's the replay", "here's the replay", "here is the replay",
]
REPLAY_GRAPHIC_TOKENS = {"replay", "rerun"}

# Sentences ending in these tokens indicate the thought continues — extend
# the cut to the next sentence rather than chop here.
_DANGLING_END_TOKENS = {
    "and", "but", "or", "so", "because", "as", "while", "when",
    "if", "then", "yet", "though", "although",
    "the", "a", "an", "to", "of", "in", "on", "at", "with", "for",
}

# Sentences that are essentially just a pronoun reference are not
# self-contained and should not end a clip.
_PRONOUN_ONLY_HEADS = {
    "he", "she", "it", "they", "him", "her", "his", "their",
    "that's", "this", "those", "these",
}

# Phrases that point outside the current clip — the model layer flags
# these so Claude can decide whether to extend or drop the clip.
ORPHAN_REFERENCE_CUES = [
    "earlier", "before", "remember", "as we saw", "we saw earlier",
    "moments ago", "a moment ago", "just like", "exactly like",
    "as i mentioned", "as we mentioned", "going back to",
    "as you saw", "you'll remember",
]


_CLEAN_RE = re.compile(r"[^a-z0-9' ]+")


def _norm(text: str) -> str:
    return _CLEAN_RE.sub(" ", text.lower()).strip()


def classify_event(text: str) -> str:
    """Return the best-matching event category, or 'mention'."""
    norm = _norm(text)
    matched: set[str] = set()
    for cat, kws in EVENT_KEYWORDS.items():
        for kw in kws:
            if kw in norm:
                matched.add(cat)
                break
    for cat in EVENT_PRIORITY:
        if cat in matched:
            return cat
    return "mention"


def contains_replay_cue(text: str) -> bool:
    norm = _norm(text)
    return any(cue in norm for cue in REPLAY_CUES)


def ocr_tokens_say_replay(tokens: list[dict]) -> bool:
    for tok in tokens:
        if tok.get("text", "").strip().lower() in REPLAY_GRAPHIC_TOKENS:
            return True
    return False


def ends_dangling(text: str) -> bool:
    """True if the sentence-final token suggests the speaker isn't done."""
    norm = _norm(text)
    if not norm:
        return False
    last = norm.split()[-1] if norm.split() else ""
    return last in _DANGLING_END_TOKENS


def is_pronoun_only_head(text: str) -> bool:
    """True if the sentence opens with an unresolved pronoun reference."""
    norm = _norm(text)
    if not norm:
        return False
    head = norm.split()[0] if norm.split() else ""
    return head in _PRONOUN_ONLY_HEADS


def find_orphan_references(text: str) -> list[str]:
    """Return any orphan-reference cues found in the text. Empty list = self-contained."""
    norm = _norm(text)
    return [cue for cue in ORPHAN_REFERENCE_CUES if cue in norm]
