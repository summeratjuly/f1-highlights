"""Stage 2: find commentary mentions of the target driver/team, plus
auxiliary signals used by the clarity layer (replay cues).

Each transcript segment (a Whisper-VAD-aligned sentence) either matches
or doesn't. When it matches, the whole segment becomes a mention so
clip boundaries snap cleanly at sentence ends.
"""
from __future__ import annotations

from clarity import contains_replay_cue
from shared import (
    WEIGHT_TEXT_DRIVER,
    WEIGHT_TEXT_TEAM,
    compile_alias_re,
)


def find_mentions(transcript: dict, driver_aliases: list[str],
                  team_aliases: list[str]) -> list[dict]:
    driver_re = compile_alias_re(driver_aliases, word_boundary=True)
    team_re = compile_alias_re(team_aliases, word_boundary=True)

    mentions = []
    for seg in transcript["segments"]:
        text = seg["text"]
        d_hit = driver_re.search(text)
        t_hit = team_re.search(text)
        if not (d_hit or t_hit):
            continue
        reasons = []
        if d_hit:
            reasons.append(f"driver:{d_hit.group(1).lower()}")
        if t_hit:
            reasons.append(f"team:{t_hit.group(1).lower()}")
        mentions.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
            "reasons": reasons,
            "weight": WEIGHT_TEXT_DRIVER if d_hit else WEIGHT_TEXT_TEAM,
        })
    return mentions


def find_replay_spans(transcript: dict) -> list[tuple[float, float]]:
    """Spans of commentary that signal a replay is being shown.

    Used by build_clips to glue replays back to the parent moment so they
    don't appear as orphan flashbacks.
    """
    return [
        (seg["start"], seg["end"])
        for seg in transcript["segments"]
        if contains_replay_cue(seg["text"])
    ]
