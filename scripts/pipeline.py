"""F1 driver/team highlight pipeline.

Usage:
  python pipeline.py \\
      --video /path/to/race.mp4 \\
      --year 2024 \\
      --session race \\
      --driver VER \\
      --output /path/to/ver_highlights.mp4

Required:
  --year YYYY             Season - loads data/grids/{year}.json (2015..2026)
  --session TYPE          practice | qualifying | sprint | race
  --video PATH            Source broadcast video
  --output PATH           Highlight video path to write
  --driver ID             Driver ID from that year's grid (e.g. VER, NOR)
                          and/or
  --team   ID             Team ID from that year's grid (e.g. RBR, MCL)
                          Team mode includes every driver on that team.

Tuning:
  --fps FLOAT             Frame sample rate for OCR (default 1.0)
  --threshold FLOAT       Relevance score cutoff (default 1.0)
  --pre-roll FLOAT        Seconds of lead-in before each segment (default 2)
  --post-roll FLOAT       Seconds of tail after each segment (default 3)
  --merge-gap FLOAT       Merge segments closer than this in seconds (default 8)
  --min-len FLOAT         Drop clips shorter than this (default 6)
  --smooth                Re-encode with crossfades (slower, smoother)
  --xfade FLOAT           Crossfade duration when --smooth (default 0.4)
  --workdir DIR           Cache directory (default: <output>.work/)
  --whisper-model NAME    faster-whisper model (default small.en)
  --text-only             Skip OCR pass - commentary mentions only (fast)
  --visual-only           Skip transcript pass - OCR only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from transcribe import transcribe
from analyze_text import find_mentions, find_replay_spans
from analyze_video import detect_visual_presence, find_replay_graphic_spans
from analyze_livery import detect_livery
from build_bridges import build_bridges
from build_timeline import build_timeline
from build_clips import build_clips
from compile import compile_highlights
from move_to_nas import archive_run as archive_run_to_nas
from refine_boundaries import refine_clips
from shared import CLIPS_JSON_NAME, HIT_COLOR, HIT_NUMBER, probe_duration


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s - out[-1][1] <= 1.5:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(round(s, 2), round(e, 2)) for s, e in out]


def _print_review(clips: list[dict], duration: float) -> None:
    """One-line-per-clip summary for the model-layer review.

    Claude (when running this skill) reads this to spot orphan-reference
    clips and near-duplicate moments worth dropping with --drop-clips.
    """
    print()
    print("[f1-highlights] === clip review ===")
    print(f"[f1-highlights] {'idx':>3}  {'tstart':>7}  {'dur':>5}  {'event':<14}  {'flags':<10} text")
    for i, c in enumerate(clips):
        flags = []
        if c.get("orphan_refs"):
            flags.append("ORPHAN")
        flag_str = ",".join(flags) or "-"
        text = (c.get("transcript") or "").replace("\n", " ")[:120]
        print(f"[f1-highlights] {i:>3}  {c['start']:>7.1f}  {c['duration']:>5.1f}  "
              f"{c.get('event_type', 'mention'):<14}  {flag_str:<10} {text}")
    print()


def _parse_drop_clips(arg: str | None) -> set[int]:
    if not arg:
        return set()
    out: set[int] = set()
    for tok in arg.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.add(int(tok))
    return out


DATA_DIR = HERE.parent / "data"
GRIDS_DIR = DATA_DIR / "grids"
TEAM_COLORS_PATH = DATA_DIR / "team_colors.json"
VALID_SESSIONS = {"practice", "qualifying", "sprint", "race"}


# Session-aware defaults. Qualifying/sprint are shorter and more
# driver-dense than a grand prix, so defaults shift accordingly.
SESSION_DEFAULTS = {
    "practice":   {"pre_roll": 2.0, "post_roll": 3.0, "merge_gap": 8.0,  "min_len": 6.0},
    "qualifying": {"pre_roll": 1.5, "post_roll": 2.5, "merge_gap": 5.0,  "min_len": 5.0},
    "sprint":     {"pre_roll": 2.0, "post_roll": 3.0, "merge_gap": 6.0,  "min_len": 6.0},
    "race":       {"pre_roll": 2.0, "post_roll": 3.0, "merge_gap": 8.0,  "min_len": 6.0},
}


def _load_grid(year: int) -> dict:
    path = GRIDS_DIR / f"{year}.json"
    if not path.exists():
        available = sorted(int(p.stem) for p in GRIDS_DIR.glob("*.json"))
        raise SystemExit(
            f"no grid for {year}. Available: {available}. "
            f"Add data/grids/{year}.json to support this season."
        )
    return json.loads(path.read_text())


def _team_name_to_id(grid: dict, team_name: str) -> str | None:
    for t in grid["teams"]:
        if t["name"].lower() == team_name.lower():
            return t["id"]
    return None


def _resolve_targets(grid: dict, driver_id: str | None,
                     team_id: str | None) -> tuple[list[str], list[str], list[dict], str]:
    """Returns (driver_aliases, team_aliases, livery_targets, label).

    livery_targets: list of dicts consumed by analyze_livery.detect_livery.
    Each dict: {team_id, driver_id?, driver_number?}.
    """
    drivers = grid["drivers"]
    teams = grid["teams"]

    driver_aliases: list[str] = []
    team_aliases: list[str] = []
    livery_targets: list[dict] = []
    label_parts: list[str] = []

    if driver_id:
        d = next((x for x in drivers if x["id"] == driver_id.upper()), None)
        if not d:
            known = ", ".join(x["id"] for x in drivers)
            raise SystemExit(f"driver '{driver_id}' not in {grid['year']} grid. Known: {known}")
        driver_aliases.extend(d["aliases"])
        label_parts.append(f"{d['name']} ({d['team']}, #{d['number']})")
        tid = _team_name_to_id(grid, d["team"])
        if tid:
            livery_targets.append({
                "team_id": tid,
                "driver_id": d["id"],
                "driver_number": d.get("number"),
            })

    if team_id:
        t = next((x for x in teams if x["id"] == team_id.upper()), None)
        if not t:
            known = ", ".join(x["id"] for x in teams)
            raise SystemExit(f"team '{team_id}' not in {grid['year']} grid. Known: {known}")
        team_aliases.extend(t["aliases"])
        team_drivers = [d for d in drivers if d["team"].lower() == t["name"].lower()]
        for d in team_drivers:
            driver_aliases.extend(d["aliases"])
            livery_targets.append({
                "team_id": t["id"],
                "driver_id": d["id"],
                "driver_number": d.get("number"),
            })
        label_parts.append(f"{t['name']} [{', '.join(d['name'] for d in team_drivers)}]")

    if not (driver_aliases or team_aliases):
        raise SystemExit("specify --driver and/or --team")

    driver_aliases = list(dict.fromkeys(driver_aliases))
    team_aliases = list(dict.fromkeys(team_aliases))

    # Deduplicate livery targets by (team_id, driver_id).
    seen = set()
    uniq_targets = []
    for t in livery_targets:
        key = (t.get("team_id"), t.get("driver_id"))
        if key in seen:
            continue
        seen.add(key)
        uniq_targets.append(t)

    return driver_aliases, team_aliases, uniq_targets, " / ".join(label_parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--year", required=True, type=int)
    ap.add_argument("--session", required=True, choices=sorted(VALID_SESSIONS))
    ap.add_argument("--driver")
    ap.add_argument("--team")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--pre-roll", type=float, default=None)
    ap.add_argument("--post-roll", type=float, default=None)
    ap.add_argument("--merge-gap", type=float, default=None)
    ap.add_argument("--min-len", type=float, default=None)
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--xfade", type=float, default=0.4)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--whisper-model", default="small.en")
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--visual-only", action="store_true")
    ap.add_argument("--no-livery", action="store_true",
                    help="Skip livery colour + number matching (useful if team colours are unreliable for this broadcast)")
    ap.add_argument("--drop-clips", default=None,
                    help="Comma-separated clip indices (from the previous run's review) to drop. "
                         "Used by the model-layer review to surgically remove orphan-reference or duplicate clips.")
    ap.add_argument("--bridges", action="store_true", default=None,
                    help="Insert interstitial 'bridge' cards between clips with large gaps. "
                         "Default: on for race/sprint, off for practice/qualifying.")
    ap.add_argument("--no-bridges", dest="bridges", action="store_false",
                    help="Disable bridge cards.")
    ap.add_argument("--bridge-min-gap", type=float, default=30.0,
                    help="Skip bridges for gaps shorter than this (default 30s).")
    ap.add_argument("--bridge-duration", type=float, default=10.0,
                    help="Seconds each bridge card stays on screen (default 10 — 3 events need reading time).")
    ap.add_argument("--refine-clips", action="store_true", default=True,
                    help="LLM-refine clip boundaries (default on). Avoids cuts that "
                         "open with orphan pronouns or end on dangling lists.")
    ap.add_argument("--no-refine-clips", dest="refine_clips", action="store_false",
                    help="Skip the LLM clip-boundary refiner (faster, but cuts may feel less natural).")
    ap.add_argument("--archive-to-nas", action="store_true", default=False,
                    help="After compile, move source / highlight / workdir to "
                         "/Volumes/Media/Recording/<year>/ following the canonical naming. "
                         "Confirmed upfront by the caller — pipeline does NOT prompt mid-run.")
    ap.add_argument("--archive-race", default=None,
                    help="Lowercase canonical race token (e.g. canada, silverstone). Required with --archive-to-nas.")
    ap.add_argument("--archive-round", type=int, default=None,
                    help="F1 calendar round number for the race-year. Required with --archive-to-nas.")
    ap.add_argument("--archive-target", default="ver",
                    help="Target focus code used in the destination filename (default 'ver').")
    ap.add_argument("--archive-session-topic", default=None,
                    help="Optional sub-topic for the session in the NAS stem "
                         "(mainly for press: drivers / teamprincipals / postrace).")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    # Validate archive args UPFRONT so a 60-min run doesn't fail at the
    # last step on a typo. The user (or the calling skill) commits to
    # archiving at invocation time — pipeline never prompts mid-run.
    if args.archive_to_nas:
        if not args.archive_race or args.archive_round is None:
            raise SystemExit(
                "[f1-highlights] --archive-to-nas requires --archive-race and "
                "--archive-round (F1 calendar round number)"
            )
        nas_base = Path("/Volumes/Media/Recording")
        if not nas_base.exists():
            raise SystemExit(
                f"[f1-highlights] --archive-to-nas: NAS not mounted at {nas_base}. "
                f"Mount the share before launching."
            )
        if not (nas_base / "README.md").exists():
            raise SystemExit(
                f"[f1-highlights] --archive-to-nas: missing {nas_base}/README.md "
                f"(naming-convention reference). Refusing to launch blindly."
            )

    # Apply session-aware defaults for any flag the user left unset.
    defaults = SESSION_DEFAULTS[args.session]
    pre_roll  = args.pre_roll  if args.pre_roll  is not None else defaults["pre_roll"]
    post_roll = args.post_roll if args.post_roll is not None else defaults["post_roll"]
    merge_gap = args.merge_gap if args.merge_gap is not None else defaults["merge_gap"]
    min_len   = args.min_len   if args.min_len   is not None else defaults["min_len"]

    grid = _load_grid(args.year)
    driver_aliases, team_aliases, livery_targets, label = _resolve_targets(grid, args.driver, args.team)

    workdir = args.workdir or args.output.with_suffix(".work")
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"[f1-highlights] season: {args.year} | session: {args.session}")
    print(f"[f1-highlights] target: {label}")
    print(f"[f1-highlights] driver aliases: {driver_aliases}")
    print(f"[f1-highlights] team aliases: {team_aliases}")
    print(f"[f1-highlights] clip rules: pre={pre_roll}s post={post_roll}s merge_gap={merge_gap}s min_len={min_len}s")
    print(f"[f1-highlights] workdir: {workdir}")

    if args.visual_only:
        transcript = {"duration": 0.0, "segments": []}
        text_mentions: list[dict] = []
    else:
        print("[f1-highlights] stage 1: transcribing (cached if resumed)...")
        transcript = transcribe(args.video, workdir, model_size=args.whisper_model)
        print(f"[f1-highlights]   {len(transcript['segments'])} segments, {transcript['duration']:.1f}s")
        print("[f1-highlights] stage 2: scanning transcript for mentions...")
        text_mentions = find_mentions(transcript, driver_aliases, team_aliases)
        print(f"[f1-highlights]   {len(text_mentions)} mention segments")

    if args.text_only:
        visual_hits: list[dict] = []
    else:
        print("[f1-highlights] stage 3: sampling frames + OCR...")
        visual_hits = detect_visual_presence(
            args.video, workdir, driver_aliases, team_aliases, fps=args.fps
        )
        print(f"[f1-highlights]   {len(visual_hits)} visual hits")

    if args.text_only or args.no_livery:
        livery_hits: list[dict] = []
    else:
        print("[f1-highlights] stage 3.5: livery colour + number detection...")
        livery_hits = detect_livery(workdir, TEAM_COLORS_PATH, args.year, livery_targets)
        color_hits = sum(1 for h in livery_hits if h["type"] == HIT_COLOR)
        num_hits = sum(1 for h in livery_hits if h["type"] == HIT_NUMBER)
        print(f"[f1-highlights]   {color_hits} colour hits, {num_hits} number hits")

    duration = transcript["duration"] or probe_duration(args.video)

    # Replay binding: glue commentary "let's see that again" + OCR REPLAY
    # graphic spans, so trailing replays of overtakes/incidents stay
    # attached to the moment they describe.
    replay_spans: list[tuple[float, float]] = []
    if not args.visual_only:
        replay_spans += find_replay_spans(transcript)
    if not args.text_only:
        replay_spans += find_replay_graphic_spans(workdir)
    replay_spans = _merge_spans(replay_spans)
    if replay_spans:
        print(f"[f1-highlights]   {len(replay_spans)} replay spans detected (will glue to parent moments)")

    print("[f1-highlights] stage 4: building relevance timeline...")
    scores = build_timeline(duration, text_mentions, visual_hits, livery_hits)

    print("[f1-highlights] stage 5: segmenting into narrative clips...")
    clips = build_clips(
        scores, transcript,
        text_mentions=text_mentions,
        replay_spans=replay_spans,
        session=args.session,
        threshold=args.threshold,
        pre_roll=pre_roll,
        post_roll=post_roll,
        merge_gap=merge_gap,
        min_len=min_len,
    )
    if not clips:
        raise SystemExit("[f1-highlights] no clips survived thresholding - lower --threshold or widen aliases")

    if args.refine_clips:
        print(f"[f1-highlights] stage 5.3: LLM-refining {len(clips)} clip boundaries (concurrency=3)...")
        clips, refine_counts = refine_clips(clips, transcript)
        # Re-validate min_len + sort by start (refinement could shrink some clips).
        clips = [c for c in clips if c["duration"] >= min_len]
        clips.sort(key=lambda c: c["start"])
        print(f"[f1-highlights]   refinement sources: {refine_counts}")

    drop_set = _parse_drop_clips(args.drop_clips)
    if drop_set:
        kept = [c for i, c in enumerate(clips) if i not in drop_set]
        print(f"[f1-highlights]   dropped {len(clips) - len(kept)} clips: {sorted(drop_set)}")
        clips = kept

    total = sum(c["duration"] for c in clips)
    print(f"[f1-highlights]   {len(clips)} clips, total {total:.1f}s "
          f"({total / duration * 100:.1f}% of source)")
    event_counts: dict[str, int] = {}
    for c in clips:
        event_counts[c.get("event_type", "mention")] = event_counts.get(c.get("event_type", "mention"), 0) + 1
    print(f"[f1-highlights]   event types: {event_counts}")

    (workdir / CLIPS_JSON_NAME).write_text(json.dumps({
        "year": args.year,
        "session": args.session,
        "target": label,
        "duration": duration,
        "replay_spans": replay_spans,
        "clips": clips,
    }, indent=2))

    _print_review(clips, duration)

    bridges_enabled = args.bridges if args.bridges is not None else (args.session in ("race", "sprint"))
    bridges: dict[int, Path] = {}
    if bridges_enabled and len(clips) >= 2:
        primary_team = livery_targets[0]["team_id"] if livery_targets else None
        print(f"[f1-highlights] stage 5.5: building bridge cards (min_gap={args.bridge_min_gap}s, dur={args.bridge_duration}s)...")
        bridges = build_bridges(
            clips, transcript, replay_spans, workdir,
            team_id=primary_team, year=args.year,
            session=args.session,
            focus_label=label,
            focus_aliases=driver_aliases,
            ocr_fps=args.fps,
            min_gap=args.bridge_min_gap,
            duration=args.bridge_duration,
        )
        print(f"[f1-highlights]   {len(bridges)} bridges generated")

    print(f"[f1-highlights] stage 6: compiling -> {args.output}")
    compile_highlights(
        args.video, clips, workdir, args.output,
        smooth=args.smooth, crossfade=args.xfade,
        bridges=bridges,
    )

    if args.archive_to_nas:
        print(f"[f1-highlights] stage 7: archiving to NAS...")
        archive_run_to_nas(
            source=args.video,
            highlight=args.output,
            workdir=workdir,
            year=args.year,
            race=args.archive_race,
            round_no=args.archive_round,
            target=args.archive_target,
            session=args.session,  # adds _quali / _sprint / _press / _fp* suffix when not race
            session_topic=args.archive_session_topic,  # e.g. press → drivers / postrace
            interactive=False,  # caller already confirmed upfront
        )

    print("[f1-highlights] done.")


if __name__ == "__main__":
    main()
