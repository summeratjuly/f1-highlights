---
name: f1-highlights
description: Generate a driver- or team-focused highlight video from an F1 broadcast (practice, qualifying, sprint, or race). Fuses Whisper commentary transcription with OCR on broadcast graphics to find every moment the target driver is on screen or being talked about, then compiles a story-flow cut that snaps to sentence boundaries so commentary never cuts mid-word. Use when the user asks to "make highlights for X", "extract VER scenes", "cut a McLaren reel", or similar from a local F1 video file. Claude must confirm the season year and session type before running, because driver/team mapping depends on both.
---

# f1-highlights

A six-stage pipeline that turns a 1-3 hour F1 broadcast into a highlight reel for a specific driver or team, across the 2015-2026 seasons.

The reel is shaped by one quality bar: **clarity of the moment** — the viewer of any clip should instantly understand what happened and why. The pipeline enforces this in code (event-aware padding, replay binding, sentence-end integrity, race bookends); the post-run review surfaces clips that still need a model-layer judgment (orphan references, near-duplicates).

## When to use

User has a **local F1 video file** (practice / qualifying / sprint / race) and wants a shorter cut focused on one driver or team. If they don't have a local file, help them record one first (see `~/f1-tv-recorder/`).

If the source is an **OBS / screen recording** that includes browser chrome, the desktop, or moments before/after the user toggled fullscreen, run the preprocessing step first to trim the recording down to just the fullscreen-broadcast portions:

```bash
python scripts/preprocess_recording.py /path/to/raw.mov /path/to/clean.mov
```

The script samples frames at 1fps, uses pixel-edge variance to detect when the broadcast is fullscreen vs. when UI chrome / taskbar / loading screens are visible, then stream-copies only the fullscreen runs into the output file. Writes a `<output>.preprocess.json` audit alongside listing kept/dropped segments. Tunable: `--sample-fps`, `--min-run-sec`, `--stdev-threshold`. Then run the regular pipeline on the cleaned output.

## Always ask the user first

Before running the pipeline, confirm with the user:

1. **Season year** (2015-2026). The grid changes every year — Hamilton was Mercedes through 2024 and Ferrari from 2025; "Kimi" meant Raikkonen through 2021 and means Antonelli from 2025; teams renamed (Toro Rosso → AlphaTauri → RB; Alfa Romeo → Kick Sauber → Audi). Do not guess.
2. **Session type**: `practice` | `qualifying` | `sprint` | `race`. This adjusts clip defaults (qualifying uses shorter pre/post-roll and merge-gap than a race).
3. **Target**: driver (`--driver ID`) or team (`--team ID`) or both. Look up the ID in `data/grids/{year}.json` — don't invent codes.

## How it works

```
video.mp4
  │
  ├─ 1. transcribe.py      → audio → faster-whisper → transcript.json (word-level timestamps)
  ├─ 2. analyze_text.py    → find alias hits in commentary → sentence-scoped mentions
  ├─ 3. analyze_video.py   → sample frames @ fps → easyocr → ocr.json + visual.json (name hits)
  ├─ 3½ analyze_livery.py  → reuse frames + OCR → team-colour coverage + colour-gated number hits
  ├─ 4. build_timeline.py  → fuse 4 signals → per-second relevance score (smoothed)
  ├─ 5. build_clips.py     → threshold + merge + pad + snap-to-sentence-boundary
  └─ 6. compile.py         → ffmpeg cut + concat (stream-copy or crossfade)
       ↓
     highlights.mp4
```

Each stage writes to `<output>.work/` and is resumable — rerunning with different thresholds skips transcription/OCR. Signal weights (per frame or per mention):

| signal                    | weight | granularity | when it fires                                                 |
|---------------------------|--------|-------------|---------------------------------------------------------------|
| transcript driver mention | 2.0    | driver      | commentator said the driver's name                            |
| transcript team mention   | 1.0    | team        | commentator said the team's name                              |
| OCR driver/team name      | 3.0/1.5| driver/team | graphic with name overlays the frame                          |
| livery colour coverage    | 0.5–2.0| team        | team's livery dominates a large area of the frame             |
| colour-gated car number   | 2.5    | driver      | driver's race number detected AND surrounded by team colour   |

## Usage

```bash
cd ~/.claude/skills/f1-highlights
source .venv/bin/activate   # one-time setup below

python scripts/pipeline.py \
  --video ~/Movies/2024_monaco_race.mp4 \
  --year 2024 --session race \
  --driver VER \
  --output ~/Movies/monaco_2024_ver.mp4
```

Team reel (includes every driver on that team in that season):
```bash
python scripts/pipeline.py --video race.mp4 --year 2025 --session race --team MCL --output mcl_2025.mp4
```

Smoother transitions (re-encodes with crossfades, ~2-3x slower):
```bash
python scripts/pipeline.py --video q.mp4 --year 2024 --session qualifying --driver NOR --output out.mp4 --smooth
```

Useful flags:
- `--threshold 0.5` → more inclusive (longer highlight)
- `--threshold 2.0` → stricter (driver-centric only)
- `--pre-roll 4 --post-roll 5` → more context around each moment (overrides session defaults)
- `--merge-gap 15` → fewer, longer, smoother-flowing clips
- `--text-only` → skip both OCR and livery (3-5x faster, misses shots with no graphic and no mention)
- `--no-livery` → keep OCR name matching but skip colour/number (useful if a broadcast has unusual colour grading or the team shares colours with a neighbour car)
- `--fps 0.5` → sample frames every 2s (faster, may miss brief on-screens)

## One-time setup

```bash
cd ~/.claude/skills/f1-highlights
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg must be installed on the system (`brew install ffmpeg`).
```

First run downloads Whisper (~150MB for `small.en`) and easyocr detection models.

## Data files — one per season

`data/grids/{year}.json` for each 2015-2026 season. Each file contains:
- `drivers[]` — `id`, `name`, `team`, `number`, `aliases` (lowercase substrings to match). The 3-letter `id` is what the user passes to `--driver`. `number` drives car-number matching.
- `teams[]`   — `id`, `name`, `aliases`. The `id` is what the user passes to `--team`.
- `_notes`    — mid-season driver swaps, sub drivers, notable changes for that year.

`data/team_colors.json` — livery colour palette for HSV matching:
- `default.<team_id>` — list of hex colours used as HSV centroids (multi-tone = multiple entries).
- `overrides.<year>.<team_id>` — year-specific override (e.g. `2020.MER = "#000000"` for the black Mercedes, `2024.WIL = "#64C4FF"` for the lighter Williams blue, `2015/2017.MCL` for the silver/transitional McLaren before papaya returned).
- Skipped if a team has no colour data — livery signal just doesn't fire for that team, other signals unaffected.

Alias discipline applied across seasons:
- Last name and full name are always included.
- First name alone is included **only** when unambiguous that season. E.g., "Nico" is omitted from 2015-16 because both Rosberg and Hulkenberg raced; included from 2017 on (Hulkenberg only).
- Raikkonen-spelling variants (`räikkönen`), Pérez accented forms, and "hülkenberg" are included to survive Whisper's transcription choices.

To add a new season, drop a new file into `data/grids/` following the same shape and run it — nothing else to configure.

## Clarity rules — baked into code

Each region of relevance is classified by *what kind of moment it is* and padded so the clip lands as a self-contained micro-story:

| event           | pre-roll | post-roll | what it captures                              |
|-----------------|----------|-----------|-----------------------------------------------|
| incident        | 3.0s     | 8.0s      | cause + impact + reaction (replay glued in)   |
| finish          | 3.0s     | 8.0s      | last lap + checkered + first reaction         |
| start           | 5.0s     | 6.0s      | lights, launch, first corner                  |
| overtake        | 4.0s     | 6.0s      | the chase + the move + the result             |
| pit             | 3.0s     | 5.0s      | in-lap context + tire change + out-lap        |
| qualifying_lap  | 2.0s     | 4.0s      | sector commentary + final time                |
| mention (default) | 1.5s   | 2.5s      | passive name-check, low action                |

Session defaults are the floor — event-specific padding is only ever applied if it's larger.

Other clarity rules in code (build_clips.py + clarity.py):

- **Replay binding**: `let's see that again`, `from a different angle`, "REPLAY" graphic in OCR → glued to the preceding moment so the replay isn't an orphan flashback.
- **Sentence-end integrity**: a clip never ends on a dangling conjunction (`...and`, `...but`, `...because`) or with a pronoun-only sentence opening it. Cut extends to the next full sentence (capped 4s).
- **Race bookends**: in race sessions, the first 9 minutes (start/grid) and last 4 minutes (finish/podium) lower threshold to 0.5 so target activity in those windows is always covered.
- **Bridge cards**: between adjacent clips with a gap > 30s (and not covered by a replay span, and not before race start), insert a 10-second 3-panel "triptych" card summarising up to 3 narrative events from the gap. Each panel = keyframe + ALL-CAPS title + 12–20-word caption. Captions and keyframe-anchor timestamps are written by Claude (via the Agent SDK using your existing Claude Code login — no separate API key) so the keyframe + caption stay anchored to the same moment. Falls back to a deterministic captioner if the SDK call fails. **Race-start detection** skips pre-grid filler: first clip with `event_type == "start"` OR first transcript hit on "lights out" / "off they go" / "off the line" / etc. Layout is locked in `render_card._TRIPTYCH_DEFAULTS` (override via `data/templates/bridge_card.json`) so every card across every reel is rendered by the same code path. When bridges are present, compile.py force-re-encodes (mixed-source MP4s don't survive stream-copy concat). Flags: `--no-bridges`, `--bridge-min-gap`, `--bridge-duration`.

## Model layer — post-run review

After the pipeline completes it prints a one-line-per-clip review (also written to `<workdir>/clips.json` with full transcript windows). Each line shows: index, start time, duration, event type, flags (e.g. `ORPHAN`), and a transcript excerpt.

When Claude runs this skill, after the pipeline finishes it should:

1. Read the review output (or `clips.json`).
2. **Orphan-reference check**: any clip flagged `ORPHAN` contains commentary like *"as we saw earlier"*, *"remember that"*, *"just like before"*. For each, judge from the transcript whether the referenced moment is in another clip:
   - If yes → consider widening `--merge-gap` so they bind, or rerun with `--threshold` lower so the referenced moment crosses the bar.
   - If no → drop the orphan with `--drop-clips <idx>`.
3. **Near-duplicate check**: if two adjacent clips share an event type and similar transcript content (e.g. both `mention` clips of "Verstappen still leading"), keep the more informative one and drop the rest.
4. **Story flow**: scan the event-type sequence — a reel that's `mention,mention,mention,mention,overtake` for a 30-min cut probably needs `--threshold` raised. A reel with no `start` or `finish` for a race needs investigation.
5. Apply edits with `--drop-clips A,B,C` (cuts/transcribe/OCR are cached, so the rerun is just stages 4–6 ≈ a few seconds).

The model layer's job is the narrative judgment that keyword rules can't reach. Don't second-guess obvious wins (clean overtakes, crashes with replays); focus on the flagged or borderline clips.

## Session-aware defaults

Set automatically from `--session` unless you pass an explicit flag:

| session    | pre_roll | post_roll | merge_gap | min_len |
|------------|----------|-----------|-----------|---------|
| practice   | 2.0      | 3.0       | 8.0       | 6.0     |
| qualifying | 1.5      | 2.5       | 5.0       | 5.0     |
| sprint     | 2.0      | 3.0       | 6.0       | 6.0     |
| race       | 2.0      | 3.0       | 8.0       | 6.0     |

Qualifying has shorter runs per driver, so tighter timing avoids bleeding into other drivers' laps. A race has long green-flag stretches where merge-gap=8s keeps narrative continuity.

## Tuning knobs, in order of likely usefulness

1. **`--threshold`** — most impactful. Start at 1.0, raise if the reel is too long/noisy, lower if scenes are missing.
2. **`--merge-gap`** — larger values make smoother story flow by keeping in-between footage.
3. **`--pre-roll` / `--post-roll`** — add context around each relevant moment.
4. **`--fps`** — OCR sampling. Drop to 0.5 for 2-hour races if full-fps is slow.

## Known limits — be upfront with the user

- **Wide racing shots with no graphic**: OCR catches onboards, replays, timing tower, team radio, interviews. The livery signal catches the *team* in wide shots even without a graphic. Teammate separation in those shots relies on the car number being OCR-readable (best in pit/grid/formation, unreliable during racing motion blur).
- **Colour gating on numbers is strict by design**: "P4" next to the lap counter won't mis-fire as Norris, because the team-colour region check only passes when a papaya-orange car is actually around the number. Downside: some clean nose-number shots may still fail the gate on unusual broadcast cameras.
- **Whisper mis-hears names** in overlapping commentary. Adding misheard spellings to that year's `aliases` is the fix.
- **Teammate confusion**: if the user wants "just NOR, not PIA", use `--driver NOR` (not `--team MCL`) — team mode deliberately includes every driver on that team that season.
- **Stream-copy cuts snap to GOP boundaries** so clip start times may shift up to ~2s. Use `--smooth` if you care about precision at the cost of encode time.
- **Long races cost time**: ~2hr race takes roughly 15-25min to transcribe + 10-15min to OCR at 1fps on an M-series Mac. Use `--text-only` for a quick first pass.
- **Historical grid accuracy**: mid-season substitutes (e.g., Bearman 2024 Jeddah, Hulkenberg COVID sub 2022, Kubica sub 2021) are included where I was confident. For a specific race with unusual lineup, check and edit the year file before running.
- **2026 grid is tentative**: Audi takeover and Cadillac entry are in flux — verify the file against the official entry list for that round.
- **Team-colour collisions**: Red Bull and RB/AlphaTauri both run blue; if you target Red Bull and RB is in the frame, colour coverage may spike. Number gating mitigates at driver-level; run `--team RBR` and the colour hits for AT/RB cars will be discarded as off-team.

## When Claude invokes this skill

1. **Confirm year, session type, and target** with the user. Do not assume.
2. Open `data/grids/{year}.json` to map a spoken name to an `id` and verify the target exists that season. Flag mid-season changes if relevant (e.g., "in 2025 Lawson started at Red Bull but was demoted after 2 rounds — is this the Australia race or later?").
3. Activate the skill's venv and run `pipeline.py` with `--year`, `--session`, `--driver`/`--team`.
4. **Review the per-clip output** the pipeline prints at the end. Apply the model-layer review (see "Model layer" section): drop ORPHAN-flagged clips that reference moments outside the reel, drop near-duplicates, and surface anything off about the event-type sequence. Apply edits with `--drop-clips` and rerun (cached stages → seconds).
5. If the output is too long/short or misses obvious scenes, adjust `--threshold` and rerun.
6. Report clip count, total duration, `% of source`, and event-type breakdown from the pipeline output.
