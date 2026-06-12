# f1-highlights

Turn a 1–3 hour F1 broadcast into a driver- or team-focused highlight reel.

It's a six-stage pipeline that fuses Whisper commentary transcription with OCR on broadcast graphics and livery-colour detection, then compiles a story-flow cut that snaps to sentence boundaries so commentary never gets clipped mid-word. Works on practice, qualifying, sprint, and race broadcasts from the 2015–2026 seasons.

This repo is a [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills) — Claude invokes it after confirming year, session, and target with you — but `pipeline.py` is a plain CLI and runs fine standalone.

## How it works

```
video.mp4
  │
  ├─ 1. transcribe.py      audio → faster-whisper → word-level transcript
  ├─ 2. analyze_text.py    alias hits in commentary → sentence-scoped mentions
  ├─ 3. analyze_video.py   sample frames → easyocr → name hits on graphics
  ├─ 3½ analyze_livery.py  team-colour coverage + colour-gated number hits
  ├─ 4. build_timeline.py  fuse signals → per-second relevance score
  ├─ 5. build_clips.py     threshold + merge + pad + snap-to-sentence
  └─ 6. compile.py         ffmpeg cut + concat (stream-copy or crossfade)
       ↓
     highlights.mp4
```

Each stage writes to `<output>.work/` and is resumable — rerunning with a different threshold only redoes stages 4–6 (seconds), skipping transcription and OCR.

| signal                    | weight  | when it fires                                                 |
|---------------------------|---------|---------------------------------------------------------------|
| transcript driver mention | 2.0     | commentator said the driver's name                            |
| transcript team mention   | 1.0     | commentator said the team's name                              |
| OCR driver/team name      | 3.0/1.5 | graphic with name overlays the frame                          |
| livery colour coverage    | 0.5–2.0 | team's livery dominates a large area of the frame             |
| colour-gated car number   | 2.5     | driver's race number detected AND surrounded by team colour   |

## Setup

```bash
git clone https://github.com/summeratjuly/f1-highlights.git ~/.claude/skills/f1-highlights
cd ~/.claude/skills/f1-highlights
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# ffmpeg must be installed on the system
brew install ffmpeg
```

First run downloads Whisper (`small.en`, ~150MB) and easyocr detection models.

## Usage

Driver reel:

```bash
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

Smoother transitions (re-encode with crossfades, ~2–3× slower):

```bash
python scripts/pipeline.py --video q.mp4 --year 2024 --session qualifying --driver NOR --output out.mp4 --smooth
```

Useful flags:

- `--threshold 0.5` — more inclusive (longer reel)
- `--threshold 2.0` — stricter (driver-centric only)
- `--pre-roll 4 --post-roll 5` — more context around each moment
- `--merge-gap 15` — fewer, longer, smoother-flowing clips
- `--text-only` — skip OCR and livery (3–5× faster, misses no-mention shots)
- `--no-livery` — keep OCR name matching, skip colour/number
- `--fps 0.5` — sample frames every 2s (faster, may miss brief on-screens)
- `--drop-clips A,B,C` — drop clips from a previous run by index (uses cache → seconds)
- `--no-bridges` / `--bridge-min-gap` / `--bridge-duration` — control inter-clip bridge cards

## Clarity rules — baked into code

Each region of relevance is classified by *what kind of moment it is* and padded so the clip lands as a self-contained micro-story:

| event             | pre-roll | post-roll | what it captures                              |
|-------------------|----------|-----------|-----------------------------------------------|
| incident          | 3.0s     | 8.0s      | cause + impact + reaction (replay glued in)   |
| finish            | 3.0s     | 8.0s      | last lap + checkered + first reaction         |
| start             | 5.0s     | 6.0s      | lights, launch, first corner                  |
| overtake          | 4.0s     | 6.0s      | the chase + the move + the result             |
| pit               | 3.0s     | 5.0s      | in-lap context + tire change + out-lap        |
| qualifying_lap    | 2.0s     | 4.0s      | sector commentary + final time                |
| mention (default) | 1.5s     | 2.5s      | passive name-check, low action                |

Other guarantees enforced in code:

- **Replay binding** — *"let's see that again"*, *"from a different angle"*, "REPLAY" graphic → glued to the preceding moment so the replay isn't an orphan flashback.
- **Sentence-end integrity** — clips never end on a dangling conjunction (`...and`, `...but`, `...because`) or open with a pronoun-only sentence. Cuts extend to the next full sentence (capped 4s).
- **Race bookends** — in race sessions, the first 9 minutes (start/grid) and last 4 minutes (finish/podium) drop threshold to 0.5 so target activity in those windows is always covered.
- **Bridge cards** — for gaps > 30s between adjacent clips, a 10-second 3-panel "triptych" card summarises up to 3 narrative events from the gap. Panels are keyframe + ALL-CAPS title + 12–20-word caption. Captions are written by Claude (via the Agent SDK using your existing Claude Code login — no separate API key) so keyframe and caption stay anchored to the same moment. Falls back to a deterministic captioner if the SDK call fails.

## Session-aware defaults

Set automatically from `--session` unless an explicit flag overrides:

| session    | pre_roll | post_roll | merge_gap | min_len |
|------------|----------|-----------|-----------|---------|
| practice   | 2.0      | 3.0       | 8.0       | 6.0     |
| qualifying | 1.5      | 2.5       | 5.0       | 5.0     |
| sprint     | 2.0      | 3.0       | 6.0       | 6.0     |
| race       | 2.0      | 3.0       | 8.0       | 6.0     |

Qualifying has shorter runs per driver, so tighter timing avoids bleeding into other drivers' laps. A race has long green-flag stretches where merge-gap=8s keeps narrative continuity.

## Grid data — one file per season

`data/grids/{year}.json` for each 2015–2026 season:

- `drivers[]` — `id`, `name`, `team`, `number`, `aliases` (lowercase substrings to match). The 3-letter `id` is what you pass to `--driver`. `number` drives car-number matching.
- `teams[]` — `id`, `name`, `aliases`. The `id` is what you pass to `--team`.
- `_notes` — mid-season driver swaps, sub drivers, notable changes for that year.

`data/team_colors.json` holds the livery colour palette for HSV matching, with `overrides.<year>.<team_id>` for year-specific liveries (e.g. `2020.MER = "#000000"` for the black Mercedes, `2015/2017.MCL` for the silver/transitional McLaren).

Alias rules applied across seasons:

- Last name and full name are always included.
- First name alone is included **only** when unambiguous that season. E.g., "Nico" is omitted from 2015–16 because both Rosberg and Hulkenberg raced; included from 2017 on (Hulkenberg only).
- Whisper-friendly spelling variants (`räikkönen`, accented Pérez forms, `hülkenberg`) are included to survive transcription choices.

To add a new season, drop a file into `data/grids/` matching the same shape.

## Tuning, in order of likely usefulness

1. **`--threshold`** — most impactful. Start at 1.0, raise if the reel is too long/noisy, lower if scenes are missing.
2. **`--merge-gap`** — larger values make smoother story flow by keeping in-between footage.
3. **`--pre-roll` / `--post-roll`** — add context around each relevant moment.
4. **`--fps`** — OCR sampling. Drop to 0.5 for 2-hour races if full-fps is slow.

## Known limits

- **Wide racing shots with no graphic** — OCR catches onboards, replays, timing tower, team radio, interviews. The livery signal catches the *team* in wide shots without a graphic. Teammate separation in those shots needs the car number to be OCR-readable (best in pit/grid/formation, unreliable during racing motion blur).
- **Colour gating on numbers is strict by design** — "P4" next to the lap counter won't mis-fire as Norris, because the team-colour region check only passes when a papaya-orange car is actually around the number. Downside: some clean nose-number shots may still fail the gate.
- **Whisper mis-hears names** in overlapping commentary. Fix: add misheard spellings to that year's `aliases`.
- **Teammate confusion** — if you want "just NOR, not PIA", use `--driver NOR`, not `--team MCL`.
- **Stream-copy cuts snap to GOP boundaries** so clip start times may shift up to ~2s. Use `--smooth` for precision at the cost of encode time.
- **Long races cost time** — ~2hr race takes roughly 15–25min to transcribe + 10–15min to OCR at 1fps on an M-series Mac. Use `--text-only` for a quick first pass.
- **Historical grid accuracy** — mid-season substitutes are included where I was confident; for a specific race with an unusual lineup, check and edit the year file before running.
- **2026 grid is tentative** — Audi takeover and Cadillac entry are still in flux; verify against the official entry list for that round.
- **Team-colour collisions** — Red Bull and RB/AlphaTauri both run blue. Number gating mitigates at driver-level; for `--team RBR` the colour hits for AT/RB cars are discarded as off-team.

## Output review

After the pipeline completes it prints a one-line-per-clip review (also written to `<workdir>/clips.json` with full transcript windows). Each line shows index, start time, duration, event type, flags (e.g. `ORPHAN`), and a transcript excerpt.

When the skill runs under Claude, it applies a model-layer review on top of the keyword rules — dropping `ORPHAN`-flagged clips whose referenced moment isn't in the reel, merging near-duplicate mentions, and flagging odd event-type sequences (e.g. a race reel with no `start` or `finish`). For standalone runs, the same flags are right there in the review output to act on with `--drop-clips`.

## Repo layout

```
scripts/        pipeline stages + helpers
data/grids/     {year}.json with drivers/teams/aliases (2015–2026)
data/team_colors.json    livery HSV palettes per team / year override
SKILL.md        instructions Claude follows when invoking the skill
```

See [SKILL.md](SKILL.md) for the full Claude-facing flow, including the archive-to-NAS hand-off and the model-layer review checklist.
