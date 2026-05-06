"""Move a finished pipeline run (source recording + highlight reel + work
dir) to the NAS Recording archive, renaming everything to follow the
canonical convention defined in `/Volumes/Media/Recording/README.md`.

Destination layout (per race):
  <NAS_BASE>/<YEAR>/r<NN>_<RACE>_<YEAR>.mov
  <NAS_BASE>/<YEAR>/r<NN>_<RACE>_<YEAR>_<TARGET>.mp4
  <NAS_BASE>/<YEAR>/r<NN>_<RACE>_<YEAR>_<TARGET>.work/

Round number must be the official F1 calendar round (zero-padded). Race
name must be the lowercase canonical token (e.g. `canada`, `silverstone`,
`hungarian`). Target is the driver/team focus code (default `ver`).

The script:
  - validates the NAS mount + that the year README exists
  - refuses to overwrite an existing NAS path
  - prints the proposed moves and (without --yes) asks for confirmation
  - moves with shutil.move (cross-volume → copy + delete on the source side)

CLI:
  python move_to_nas.py \
    --source ~/Movies/Recording/2016_canada.mov \
    --highlight ~/Movies/Recording/canada_2016_ver.mp4 \
    --workdir ~/Movies/Recording/canada_2016_ver.work \
    --year 2016 --race canada --round 7
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


_NAS_BASE_DEFAULT = Path("/Volumes/Media/Recording")
_RACE_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]*$")


_SESSION_SUFFIX = {
    "race":       "",          # default — no suffix, e.g. r07_canada_2016.mov
    "qualifying": "_quali",    # r07_canada_quali_2016.mov
    "sprint":     "_sprint",
    "practice":   "_practice", # less common; usually we'd use fp1/fp2/fp3
    "fp1":        "_fp1",
    "fp2":        "_fp2",
    "fp3":        "_fp3",
}


def _build_dest_paths(nas_base: Path, year: int, race: str,
                      round_no: int, target: str,
                      session: str = "race") -> tuple[Path, Path, Path]:
    if not _RACE_TOKEN_RE.match(race):
        raise SystemExit(
            f"[move-to-nas] race name must be lowercase ascii / digits / underscore: '{race}'"
        )
    if not _RACE_TOKEN_RE.match(target):
        raise SystemExit(
            f"[move-to-nas] target code must be lowercase ascii / digits / underscore: '{target}'"
        )
    if not (1 <= round_no <= 30):
        raise SystemExit(f"[move-to-nas] round number {round_no} out of range (1-30)")
    if session not in _SESSION_SUFFIX:
        raise SystemExit(
            f"[move-to-nas] unknown session '{session}'. "
            f"Known: {', '.join(_SESSION_SUFFIX)}"
        )

    suffix = _SESSION_SUFFIX[session]
    stem = f"r{round_no:02d}_{race}{suffix}_{year}"
    year_dir = nas_base / str(year)
    return (
        year_dir / f"{stem}.mov",
        year_dir / f"{stem}_{target}.mp4",
        year_dir / f"{stem}_{target}.work",
    )


def _refuse_overwrite(p: Path) -> None:
    if p.exists():
        raise SystemExit(
            f"[move-to-nas] refusing to overwrite existing NAS path: {p}"
        )


def _confirm(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def archive_run(*, source: Path, highlight: Path, workdir: Path,
                year: int, race: str, round_no: int,
                target: str = "ver",
                session: str = "race",
                nas_base: Path = _NAS_BASE_DEFAULT,
                interactive: bool = True,
                dry_run: bool = False) -> None:
    """Programmatic entry point — same checks/move logic as the CLI.

    Used by pipeline.py when `--archive-to-nas` is set so the move can run
    immediately after compile without a separate invocation. Pass
    interactive=False to skip the y/N prompt (the caller has already
    confirmed upfront).
    """
    if not nas_base.exists():
        raise SystemExit(
            f"[move-to-nas] NAS base not mounted: {nas_base}. Mount the share and retry."
        )
    readme = nas_base / "README.md"
    if not readme.exists():
        raise SystemExit(
            f"[move-to-nas] no README.md at {readme} — refusing to write blindly."
        )
    for p in (source, highlight, workdir):
        if not p.exists():
            raise SystemExit(f"[move-to-nas] not found: {p}")
    if not workdir.is_dir():
        raise SystemExit(f"[move-to-nas] workdir is not a directory: {workdir}")

    new_source, new_highlight, new_workdir = _build_dest_paths(
        nas_base, year, race.lower(), round_no, target.lower(),
        session=session.lower(),
    )

    print("[move-to-nas] planned moves:")
    print(f"  source     {source}")
    print(f"             →  {new_source}")
    print(f"  highlight  {highlight}")
    print(f"             →  {new_highlight}")
    print(f"  workdir    {workdir}")
    print(f"             →  {new_workdir}")

    if dry_run:
        print("[move-to-nas] dry-run; nothing moved.")
        return

    for p in (new_source, new_highlight, new_workdir):
        _refuse_overwrite(p)

    if interactive and not _confirm("[move-to-nas] proceed with the moves above?"):
        print("[move-to-nas] aborted by user.")
        return

    new_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(new_source))
    print(f"[move-to-nas] ✓ source     → {new_source}")
    shutil.move(str(highlight), str(new_highlight))
    print(f"[move-to-nas] ✓ highlight  → {new_highlight}")
    _move_dir_safe(workdir, new_workdir)
    print(f"[move-to-nas] ✓ workdir    → {new_workdir}")
    print("[move-to-nas] done.")


def _move_dir_safe(src: Path, dst: Path) -> None:
    """Cross-device dir move that survives macOS Spotlight adding
    `.DS_Store` mid-operation. shutil.move's internal `rmtree` raises
    "Directory not empty" if any file appears in the source between
    `copytree` and the final `os.rmdir` — easy to hit when the source
    is a Finder-visible folder during a multi-second copy. We do the
    copy ourselves and then force-clean the source via `rm -rf` which
    tolerates such transient files.
    """
    import subprocess
    if dst.exists():
        raise FileExistsError(f"destination exists: {dst}")
    shutil.copytree(src, dst, symlinks=True)
    subprocess.run(["rm", "-rf", str(src)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source", type=Path, required=True,
                    help="Original race recording (.mov / .mp4) to archive.")
    ap.add_argument("--highlight", type=Path, required=True,
                    help="Generated highlight reel (.mp4).")
    ap.add_argument("--workdir", type=Path, required=True,
                    help="Pipeline work directory containing transcript / ocr / clips / bridges JSON.")
    ap.add_argument("--year", type=int, required=True,
                    help="Season year (e.g. 2016).")
    ap.add_argument("--race", required=True,
                    help="Lowercase canonical race name (e.g. canada, silverstone, hungarian).")
    ap.add_argument("--round", dest="round_no", type=int, required=True,
                    help="F1 calendar round number (1-based, zero-padded in the resulting filename).")
    ap.add_argument("--target", default="ver",
                    help="Driver/team focus code (default 'ver'; also: 'ham', 'rbr', etc.).")
    ap.add_argument("--nas-base", type=Path, default=_NAS_BASE_DEFAULT,
                    help=f"NAS Recording root (default {_NAS_BASE_DEFAULT}).")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the interactive confirmation prompt.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned moves and exit without touching anything.")
    args = ap.parse_args()

    archive_run(
        source=args.source, highlight=args.highlight, workdir=args.workdir,
        year=args.year, race=args.race, round_no=args.round_no,
        target=args.target, nas_base=args.nas_base,
        interactive=not args.yes, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
