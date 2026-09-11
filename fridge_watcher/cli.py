"""Command line interface.

  python -m fridge_watcher run
  python -m fridge_watcher replay path/to/clip.mp4 [--write]
  python -m fridge_watcher replay-all ./captures --expect expectations.json
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from types import FrameType

from .config import Config, ConfigError, TriggerMode, load_config
from .evaluate import Expectation, Report, Score, format_report, load_expectations
from .frame_extractor import FrameExtractionError, load_frames
from .frigate import FrigateClient
from .logging_setup import configure_logging
from .mqtt_listener import FrigateListener
from .pipeline import Pipeline, PipelineOutcome, Trigger
from .store import SupabaseStore, status_for
from .vision import VisionAnalyzer

log = logging.getLogger("fridge_watcher")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fridge_watcher",
        description="Turn Frigate camera events into FridgeFriend inventory updates.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Listen to Frigate and process events live.")
    run.add_argument(
        "--trigger-mode",
        choices=[m.value for m in TriggerMode],
        help="Override TRIGGER_MODE for this run.",
    )

    replay = sub.add_parser(
        "replay", help="Run the full pipeline against a local video file."
    )
    replay.add_argument("clip", type=Path, help="Path to an .mp4 clip.")
    replay.add_argument(
        "--write",
        action="store_true",
        help="Actually write the result to Supabase (default: dry run).",
    )
    replay.add_argument(
        "--capture-id",
        help="ID to file the capture under (default: the video file's stem).",
    )
    replay.add_argument(
        "--frames", type=int, help="Override FRAME_COUNT for this run."
    )
    replay.add_argument("--json", action="store_true", help="Print JSON only.")

    replay_all = sub.add_parser(
        "replay-all",
        help="Re-run every saved capture and score it against hand labels.",
    )
    replay_all.add_argument(
        "captures", type=Path, nargs="?", help="Captures directory (default: CAPTURES_DIR)."
    )
    replay_all.add_argument(
        "--expect",
        type=Path,
        required=True,
        help="JSON file of hand-labelled expectations.",
    )
    replay_all.add_argument("--json", action="store_true", help="Print JSON only.")
    replay_all.add_argument(
        "--fail-under",
        type=float,
        metavar="ACCURACY",
        help="Exit non-zero if overall accuracy is below this (e.g. 0.9), so a "
             "prompt change can be gated in CI.",
    )
    replay_all.add_argument(
        "--only",
        action="append",
        default=[],
        help="Only score this capture id (repeatable).",
    )
    return parser


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config()
    if args.trigger_mode:
        config = replace_trigger_mode(config, TriggerMode(args.trigger_mode))

    analyzer = VisionAnalyzer(config)
    store = SupabaseStore(config)
    frigate = FrigateClient(config)
    pipeline = Pipeline(config, analyzer=analyzer, store=store, frigate=frigate)

    listener = FrigateListener(config, handler=pipeline.handle)

    def shutdown(signum: int, _frame: FrameType | None) -> None:
        log.info("shutting_down", extra={"signal": signum})
        listener.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info(
        "starting",
        extra={
            "trigger_mode": config.trigger_mode.value,
            "camera": config.frigate_camera_name,
            "zone": config.zone_name,
            "frame_count": config.frame_count,
            "confidence_threshold": config.confidence_threshold,
            "captures_dir": str(config.captures_dir),
            "model": config.anthropic_model,
        },
    )
    try:
        listener.run_forever()
    finally:
        frigate.close()
    return 0


def replace_trigger_mode(config: Config, mode: TriggerMode) -> Config:
    from dataclasses import replace

    return replace(config, trigger_mode=mode)


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def cmd_replay(args: argparse.Namespace) -> int:
    clip: Path = args.clip
    if not clip.exists():
        print(f"no such clip: {clip}", file=sys.stderr)
        return 2

    config = load_config(require_supabase=args.write)
    if args.frames:
        from dataclasses import replace

        config = replace(config, frame_count=args.frames)

    analyzer = VisionAnalyzer(config)
    store = SupabaseStore(config) if args.write else None
    pipeline = Pipeline(config, analyzer=analyzer, store=store)

    capture_id = args.capture_id or clip.stem
    outcome = pipeline.handle(
        Trigger(capture_id=capture_id, local_clip=clip, source="replay")
    )
    print_outcome(outcome, as_json=args.json, wrote=args.write)
    return 0 if outcome.result.parsed else 1


def print_outcome(outcome: PipelineOutcome, *, as_json: bool, wrote: bool) -> None:
    summary = outcome.summary()
    if as_json:
        print(json.dumps(summary, indent=2))
        return

    result = outcome.result
    print()
    print(f"capture      {outcome.capture_id}")
    print(f"item         {result.item or '-'}  ({result.category or '-'})")
    print(f"quantity     {result.quantity}")
    print(f"direction    {result.direction.value}")
    print(f"confidence   {result.confidence:.2f}")
    print(f"reasoning    {result.reasoning}")
    print(f"status       {outcome.status.value}")
    print(f"frames       {outcome.frames_path or '-'}")
    if result.parse_error:
        print(f"parse error  {result.parse_error}")
        print(f"raw response {result.raw_response[:500]}")
    print(f"supabase     {'written' if wrote else 'dry run (pass --write to persist)'}")
    print()


# ---------------------------------------------------------------------------
# replay-all
# ---------------------------------------------------------------------------


def cmd_replay_all(args: argparse.Namespace) -> int:
    config = load_config(require_supabase=False)
    captures_dir: Path = args.captures or config.captures_dir
    if not captures_dir.is_dir():
        print(f"no such captures directory: {captures_dir}", file=sys.stderr)
        return 2

    try:
        expectations = load_expectations(args.expect)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"could not read expectations: {exc}", file=sys.stderr)
        return 2

    analyzer = VisionAnalyzer(config)
    pipeline = Pipeline(config, analyzer=analyzer, store=None)

    available = {
        path.name: path
        for path in sorted(captures_dir.iterdir())
        if path.is_dir() and any(path.glob("frame_*.jpg"))
    }
    wanted = set(args.only) if args.only else None

    scores: list[Score] = []
    missing: list[str] = []
    for capture_id, expectation in expectations.items():
        if wanted is not None and capture_id not in wanted:
            continue
        directory = available.get(capture_id)
        if directory is None:
            missing.append(capture_id)
            continue
        score = score_capture(pipeline, config, capture_id, directory, expectation)
        if score is not None:
            scores.append(score)

    unlabelled = sorted(set(available) - set(expectations))
    report = Report(scores=scores, missing=missing, unlabelled=unlabelled)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))

    if not report.total:
        print("nothing scored: no capture directory matched the expectations file",
              file=sys.stderr)
        return 2
    if args.fail_under is not None and report.overall_accuracy < args.fail_under:
        print(
            f"overall accuracy {report.overall_accuracy:.1%} is below "
            f"--fail-under {args.fail_under:.1%}",
            file=sys.stderr,
        )
        return 1
    return 0


def score_capture(
    pipeline: Pipeline,
    config: Config,
    capture_id: str,
    directory: Path,
    expectation: Expectation,
) -> Score | None:
    try:
        frames = load_frames(directory)
    except FrameExtractionError as exc:
        log.error("capture_unreadable", extra={"capture_id": capture_id, "error": str(exc)})
        return None

    outcome = pipeline.analyze_frames(frames, capture_id, directory)
    status = status_for(outcome.result, config.confidence_threshold)
    return Score(
        capture_id=capture_id,
        expected=expectation,
        result=outcome.result,
        applied=status.value == "applied" and outcome.result.has_item,
    )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    handlers = {
        "run": cmd_run,
        "replay": cmd_replay,
        "replay-all": cmd_replay_all,
    }
    try:
        return handlers[args.command](args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
