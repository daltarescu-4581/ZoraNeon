"""End-to-end through the pipeline with a stubbed model, which is exactly what
`replay` does minus the Anthropic call."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fridge_watcher.config import Config
from fridge_watcher.frame_extractor import Frame
from fridge_watcher.pipeline import Pipeline, Trigger
from fridge_watcher.store import Status, WriteOutcome
from fridge_watcher.vision import Direction, VisionResult

from test_frame_extractor import write_clip


class StubAnalyzer:
    def __init__(self, result: VisionResult) -> None:
        self.result = result
        self.seen: list[list[Frame]] = []

    def analyze(self, frames):
        self.seen.append(frames)
        return self.result


class RecordingStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, VisionResult, str | None]] = []

    def record(self, event_id, result, frames_path):
        self.calls.append((event_id, result, frames_path))
        return WriteOutcome(
            status=Status.APPLIED, logged=True, duplicate=False, applied=True, quantity_after=2
        )


def out_result(confidence=0.88):
    return VisionResult(
        item="oat milk carton",
        category="dairy_alternative",
        quantity=1,
        direction=Direction.OUT,
        confidence=confidence,
        reasoning="near the door in Frame 2, by the sink in Frame 5",
    )


def test_replay_saves_frames_and_writes_when_a_store_is_present(config: Config, tmp_path: Path):
    clip = write_clip(tmp_path / "clip.mp4", frames=30)
    analyzer = StubAnalyzer(out_result())
    store = RecordingStore()
    pipeline = Pipeline(config, analyzer=analyzer, store=store)

    outcome = pipeline.handle(Trigger(capture_id="evt-1", local_clip=clip, source="replay"))

    capture_dir = config.captures_dir / "evt-1"
    assert len(list(capture_dir.glob("frame_*.jpg"))) == config.frame_count
    assert len(analyzer.seen[0]) == config.frame_count
    assert outcome.status is Status.APPLIED
    assert store.calls[0][0] == "evt-1"
    assert store.calls[0][2] == str(capture_dir)


def test_capture_directory_is_self_describing(config: Config, tmp_path: Path):
    clip = write_clip(tmp_path / "clip.mp4", frames=30)
    pipeline = Pipeline(config, analyzer=StubAnalyzer(out_result()), store=None)

    pipeline.handle(Trigger(capture_id="evt-2", local_clip=clip, source="replay"))

    metadata = json.loads((config.captures_dir / "evt-2" / "result.json").read_text())
    assert metadata["capture_id"] == "evt-2"
    assert metadata["result"]["direction"] == "OUT"
    assert metadata["result"]["item"] == "oat milk carton"
    assert metadata["model"] == config.anthropic_model


def test_dry_run_never_writes(config: Config, tmp_path: Path):
    clip = write_clip(tmp_path / "clip.mp4", frames=30)
    pipeline = Pipeline(config, analyzer=StubAnalyzer(out_result()), store=None)

    outcome = pipeline.handle(Trigger(capture_id="evt-3", local_clip=clip, source="replay"))

    assert outcome.write is None
    assert outcome.status is Status.APPLIED  # what it *would* have been
    # Frames are still saved, so a dry run still builds the regression set.
    assert (config.captures_dir / "evt-3" / "frame_01.jpg").exists()


def test_low_confidence_is_routed_to_review(config: Config, tmp_path: Path):
    clip = write_clip(tmp_path / "clip.mp4", frames=30)
    pipeline = Pipeline(config, analyzer=StubAnalyzer(out_result(confidence=0.4)), store=None)

    outcome = pipeline.handle(Trigger(capture_id="evt-4", local_clip=clip, source="replay"))

    assert outcome.status is Status.PENDING_REVIEW


def test_no_item_skips_the_store_entirely(config: Config, tmp_path: Path):
    clip = write_clip(tmp_path / "clip.mp4", frames=30)
    store = RecordingStore()
    result = VisionResult(None, None, 0, Direction.NO_ITEM, 0.0, "nothing in frame")
    pipeline = Pipeline(config, analyzer=StubAnalyzer(result), store=store)

    outcome = pipeline.handle(Trigger(capture_id="evt-5", local_clip=clip, source="replay"))

    assert store.calls == []
    assert outcome.status is Status.REJECTED
    # Frames are still on disk: a NO_ITEM we disagree with is worth labelling.
    assert (config.captures_dir / "evt-5" / "frame_01.jpg").exists()


def test_remote_trigger_without_a_frigate_client_is_an_error(config: Config):
    pipeline = Pipeline(config, analyzer=StubAnalyzer(out_result()), store=None)
    with pytest.raises(RuntimeError, match="Frigate client"):
        pipeline.handle(Trigger(capture_id="evt-6", event_id="evt-6"))


def test_capture_ids_cannot_escape_the_captures_dir(config: Config, tmp_path: Path):
    clip = write_clip(tmp_path / "clip.mp4", frames=10)
    pipeline = Pipeline(config, analyzer=StubAnalyzer(out_result()), store=None)

    outcome = pipeline.handle(
        Trigger(capture_id="../../etc/passwd", local_clip=clip, source="replay")
    )

    assert config.captures_dir in outcome.frames_path.parents
