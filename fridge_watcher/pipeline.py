"""Clip -> frames -> vision -> Supabase, in one place.

The live listener, `replay` and `replay-all` all funnel through here so a
replayed capture exercises exactly the code path a real event does.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .frame_extractor import Frame, extract_frames, save_frames
from .frigate import FrigateClient
from .store import InventoryStore, Status, WriteOutcome, status_for
from .vision import Direction, VisionAnalyzer, VisionResult

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Trigger:
    """Something happened that we should look at.

    Exactly one of `event_id` (events mode) or `window` (motion mode) drives
    the clip fetch; `local_clip` short-circuits both for replay.
    """

    capture_id: str
    event_id: str | None = None
    window: tuple[float, float] | None = None
    local_clip: Path | None = None
    source: str = "events"
    # Seconds to wait before touching Frigate, so it can finish writing the
    # recording segments we are about to ask for.
    delay_s: float = 0.0


@dataclass(frozen=True)
class PipelineOutcome:
    capture_id: str
    result: VisionResult
    status: Status
    frames_path: Path | None
    write: WriteOutcome | None

    def summary(self) -> dict[str, object]:
        return {
            "capture_id": self.capture_id,
            "item": self.result.item,
            "category": self.result.category,
            "quantity": self.result.quantity,
            "direction": self.result.direction.value,
            "confidence": round(self.result.confidence, 4),
            "reasoning": self.result.reasoning,
            "status": self.status.value,
            "frames_path": str(self.frames_path) if self.frames_path else None,
            "parse_error": self.result.parse_error,
            "written": bool(self.write and self.write.logged),
            "duplicate": bool(self.write and self.write.duplicate),
        }


class Pipeline:
    def __init__(
        self,
        config: Config,
        analyzer: VisionAnalyzer,
        store: InventoryStore | None = None,
        frigate: FrigateClient | None = None,
    ) -> None:
        self._config = config
        self._analyzer = analyzer
        self._store = store  # None => dry run (replay without --write)
        self._frigate = frigate

    # -- entry points -------------------------------------------------------

    def handle(self, trigger: Trigger) -> PipelineOutcome:
        started = time.monotonic()
        clip_path, cleanup = self._resolve_clip(trigger)
        try:
            frames = extract_frames(clip_path, self._config.frame_count)
            capture_dir = self._config.capture_dir_for(trigger.capture_id)
            save_frames(frames, capture_dir)
        finally:
            if cleanup and clip_path.exists():
                clip_path.unlink(missing_ok=True)

        outcome = self.analyze_frames(frames, trigger.capture_id, capture_dir)
        self._write_capture_metadata(capture_dir, trigger, outcome)
        log.info(
            "event_processed",
            extra={
                "capture_id": trigger.capture_id,
                "source": trigger.source,
                "status": outcome.status.value,
                "elapsed_s": round(time.monotonic() - started, 2),
            },
        )
        return outcome

    def analyze_frames(
        self, frames: list[Frame], capture_id: str, frames_path: Path | None
    ) -> PipelineOutcome:
        """Run the vision call and (optionally) the write for a frame set."""
        result = self._analyzer.analyze(frames)
        status = status_for(result, self._config.confidence_threshold)

        if result.parsed and result.direction is Direction.NO_ITEM:
            # Nothing crossed the frame: log it here and write nothing.
            log.info("no_item", extra={"capture_id": capture_id, "reasoning": result.reasoning})
            return PipelineOutcome(capture_id, result, Status.REJECTED, frames_path, None)

        write: WriteOutcome | None = None
        if self._store is not None:
            write = self._store.record(
                capture_id, result, str(frames_path) if frames_path else None
            )
            status = write.status
        else:
            log.info(
                "dry_run",
                extra={"capture_id": capture_id, "would_be_status": status.value},
            )
        return PipelineOutcome(capture_id, result, status, frames_path, write)

    # -- helpers ------------------------------------------------------------

    def _resolve_clip(self, trigger: Trigger) -> tuple[Path, bool]:
        """Return (clip path, should_delete_after_use)."""
        if trigger.local_clip is not None:
            return trigger.local_clip, False

        if self._frigate is None:
            raise RuntimeError("a Frigate client is required to fetch remote clips")

        temp_dir = Path(tempfile.gettempdir())
        destination = temp_dir / f"fridge-watcher-{trigger.capture_id}.mp4"

        if trigger.event_id is not None:
            return self._frigate.download_event_clip(trigger.event_id, destination), True
        if trigger.window is not None:
            start, end = trigger.window
            return self._frigate.download_recording(start, end, destination), True
        raise ValueError(f"trigger {trigger.capture_id} has no clip source")

    def _write_capture_metadata(
        self, capture_dir: Path, trigger: Trigger, outcome: PipelineOutcome
    ) -> None:
        """Drop result.json next to the frames so captures are self-describing
        and `replay-all` has something to diff prompt changes against."""
        payload = {
            "capture_id": trigger.capture_id,
            "source": trigger.source,
            "event_id": trigger.event_id,
            "window": list(trigger.window) if trigger.window else None,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "model": self._config.anthropic_model,
            "result": outcome.result.to_dict(),
            "status": outcome.status.value,
        }
        try:
            (capture_dir / "result.json").write_text(json.dumps(payload, indent=2))
        except OSError as exc:
            log.warning(
                "capture_metadata_write_failed",
                extra={"capture_dir": str(capture_dir), "error": str(exc)},
            )
