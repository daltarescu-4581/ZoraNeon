"""Frigate HTTP API client (clip + recording downloads)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

from .config import Config

log = logging.getLogger(__name__)

# Frigate finalises the mp4 a moment after it publishes the `end` event, so the
# first GET often 404s or returns a zero-byte body. Back off and try again.
CLIP_ATTEMPTS = 5
CLIP_BASE_DELAY = 1.0
CLIP_TIMEOUT = 60.0
MIN_CLIP_BYTES = 1024


class ClipUnavailable(RuntimeError):
    """Frigate never produced a usable clip for this event/window."""


class FrigateClient:
    def __init__(self, config: Config, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=CLIP_TIMEOUT, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FrigateClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def download_event_clip(self, event_id: str, destination: Path) -> Path:
        url = f"{self._config.frigate_url}/api/events/{event_id}/clip.mp4"
        return self._download_with_retry(url, destination, context={"event_id": event_id})

    def download_recording(self, start_ts: float, end_ts: float, destination: Path) -> Path:
        """Motion mode: export the recording covering the motion window.

        Frigate has no event to hang a clip off in motion mode, so we ask the
        recordings API for the exact ON..OFF span instead.
        """
        camera = self._config.frigate_camera_name
        url = (
            f"{self._config.frigate_url}/api/{camera}"
            f"/start/{start_ts:.3f}/end/{end_ts:.3f}/clip.mp4"
        )
        return self._download_with_retry(
            url, destination, context={"camera": camera, "start": start_ts, "end": end_ts}
        )

    def _download_with_retry(
        self, url: str, destination: Path, context: dict[str, object]
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        last_error = "no attempt made"

        for attempt in range(1, CLIP_ATTEMPTS + 1):
            try:
                response = self._client.get(url)
                if response.status_code == 200 and len(response.content) >= MIN_CLIP_BYTES:
                    destination.write_bytes(response.content)
                    log.info(
                        "clip_downloaded",
                        extra={"url": url, "bytes": len(response.content), "attempt": attempt},
                    )
                    return destination
                last_error = (
                    f"HTTP {response.status_code}, {len(response.content)} bytes"
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < CLIP_ATTEMPTS:
                delay = CLIP_BASE_DELAY * (2 ** (attempt - 1))
                log.warning(
                    "clip_retry",
                    extra={"url": url, "attempt": attempt, "delay": delay, "error": last_error, **context},
                )
                time.sleep(delay)

        raise ClipUnavailable(f"could not download {url} after {CLIP_ATTEMPTS} attempts: {last_error}")
