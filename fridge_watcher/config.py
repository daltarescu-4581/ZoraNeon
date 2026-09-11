"""Environment-backed configuration for fridge-watcher.

Everything the service needs is read once at startup into an immutable
`Config`. Nothing else in the package touches `os.environ`, so tests and the
replay CLI can construct a `Config` by hand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

DEFAULT_MODEL: Final[str] = "claude-sonnet-4-6"

# Motion-mode guard rails. These are deliberately constants rather than env
# vars: they describe the physics of someone opening a fridge, not a
# deployment choice. A 0.5s blip is a cat walking past; a 90s window is
# somebody unpacking groceries and is better handled as several events.
MOTION_MIN_SECONDS: Final[float] = 1.0
MOTION_MAX_SECONDS: Final[float] = 45.0
# Frigate needs a moment to flush recording segments to disk before the
# recordings API will hand back a clip for the window we just watched.
MOTION_SETTLE_SECONDS: Final[float] = 3.0


class TriggerMode(str, Enum):
    """How we learn that something happened at the fridge."""

    EVENTS = "events"
    MOTION = "motion"


class ConfigError(RuntimeError):
    """Raised when the environment is missing something we cannot default."""


@dataclass(frozen=True)
class Config:
    frigate_url: str
    frigate_camera_name: str
    mqtt_host: str
    mqtt_port: int
    mqtt_user: str | None
    mqtt_password: str | None
    trigger_mode: TriggerMode
    zone_name: str
    frame_count: int
    confidence_threshold: float
    anthropic_api_key: str
    anthropic_model: str
    supabase_url: str
    supabase_service_key: str
    captures_dir: Path

    @property
    def events_topic(self) -> str:
        return "frigate/events"

    @property
    def motion_topic(self) -> str:
        return f"frigate/{self.frigate_camera_name}/motion"

    @property
    def mqtt_topic(self) -> str:
        if self.trigger_mode is TriggerMode.MOTION:
            return self.motion_topic
        return self.events_topic

    def capture_dir_for(self, event_id: str) -> Path:
        return self.captures_dir / _safe_segment(event_id)


def _safe_segment(value: str) -> str:
    """Frigate event IDs are filesystem-safe already, but motion IDs and
    replay stems come from user input, so keep them from escaping the dir."""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    return cleaned.strip("._") or "unknown"


def _require(name: str, value: str | None) -> str:
    if not value:
        raise ConfigError(f"{name} is required but not set (see .env.example)")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def load_config(*, require_supabase: bool = True, require_anthropic: bool = True) -> Config:
    """Read config from the environment (and a local .env if present).

    `replay` without `--write` never talks to Supabase, so it can relax the
    Supabase requirement instead of forcing a dummy URL into the environment.
    """
    load_dotenv()

    raw_mode = (os.getenv("TRIGGER_MODE") or TriggerMode.EVENTS.value).strip().lower()
    try:
        trigger_mode = TriggerMode(raw_mode)
    except ValueError as exc:
        allowed = ", ".join(m.value for m in TriggerMode)
        raise ConfigError(f"TRIGGER_MODE must be one of: {allowed} (got {raw_mode!r})") from exc

    frame_count = _env_int("FRAME_COUNT", 6)
    if frame_count < 2:
        # One frame cannot establish a trajectory, which is the whole point.
        raise ConfigError(f"FRAME_COUNT must be at least 2, got {frame_count}")

    threshold = _env_float("CONFIDENCE_THRESHOLD", 0.75)
    if not 0.0 <= threshold <= 1.0:
        raise ConfigError(f"CONFIDENCE_THRESHOLD must be between 0 and 1, got {threshold}")

    return Config(
        frigate_url=(os.getenv("FRIGATE_URL") or "http://localhost:5000").rstrip("/"),
        frigate_camera_name=os.getenv("FRIGATE_CAMERA_NAME") or "fridge",
        mqtt_host=os.getenv("MQTT_HOST") or "localhost",
        mqtt_port=_env_int("MQTT_PORT", 1883),
        mqtt_user=os.getenv("MQTT_USER") or None,
        mqtt_password=os.getenv("MQTT_PASSWORD") or None,
        trigger_mode=trigger_mode,
        zone_name=os.getenv("ZONE_NAME") or "fridge_door",
        frame_count=frame_count,
        confidence_threshold=threshold,
        anthropic_api_key=(
            _require("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
            if require_anthropic
            else os.getenv("ANTHROPIC_API_KEY", "")
        ),
        anthropic_model=os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL,
        supabase_url=(
            _require("SUPABASE_URL", os.getenv("SUPABASE_URL"))
            if require_supabase
            else os.getenv("SUPABASE_URL", "")
        ),
        supabase_service_key=(
            _require("SUPABASE_SERVICE_KEY", os.getenv("SUPABASE_SERVICE_KEY"))
            if require_supabase
            else os.getenv("SUPABASE_SERVICE_KEY", "")
        ),
        captures_dir=Path(os.getenv("CAPTURES_DIR") or "./captures").expanduser(),
    )
