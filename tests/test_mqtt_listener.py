"""Event filtering and motion windowing. No broker involved."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from fridge_watcher.config import MOTION_MAX_SECONDS, Config, TriggerMode
from fridge_watcher.mqtt_listener import FrigateListener, _RecentIds
from fridge_watcher.pipeline import Trigger


class Message:
    def __init__(self, payload: Any, topic: str = "frigate/events") -> None:
        self.topic = topic
        self.payload = (
            payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        )


def end_event(**after: Any) -> dict[str, Any]:
    payload = {
        "id": "1727213456.123456-abc12d",
        "camera": "fridge",
        "label": "person",
        "entered_zones": ["fridge_door"],
    }
    payload.update(after)
    return {"type": "end", "before": {}, "after": payload}


@pytest.fixture
def listener(config: Config) -> FrigateListener:
    return FrigateListener(config, handler=lambda _t: None)


def test_accepts_end_event_in_zone(listener: FrigateListener):
    trigger = listener.trigger_from_event(end_event())
    assert trigger is not None
    assert trigger.event_id == "1727213456.123456-abc12d"
    assert trigger.capture_id == "1727213456.123456-abc12d"
    assert trigger.source == "events"


@pytest.mark.parametrize("event_type", ["start", "update"])
def test_ignores_non_end_events(listener: FrigateListener, event_type: str):
    payload = end_event()
    payload["type"] = event_type
    assert listener.trigger_from_event(payload) is None


def test_ignores_events_outside_the_zone(listener: FrigateListener):
    assert listener.trigger_from_event(end_event(entered_zones=["counter"])) is None
    assert listener.trigger_from_event(end_event(entered_zones=[])) is None


def test_ignores_other_cameras(listener: FrigateListener):
    assert listener.trigger_from_event(end_event(camera="driveway")) is None


def test_ignores_events_without_an_id(listener: FrigateListener):
    assert listener.trigger_from_event(end_event(id=None)) is None


def test_zone_name_is_configurable(config: Config):
    listener = FrigateListener(replace(config, zone_name="cold_zone"), handler=lambda _t: None)
    assert listener.trigger_from_event(end_event()) is None
    assert listener.trigger_from_event(end_event(entered_zones=["cold_zone"])) is not None


def test_malformed_payload_does_not_raise(listener: FrigateListener):
    listener._on_message(None, None, Message("{not json"))  # must not raise


def test_recent_ids_dedupe():
    recent = _RecentIds(size=3)
    assert recent.add_if_new("a") is True
    assert recent.add_if_new("a") is False
    for key in "bcd":
        recent.add_if_new(key)
    assert recent.add_if_new("a") is True  # evicted, so new again


def test_redelivered_event_is_only_queued_once(config: Config):
    seen: list[Trigger] = []
    listener = FrigateListener(config, handler=seen.append)
    message = Message(end_event())

    listener._on_message(None, None, message)
    listener._on_message(None, None, message)

    assert listener._queue.qsize() == 1


# -- motion mode ------------------------------------------------------------


@pytest.fixture
def motion_listener(config: Config) -> FrigateListener:
    return FrigateListener(replace(config, trigger_mode=TriggerMode.MOTION), handler=lambda _t: None)


def test_motion_topic_follows_camera_name(config: Config):
    motion = replace(config, trigger_mode=TriggerMode.MOTION)
    assert motion.mqtt_topic == "frigate/fridge/motion"
    assert config.mqtt_topic == "frigate/events"


def test_motion_on_off_produces_a_window(motion_listener: FrigateListener, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("fridge_watcher.mqtt_listener.time.time", lambda: clock[0])
    topic = "frigate/fridge/motion"

    motion_listener._on_message(None, None, Message("ON", topic))
    clock[0] += 6.0
    motion_listener._on_message(None, None, Message("OFF", topic))

    trigger = motion_listener._queue.get_nowait()
    assert trigger.window == (1000.0, 1006.0)
    assert trigger.source == "motion"
    assert trigger.capture_id == "motion-fridge-1000"
    assert trigger.delay_s > 0  # let Frigate flush its recording segments


def test_motion_blips_are_ignored(motion_listener: FrigateListener, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("fridge_watcher.mqtt_listener.time.time", lambda: clock[0])
    topic = "frigate/fridge/motion"

    motion_listener._on_message(None, None, Message("ON", topic))
    clock[0] += 0.2
    motion_listener._on_message(None, None, Message("OFF", topic))

    assert motion_listener._queue.empty()


def test_long_motion_window_is_clamped(motion_listener: FrigateListener, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("fridge_watcher.mqtt_listener.time.time", lambda: clock[0])
    topic = "frigate/fridge/motion"

    motion_listener._on_message(None, None, Message("ON", topic))
    clock[0] += 300.0
    motion_listener._on_message(None, None, Message("OFF", topic))

    trigger = motion_listener._queue.get_nowait()
    start, end = trigger.window
    assert end - start == pytest.approx(MOTION_MAX_SECONDS)


def test_motion_off_without_on_is_ignored(motion_listener: FrigateListener):
    motion_listener._on_message(None, None, Message("OFF", "frigate/fridge/motion"))
    assert motion_listener._queue.empty()
