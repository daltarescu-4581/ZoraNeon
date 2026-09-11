"""MQTT subscription to Frigate, in either `events` or `motion` mode.

Processing one trigger takes several seconds (clip download + vision call), so
messages are handed to a worker thread rather than processed inside paho's
network callback -- blocking there stops keepalive pings and the broker
eventually drops us mid-clip.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

import paho.mqtt.client as mqtt

from .config import (
    MOTION_MAX_SECONDS,
    MOTION_MIN_SECONDS,
    MOTION_SETTLE_SECONDS,
    Config,
    TriggerMode,
)
from .pipeline import Trigger

log = logging.getLogger(__name__)

QUEUE_SIZE = 32
RECENT_EVENTS_SIZE = 256
RECONNECT_MIN_DELAY = 1
RECONNECT_MAX_DELAY = 60

TriggerHandler = Callable[[Trigger], None]


class _RecentIds:
    """Small LRU of trigger IDs we already queued.

    Supabase's UNIQUE constraint is the real idempotency guard; this just saves
    a clip download and a vision call when MQTT redelivers a message.
    """

    def __init__(self, size: int = RECENT_EVENTS_SIZE) -> None:
        self._size = size
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def add_if_new(self, key: str) -> bool:
        with self._lock:
            if key in self._seen:
                self._seen.move_to_end(key)
                return False
            self._seen[key] = None
            while len(self._seen) > self._size:
                self._seen.popitem(last=False)
            return True


class FrigateListener:
    def __init__(self, config: Config, handler: TriggerHandler) -> None:
        self._config = config
        self._handler = handler
        self._queue: queue.Queue[Trigger | None] = queue.Queue(maxsize=QUEUE_SIZE)
        self._recent = _RecentIds()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        # Motion mode only: when the current ON period started.
        self._motion_started_at: float | None = None

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"fridge-watcher-{int(time.time())}",
        )
        if config.mqtt_user:
            self._client.username_pw_set(config.mqtt_user, config.mqtt_password or "")
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(RECONNECT_MIN_DELAY, RECONNECT_MAX_DELAY)

    # -- lifecycle ----------------------------------------------------------

    def run_forever(self) -> None:
        self._worker = threading.Thread(target=self._drain, name="pipeline", daemon=True)
        self._worker.start()
        log.info(
            "mqtt_connecting",
            extra={
                "host": self._config.mqtt_host,
                "port": self._config.mqtt_port,
                "topic": self._config.mqtt_topic,
                "trigger_mode": self._config.trigger_mode.value,
            },
        )
        self._connect_with_retry()
        try:
            self._client.loop_forever()
        finally:
            self.stop()

    def _connect_with_retry(self) -> None:
        """Wait for the broker rather than dying if it is not up yet.

        Under docker compose this service and Mosquitto start together, so the
        first connect routinely lands before the broker is listening. paho only
        reconnects on its own after one successful connection, so the very
        first one is ours to retry.
        """
        delay = RECONNECT_MIN_DELAY
        attempt = 0
        while not self._stop.is_set():
            attempt += 1
            try:
                self._client.connect(
                    self._config.mqtt_host, self._config.mqtt_port, keepalive=60
                )
                return
            except OSError as exc:
                log.warning(
                    "mqtt_connect_retry",
                    extra={
                        "host": self._config.mqtt_host,
                        "port": self._config.mqtt_port,
                        "attempt": attempt,
                        "delay": delay,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                self._stop.wait(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY)

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._client.disconnect()
        except Exception:  # noqa: BLE001 - shutting down anyway
            pass
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=30)

    # -- worker -------------------------------------------------------------

    def _drain(self) -> None:
        while not self._stop.is_set():
            trigger = self._queue.get()
            if trigger is None:
                break
            try:
                if trigger.delay_s > 0:
                    time.sleep(trigger.delay_s)
                self._handler(trigger)
            except Exception:  # noqa: BLE001 - one bad clip must not kill the service
                log.exception("trigger_failed", extra={"capture_id": trigger.capture_id})
            finally:
                self._queue.task_done()

    def _enqueue(self, trigger: Trigger) -> None:
        if not self._recent.add_if_new(trigger.capture_id):
            log.info("duplicate_trigger_skipped", extra={"capture_id": trigger.capture_id})
            return
        try:
            self._queue.put_nowait(trigger)
        except queue.Full:
            log.error("queue_full_dropped", extra={"capture_id": trigger.capture_id})

    # -- paho callbacks -----------------------------------------------------

    def _on_connect(self, client: mqtt.Client, _userdata: Any, _flags: Any, reason_code: Any, _properties: Any = None) -> None:
        if getattr(reason_code, "is_failure", False):
            log.error("mqtt_connect_failed", extra={"reason": str(reason_code)})
            return
        topic = self._config.mqtt_topic
        client.subscribe(topic)
        log.info("mqtt_subscribed", extra={"topic": topic})

    def _on_disconnect(self, _client: mqtt.Client, _userdata: Any, _flags: Any = None, reason_code: Any = None, _properties: Any = None) -> None:
        log.warning("mqtt_disconnected", extra={"reason": str(reason_code)})

    def _on_message(self, _client: mqtt.Client, _userdata: Any, message: mqtt.MQTTMessage) -> None:
        try:
            if self._config.trigger_mode is TriggerMode.MOTION:
                self._handle_motion(message)
            else:
                self._handle_event(message)
        except Exception:  # noqa: BLE001 - never let a bad payload kill the loop
            log.exception("message_handling_failed", extra={"topic": message.topic})

    # -- payload handling ---------------------------------------------------

    def _handle_event(self, message: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(message.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            log.warning("bad_event_payload", extra={"error": str(exc)})
            return
        trigger = self.trigger_from_event(payload)
        if trigger is not None:
            self._enqueue(trigger)

    def trigger_from_event(self, payload: dict[str, Any]) -> Trigger | None:
        """Filter frigate/events down to "something finished at the door"."""
        if payload.get("type") != "end":
            return None

        after = payload.get("after") or {}
        event_id = after.get("id")
        if not event_id:
            log.warning("event_missing_id", extra={"payload_keys": sorted(payload)})
            return None

        camera = after.get("camera")
        if camera and camera != self._config.frigate_camera_name:
            return None

        entered = after.get("entered_zones") or []
        if self._config.zone_name not in entered:
            log.debug(
                "event_outside_zone",
                extra={"event_id": event_id, "entered_zones": entered},
            )
            return None

        log.info(
            "event_accepted",
            extra={
                "event_id": event_id,
                "label": after.get("label"),
                "entered_zones": entered,
            },
        )
        return Trigger(capture_id=str(event_id), event_id=str(event_id), source="events")

    def _handle_motion(self, message: mqtt.MQTTMessage) -> None:
        """Motion mode: ON starts a window, OFF closes it and fires.

        Frigate's object detector tracks people, not groceries, so an arm
        reaching into the fridge can produce no event at all. Motion is the
        blunter but more reliable trigger.
        """
        state = message.payload.decode("utf-8", errors="replace").strip().upper()
        now = time.time()

        if state == "ON":
            if self._motion_started_at is None:
                self._motion_started_at = now
                log.info("motion_started", extra={"at": now})
            return

        if state != "OFF":
            log.debug("motion_payload_ignored", extra={"payload": state})
            return

        started = self._motion_started_at
        self._motion_started_at = None
        if started is None:
            log.debug("motion_off_without_on")
            return

        duration = now - started
        if duration < MOTION_MIN_SECONDS:
            log.info("motion_too_short", extra={"duration_s": round(duration, 2)})
            return
        if duration > MOTION_MAX_SECONDS:
            # Somebody is unpacking groceries; a 90s clip is useless to the
            # model, so keep the tail where the last item was handled.
            log.info("motion_window_clamped", extra={"duration_s": round(duration, 2)})
            started = now - MOTION_MAX_SECONDS

        capture_id = f"motion-{self._config.frigate_camera_name}-{int(started)}"
        log.info(
            "motion_accepted",
            extra={"capture_id": capture_id, "duration_s": round(now - started, 2)},
        )
        self._enqueue(
            Trigger(
                capture_id=capture_id,
                window=(started, now),
                source="motion",
                # Frigate needs a moment to flush the recording segments for
                # this window; wait in the worker, not in the network callback.
                delay_s=MOTION_SETTLE_SECONDS,
            )
        )
