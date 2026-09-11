"""Supabase writes.

Two rules drive everything here:
  * Frigate event IDs are unique, so a redelivered MQTT message must not
    double-count. The UNIQUE constraint on inventory_events.event_id is the
    guard, not a client-side "have I seen this?" check.
  * The event row and the stock delta belong in one transaction, which is why
    the happy path is a single RPC (see migrations/0001_inventory.sql). The
    two-round-trip fallback exists only so the service still works if the
    migration has not been run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from supabase import Client, create_client

from .config import Config
from .vision import Direction, VisionResult

log = logging.getLogger(__name__)

RPC_NAME = "apply_inventory_event"


class Status(str, Enum):
    APPLIED = "applied"
    PENDING_REVIEW = "pending_review"
    REJECTED = "rejected"


@dataclass(frozen=True)
class WriteOutcome:
    """What actually happened in the database for one event."""

    status: Status
    logged: bool          # an inventory_events row exists because of us
    duplicate: bool       # we had already processed this event_id
    applied: bool         # inventory_items was moved
    quantity_after: int | None = None


class InventoryStore(Protocol):
    def record(self, event_id: str, result: VisionResult, frames_path: str | None) -> WriteOutcome:
        ...


def normalize_item_name(name: str) -> str:
    """Collapse casing/spacing so upserts hit one row per real-world item."""
    return " ".join(name.strip().lower().split())


def status_for(result: VisionResult, threshold: float) -> Status:
    if not result.parsed:
        return Status.PENDING_REVIEW
    if result.confidence >= threshold:
        return Status.APPLIED
    return Status.PENDING_REVIEW


def _error_text(exc: Exception) -> str:
    parts = [str(exc)]
    for attr in ("code", "message", "details", "hint"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(str(value))
    return " ".join(parts).lower()


def _is_unique_violation(exc: Exception) -> bool:
    text = _error_text(exc)
    return "23505" in text or "duplicate key" in text


def _is_missing_function(exc: Exception) -> bool:
    text = _error_text(exc)
    return "pgrst202" in text or "42883" in text or "could not find the function" in text


class SupabaseStore:
    """Writes inventory events and stock levels to Supabase."""

    def __init__(self, config: Config, client: Client | None = None) -> None:
        self._config = config
        self._client = client or create_client(
            config.supabase_url, config.supabase_service_key
        )
        self._rpc_available = True

    def record(
        self, event_id: str, result: VisionResult, frames_path: str | None
    ) -> WriteOutcome:
        """Log the event and, above threshold, move the stock. Idempotent."""
        if result.parsed and result.direction is Direction.NO_ITEM:
            # Nothing was carried, so there is nothing worth a row.
            log.info("no_item_skipped", extra={"event_id": event_id})
            return WriteOutcome(
                status=Status.REJECTED, logged=False, duplicate=False, applied=False
            )

        status = status_for(result, self._config.confidence_threshold)
        item_name = normalize_item_name(result.item) if result.item else None
        payload = {
            "p_event_id": event_id,
            "p_item_name": item_name,
            "p_category": result.category,
            "p_direction": result.direction.value,
            "p_quantity": max(1, result.quantity),
            "p_confidence": round(result.confidence, 4),
            "p_reasoning": result.reasoning or (result.parse_error or ""),
            "p_frames_path": frames_path,
            "p_status": status.value,
        }

        if self._rpc_available:
            try:
                outcome = self._record_via_rpc(payload, status)
            except Exception as exc:  # noqa: BLE001 - we classify below
                if not _is_missing_function(exc):
                    raise
                self._rpc_available = False
                log.warning(
                    "rpc_missing_falling_back",
                    extra={
                        "rpc": RPC_NAME,
                        "hint": "run migrations/0001_inventory.sql for atomic writes",
                    },
                )
                outcome = self._record_via_tables(payload, status)
        else:
            outcome = self._record_via_tables(payload, status)

        log.info(
            "inventory_write",
            extra={
                "event_id": event_id,
                "item": item_name,
                "direction": result.direction.value,
                "confidence": round(result.confidence, 4),
                "status": outcome.status.value,
                "duplicate": outcome.duplicate,
                "applied": outcome.applied,
                "quantity_after": outcome.quantity_after,
            },
        )
        return outcome

    # -- write paths --------------------------------------------------------

    def _record_via_rpc(self, payload: dict[str, Any], status: Status) -> WriteOutcome:
        response = self._client.rpc(RPC_NAME, payload).execute()
        data = response.data
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            data = {}
        return WriteOutcome(
            status=status,
            logged=bool(data.get("logged", True)),
            duplicate=bool(data.get("duplicate", False)),
            applied=bool(data.get("applied", False)),
            quantity_after=data.get("quantity"),
        )

    def _record_via_tables(self, payload: dict[str, Any], status: Status) -> WriteOutcome:
        """Fallback for a database without the migration's RPC.

        Not atomic: a crash between the two writes loses the stock delta. The
        event row still lands first, so a redelivery is never double-counted --
        we would rather drop a delta than invent one.
        """
        row = {
            "event_id": payload["p_event_id"],
            "item_name": payload["p_item_name"],
            "direction": payload["p_direction"],
            "quantity": payload["p_quantity"],
            "confidence": payload["p_confidence"],
            "reasoning": payload["p_reasoning"],
            "frames_path": payload["p_frames_path"],
            "status": payload["p_status"],
        }
        try:
            self._client.table("inventory_events").insert(row).execute()
        except Exception as exc:  # noqa: BLE001 - unique violation means "seen it"
            if not _is_unique_violation(exc):
                raise
            return WriteOutcome(
                status=status, logged=False, duplicate=True, applied=False
            )

        if status is not Status.APPLIED:
            return WriteOutcome(status=status, logged=True, duplicate=False, applied=False)

        quantity_after = self._apply_delta(
            name=payload["p_item_name"],
            category=payload["p_category"],
            direction=payload["p_direction"],
            quantity=payload["p_quantity"],
        )
        return WriteOutcome(
            status=status,
            logged=True,
            duplicate=False,
            applied=True,
            quantity_after=quantity_after,
        )

    def _apply_delta(
        self, *, name: str, category: str | None, direction: str, quantity: int
    ) -> int:
        delta = -quantity if direction == Direction.OUT.value else quantity
        existing = (
            self._client.table("inventory_items")
            .select("quantity, category")
            .eq("name", name)
            .limit(1)
            .execute()
        )
        current = existing.data[0]["quantity"] if existing.data else 0
        # Never let stock go negative: taking out something we never saw go in
        # is a gap in our history, not evidence of a negative fridge.
        new_quantity = max(0, current + delta)
        row: dict[str, Any] = {
            "name": name,
            "quantity": new_quantity,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if category:
            row["category"] = category
        self._client.table("inventory_items").upsert(row, on_conflict="name").execute()
        return new_quantity
