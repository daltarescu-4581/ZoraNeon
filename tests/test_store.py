"""Idempotency and threshold routing -- the two things that corrupt inventory."""

from __future__ import annotations

from typing import Any

import pytest

from fridge_watcher.config import Config
from fridge_watcher.store import (
    RPC_NAME,
    Status,
    SupabaseStore,
    normalize_item_name,
    status_for,
)
from fridge_watcher.vision import Direction, VisionResult


class _Response:
    def __init__(self, data: Any) -> None:
        self.data = data


class FakeRpc:
    """Stands in for a Supabase project that has the migration applied."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.seen_event_ids: set[str] = set()
        self.stock: dict[str, int] = {}

    def rpc(self, name: str, params: dict[str, Any]):
        assert name == RPC_NAME
        self.calls.append(params)
        return self

    def execute(self) -> _Response:
        params = self.calls[-1]
        event_id = params["p_event_id"]
        if event_id in self.seen_event_ids:
            return _Response({"logged": False, "duplicate": True, "applied": False})
        self.seen_event_ids.add(event_id)
        if params["p_status"] != Status.APPLIED.value:
            return _Response({"logged": True, "duplicate": False, "applied": False})
        delta = (
            -params["p_quantity"]
            if params["p_direction"] == "OUT"
            else params["p_quantity"]
        )
        name = params["p_item_name"]
        self.stock[name] = max(0, self.stock.get(name, 0) + delta)
        return _Response(
            {"logged": True, "duplicate": False, "applied": True, "quantity": self.stock[name]}
        )


def result(direction=Direction.OUT, confidence=0.9, item="Oat Milk Carton", quantity=1):
    return VisionResult(
        item=item,
        category="dairy_alternative",
        quantity=quantity,
        direction=direction,
        confidence=confidence,
        reasoning="trajectory moves away from the door",
    )


def test_normalize_item_name():
    assert normalize_item_name("  Oat   Milk Carton ") == "oat milk carton"


def test_status_routing(config: Config):
    assert status_for(result(confidence=0.9), 0.75) is Status.APPLIED
    assert status_for(result(confidence=0.75), 0.75) is Status.APPLIED
    assert status_for(result(confidence=0.74), 0.75) is Status.PENDING_REVIEW
    assert status_for(VisionResult.unparsable("junk", "bad"), 0.75) is Status.PENDING_REVIEW


def test_above_threshold_applies_delta(config: Config):
    fake = FakeRpc()
    store = SupabaseStore(config, client=fake)
    fake.stock["oat milk carton"] = 3

    outcome = store.record("evt-1", result(Direction.OUT, 0.9), "captures/evt-1")

    assert outcome.status is Status.APPLIED
    assert outcome.applied is True
    assert fake.stock["oat milk carton"] == 2
    assert fake.calls[-1]["p_item_name"] == "oat milk carton"  # normalised
    assert fake.calls[-1]["p_frames_path"] == "captures/evt-1"


def test_in_direction_adds(config: Config):
    fake = FakeRpc()
    store = SupabaseStore(config, client=fake)
    store.record("evt-in", result(Direction.IN, 0.95), None)
    assert fake.stock["oat milk carton"] == 1


def test_below_threshold_queues_for_review_without_touching_stock(config: Config):
    fake = FakeRpc()
    store = SupabaseStore(config, client=fake)

    outcome = store.record("evt-2", result(confidence=0.5), None)

    assert outcome.status is Status.PENDING_REVIEW
    assert outcome.applied is False
    assert fake.stock == {}


def test_unparsable_response_goes_to_review(config: Config):
    fake = FakeRpc()
    store = SupabaseStore(config, client=fake)

    outcome = store.record("evt-3", VisionResult.unparsable("not json", "invalid"), None)

    assert outcome.status is Status.PENDING_REVIEW
    assert fake.stock == {}
    assert fake.calls[-1]["p_reasoning"]  # the parse error is preserved


def test_no_item_writes_nothing(config: Config):
    fake = FakeRpc()
    store = SupabaseStore(config, client=fake)

    outcome = store.record(
        "evt-4",
        VisionResult(None, None, 0, Direction.NO_ITEM, 0.0, "empty scene"),
        None,
    )

    assert outcome.logged is False
    assert fake.calls == []


def test_redelivered_event_does_not_double_count(config: Config):
    fake = FakeRpc()
    store = SupabaseStore(config, client=fake)
    fake.stock["oat milk carton"] = 5

    first = store.record("evt-dup", result(Direction.OUT, 0.9), None)
    second = store.record("evt-dup", result(Direction.OUT, 0.9), None)

    assert first.applied is True
    assert second.duplicate is True
    assert second.applied is False
    assert fake.stock["oat milk carton"] == 4  # decremented exactly once


# ---------------------------------------------------------------------------
# Fallback path: a database without the migration's RPC.
# ---------------------------------------------------------------------------


class MissingFunctionError(Exception):
    code = "PGRST202"

    def __str__(self) -> str:
        return "Could not find the function public.apply_inventory_event"


class UniqueViolation(Exception):
    code = "23505"

    def __str__(self) -> str:
        return 'duplicate key value violates unique constraint "inventory_events_event_id_key"'


class FakeTables:
    """Supabase project with the tables but no RPC."""

    def __init__(self) -> None:
        self.events: dict[str, dict[str, Any]] = {}
        self.items: dict[str, dict[str, Any]] = {}
        self._table: str | None = None
        self._op: tuple[str, Any] | None = None
        self._filters: dict[str, Any] = {}

    def rpc(self, name: str, params: dict[str, Any]):
        raise MissingFunctionError()

    def table(self, name: str):
        self._table = name
        self._filters = {}
        return self

    def insert(self, row: dict[str, Any]):
        self._op = ("insert", row)
        return self

    def upsert(self, row: dict[str, Any], on_conflict: str | None = None):
        self._op = ("upsert", row)
        return self

    def select(self, _columns: str):
        self._op = ("select", None)
        return self

    def eq(self, column: str, value: Any):
        self._filters[column] = value
        return self

    def limit(self, _n: int):
        return self

    def execute(self) -> _Response:
        op, row = self._op  # type: ignore[misc]
        if self._table == "inventory_events":
            if op != "insert":
                raise AssertionError(f"unexpected {op} on inventory_events")
            if row["event_id"] in self.events:
                raise UniqueViolation()
            self.events[row["event_id"]] = row
            return _Response([row])
        if op == "select":
            name = self._filters.get("name")
            found = self.items.get(name)
            return _Response([found] if found else [])
        self.items[row["name"]] = {**self.items.get(row["name"], {}), **row}
        return _Response([row])


def test_falls_back_to_table_writes_when_rpc_is_missing(config: Config):
    fake = FakeTables()
    store = SupabaseStore(config, client=fake)
    fake.items["oat milk carton"] = {"name": "oat milk carton", "quantity": 4}

    outcome = store.record("evt-fb", result(Direction.OUT, 0.9), None)

    assert outcome.applied is True
    assert fake.items["oat milk carton"]["quantity"] == 3
    assert "evt-fb" in fake.events


def test_fallback_is_idempotent_on_unique_violation(config: Config):
    fake = FakeTables()
    store = SupabaseStore(config, client=fake)
    fake.items["oat milk carton"] = {"name": "oat milk carton", "quantity": 4}

    store.record("evt-fb2", result(Direction.OUT, 0.9), None)
    second = store.record("evt-fb2", result(Direction.OUT, 0.9), None)

    assert second.duplicate is True
    assert fake.items["oat milk carton"]["quantity"] == 3  # not 2


def test_fallback_clamps_stock_at_zero(config: Config):
    fake = FakeTables()
    store = SupabaseStore(config, client=fake)

    store.record("evt-neg", result(Direction.OUT, 0.9), None)

    assert fake.items["oat milk carton"]["quantity"] == 0


def test_unexpected_errors_are_not_swallowed(config: Config):
    class Boom:
        def rpc(self, *_args, **_kwargs):
            raise RuntimeError("connection refused")

    store = SupabaseStore(config, client=Boom())
    with pytest.raises(RuntimeError):
        store.record("evt-boom", result(), None)
