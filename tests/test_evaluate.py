"""Scoring for replay-all."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fridge_watcher.evaluate import (
    Expectation,
    Report,
    Score,
    format_report,
    items_match,
    load_expectations,
)
from fridge_watcher.vision import Direction, VisionResult


def vr(item, direction, confidence=0.9, quantity=1):
    return VisionResult(
        item=item,
        category="dairy",
        quantity=quantity,
        direction=direction,
        confidence=confidence,
        reasoning="because",
    )


@pytest.mark.parametrize(
    "got,expected",
    [
        ("oat milk carton", "oat milk carton"),
        ("carton of oat milk", "oat milk carton"),
        ("oat milk", "oat milk carton"),
        ("Oat Milk Carton!", "oat milk carton"),
    ],
)
def test_names_that_should_match(got, expected):
    assert items_match(got, expected)


@pytest.mark.parametrize(
    "got,expected",
    [
        # Only the packaging word is shared -- says nothing about contents.
        ("oat milk carton", "egg carton"),
        ("carton", "egg carton"),
        ("butter", "oat milk carton"),
        ("", "oat milk"),
    ],
)
def test_names_that_should_not_match(got, expected):
    assert not items_match(got, expected)


def test_score_flags_direction_error():
    score = Score(
        capture_id="c1",
        expected=Expectation("c1", "oat milk carton", Direction.IN),
        result=vr("oat milk carton", Direction.OUT),
        applied=True,
    )
    assert score.item_ok
    assert not score.direction_ok
    assert not score.correct


def test_score_accepts_alias():
    score = Score(
        capture_id="c1",
        expected=Expectation("c1", "tupperware", Direction.IN, aliases=("leftovers container",)),
        result=vr("leftovers container", Direction.IN),
        applied=True,
    )
    assert score.correct


def test_quantity_mismatch_fails_the_capture():
    score = Score(
        capture_id="c1",
        expected=Expectation("c1", "egg", Direction.OUT, quantity=2),
        result=vr("egg", Direction.OUT, quantity=1),
        applied=True,
    )
    assert score.quantity_ok is False
    assert not score.correct


def test_unlabelled_quantity_is_not_scored():
    score = Score(
        capture_id="c1",
        expected=Expectation("c1", "egg", Direction.OUT),
        result=vr("egg", Direction.OUT, quantity=4),
        applied=True,
    )
    assert score.quantity_ok is None
    assert score.correct


def test_report_separates_false_applies_from_review_noise():
    report = Report(
        scores=[
            # wrong and confident -> silently corrupts inventory
            Score("a", Expectation("a", "milk", Direction.IN), vr("milk", Direction.OUT, 0.9), True),
            # right but timid -> review-queue noise
            Score("b", Expectation("b", "eggs", Direction.OUT), vr("eggs", Direction.OUT, 0.4), False),
            Score("c", Expectation("c", "butter", Direction.IN), vr("butter", Direction.IN, 0.95), True),
        ],
        missing=[],
        unlabelled=[],
    )

    assert report.total == 3
    assert report.direction_accuracy == pytest.approx(2 / 3)
    assert report.item_accuracy == 1.0
    assert report.overall_accuracy == pytest.approx(2 / 3)
    assert [s.capture_id for s in report.false_applies] == ["a"]
    assert [s.capture_id for s in report.missed_applies] == ["b"]
    assert "false applies        1" in format_report(report)


def test_no_item_expectation_scores_correctly():
    score = Score(
        capture_id="c1",
        expected=Expectation("c1", None, Direction.NO_ITEM),
        result=vr(None, Direction.NO_ITEM, 0.0),
        applied=False,
    )
    assert score.correct


def test_load_expectations_object_form(tmp_path: Path):
    path = tmp_path / "exp.json"
    path.write_text(json.dumps({
        "c1": {"item": "oat milk", "direction": "out", "quantity": 1, "aliases": ["oat milk carton"]},
        "c2": {"item": None, "direction": "NO_ITEM"},
    }))

    expectations = load_expectations(path)

    assert expectations["c1"].direction is Direction.OUT
    assert expectations["c1"].aliases == ("oat milk carton",)
    assert expectations["c2"].item is None


def test_load_expectations_array_form(tmp_path: Path):
    path = tmp_path / "exp.json"
    path.write_text(json.dumps([{"event_id": "c1", "item": "eggs", "direction": "IN"}]))
    assert load_expectations(path)["c1"].item == "eggs"


def test_bad_direction_label_is_reported_clearly(tmp_path: Path):
    path = tmp_path / "exp.json"
    path.write_text(json.dumps({"c1": {"item": "eggs", "direction": "OUTWARD"}}))
    with pytest.raises(ValueError, match="OUTWARD"):
        load_expectations(path)
