"""The parser must never crash on model output, however malformed."""

from __future__ import annotations

import pytest

from fridge_watcher.vision import Direction, build_message_content, parse_response
from fridge_watcher.frame_extractor import Frame

GOOD = """{
  "item": "oat milk carton",
  "category": "dairy_alternative",
  "quantity": 1,
  "direction": "OUT",
  "confidence": 0.82,
  "reasoning": "carton is at the door in Frame 2 and by the sink in Frame 5"
}"""


def test_parses_clean_json():
    result = parse_response(GOOD)
    assert result.parsed
    assert result.item == "oat milk carton"
    assert result.category == "dairy_alternative"
    assert result.direction is Direction.OUT
    assert result.confidence == pytest.approx(0.82)
    assert result.quantity == 1


def test_strips_markdown_fences():
    result = parse_response("```json\n" + GOOD + "\n```")
    assert result.parsed
    assert result.direction is Direction.OUT


def test_extracts_object_from_surrounding_prose():
    raw = "Here is my analysis:\n" + GOOD + "\nHope that helps!"
    result = parse_response(raw)
    assert result.parsed
    assert result.item == "oat milk carton"


def test_braces_inside_strings_do_not_break_extraction():
    raw = 'prose {"item": "jar {special}", "category": "condiment", "quantity": 1, ' \
          '"direction": "IN", "confidence": 0.9, "reasoning": "moves toward door"} tail'
    result = parse_response(raw)
    assert result.parsed
    assert result.item == "jar {special}"


def test_no_item_zeroes_quantity_and_confidence():
    result = parse_response(
        '{"item": null, "category": null, "quantity": 1, "direction": "NO_ITEM",'
        ' "confidence": 0.4, "reasoning": "empty scene"}'
    )
    assert result.parsed
    assert result.direction is Direction.NO_ITEM
    assert result.quantity == 0
    assert result.confidence == 0.0
    assert not result.has_item


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "I'm sorry, I can't tell what that is.",
        "{not json at all",
        "[1, 2, 3]",
        '{"direction": "SIDEWAYS", "item": "x", "confidence": 0.9}',
        '{"item": "milk", "direction": "OUT"',
        '{"item": null, "direction": "OUT", "confidence": 0.9}',
    ],
)
def test_bad_output_routes_to_review_instead_of_crashing(raw):
    result = parse_response(raw)
    assert not result.parsed
    assert result.parse_error
    assert result.raw_response == raw


def test_confidence_is_clamped_and_coerced():
    assert parse_response(
        '{"item": "x", "direction": "IN", "confidence": 4.2, "quantity": 1}'
    ).confidence == 1.0
    assert parse_response(
        '{"item": "x", "direction": "IN", "confidence": -3, "quantity": 1}'
    ).confidence == 0.0
    assert parse_response(
        '{"item": "x", "direction": "IN", "confidence": "not a number", "quantity": 1}'
    ).confidence == 0.0


def test_quantity_falls_back_to_one():
    assert parse_response('{"item": "x", "direction": "IN", "confidence": 0.9}').quantity == 1
    assert parse_response(
        '{"item": "x", "direction": "IN", "confidence": 0.9, "quantity": 0}'
    ).quantity == 1
    assert parse_response(
        '{"item": "x", "direction": "IN", "confidence": 0.9, "quantity": "3"}'
    ).quantity == 3


def test_unknown_category_falls_back_to_other():
    result = parse_response(
        '{"item": "x", "category": "space food", "direction": "IN", "confidence": 0.9}'
    )
    assert result.category == "other"


def test_message_content_is_labelled_and_ordered():
    frames = [Frame(position=i + 1, total=3, jpeg=b"\xff\xd8fake") for i in range(3)]
    content = build_message_content(frames)

    # preamble + (label, image) per frame
    assert len(content) == 1 + 2 * 3
    labels = [b["text"] for b in content if b["type"] == "text"]
    assert labels[1:] == ["Frame 1 of 3", "Frame 2 of 3", "Frame 3 of 3"]
    images = [b for b in content if b["type"] == "image"]
    assert all(img["source"]["media_type"] == "image/jpeg" for img in images)
