"""The vision call: frame sequence in, structured inventory verdict out.

The direction of travel is inferred from the item's *trajectory* across the
sequence, never from a before/after comparison of the counter -- items never
land in the same place twice, so state diffing is hopeless here.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

from anthropic import Anthropic

from .config import Config
from .frame_extractor import Frame

log = logging.getLogger(__name__)

MAX_TOKENS = 700
# Deterministic by design: replay-all is only a useful regression harness if
# the same frames produce the same verdict when the prompt has not changed.
TEMPERATURE = 0.0

CATEGORIES = (
    "dairy",
    "dairy_alternative",
    "produce",
    "meat",
    "seafood",
    "beverage",
    "condiment",
    "leftovers",
    "baked_goods",
    "frozen",
    "prepared",
    "other",
)


class Direction(str, Enum):
    IN = "IN"
    OUT = "OUT"
    NO_ITEM = "NO_ITEM"


SYSTEM_PROMPT = f"""\
You are the vision stage of a kitchen inventory system. A camera is mounted on \
the ceiling above a refrigerator, angled so that it sees both the fridge door \
and the counter area in front of it. You receive an ordered sequence of still \
frames sampled from one short clip, labelled "Frame 1 of N" through \
"Frame N of N" in chronological order.

Your job is to answer two questions about that clip:
  1. WHAT single grocery item was being handled?
  2. Which WAY was it travelling: out of the fridge, or back into it?

## How to determine direction

Direction comes from the item's TRAJECTORY across the sequence. Do not try to \
compare the "before" and "after" state of the counter -- items are never put \
down in the same place twice and that comparison is worthless here.

Work it out in this order:

1. Locate the fridge door in the frame. It is the large appliance face or the \
   dark rectangular opening, and it stays in the same place in every frame. \
   Anything that moves between frames is a person, a hand, or an item.
2. Find the item and note its approximate position in each frame where you can \
   see it. It does not need to be visible in every frame.
3. Compare those positions in order:
   - Position starts at or near the fridge door and gets FARTHER from it in \
     later frames -> direction is "OUT" (the item was removed from the fridge).
   - Position starts away from the fridge door and gets CLOSER to it in later \
     frames -> direction is "IN" (the item was returned to the fridge).
   - Frames where the item is hidden behind a body or an open door are fine. \
     Two or three sightings that agree on a direction is enough to call it.
4. If no identifiable grocery item appears in ANY frame -- an empty scene, a \
   person with nothing in their hands, a door opening and closing, a pet, a \
   hand with no visible object -- the direction is "NO_ITEM".

Hands, arms, bodies and the door itself are carriers, not items. Never report \
a hand or a person as the item.

## How to set confidence

`confidence` is your JOINT certainty that BOTH the item name and the direction \
are correct. Take the LOWER of your two certainties -- never the average. If \
the item is unmistakable but the direction is a coin flip, confidence must be \
low. Anything at or above 0.75 will be written straight into the user's \
inventory without a human looking at it; below that a human reviews it. Aim to \
be right about which side of that line you land on.

Rough calibration:
  0.90 - 1.00  Item unmistakable AND three or more sightings moving \
consistently in one direction.
  0.75 - 0.89  Item clearly identifiable AND two or three sightings that agree \
on a direction.
  0.40 - 0.74  Item identifiable but the direction is weak: only two sightings, \
motion is mostly across the frame rather than toward or away from the door, or \
the item is partly occluded. Still pick the more likely direction.
  0.00 - 0.39  Item is a guess, or motion is genuinely ambiguous.

Only one or two sightings of the item still gets a direction call -- just lower \
the confidence to match.

## Output

Reply with a single JSON object and nothing else. No prose, no explanation \
outside the JSON, no markdown code fences.

{{
  "item": "short lowercase noun phrase, e.g. \\"oat milk carton\\"",
  "category": "one of: {', '.join(CATEGORIES)}",
  "quantity": 1,
  "direction": "IN" | "OUT" | "NO_ITEM",
  "confidence": 0.0,
  "reasoning": "one sentence citing the trajectory evidence, naming the frames"
}}

Rules for the fields:
  - "item": the specific product if you can read a label, otherwise the generic \
form ("milk carton", "leftovers container"). Use null when direction is \
"NO_ITEM".
  - "quantity": how many units of that item moved. Default to 1. Only go higher \
when you can count distinct units. Use 0 when direction is "NO_ITEM".
  - "confidence": 0.0 to 1.0. Use 0.0 when direction is "NO_ITEM".
  - "reasoning": one sentence, and it must reference the frames you used, e.g. \
"carton is at the door edge in Frame 2 and near the sink by Frame 5".
"""

USER_PREAMBLE = (
    "Here are {count} frames from one clip at the fridge, in chronological "
    "order. Identify the item and its direction of travel relative to the "
    "fridge door, then reply with the JSON object only."
)


@dataclass(frozen=True)
class VisionResult:
    """Parsed model verdict, plus whatever we need to audit it later."""

    item: str | None
    category: str | None
    quantity: int
    direction: Direction
    confidence: float
    reasoning: str
    raw_response: str = ""
    parse_error: str | None = None

    @property
    def parsed(self) -> bool:
        return self.parse_error is None

    @property
    def has_item(self) -> bool:
        return self.direction is not Direction.NO_ITEM and bool(self.item)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["direction"] = self.direction.value
        return data

    @classmethod
    def unparsable(cls, raw: str, error: str) -> "VisionResult":
        """A verdict we could not read is not a crash -- it is a review item."""
        return cls(
            item=None,
            category=None,
            quantity=1,
            direction=Direction.NO_ITEM,
            confidence=0.0,
            reasoning="model response could not be parsed",
            raw_response=raw,
            parse_error=error,
        )


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    cleaned = _FENCE_RE.sub("", text.strip())
    return cleaned.strip()


def _first_json_object(text: str) -> str | None:
    """Find the outermost {...} in case the model wrapped it in prose."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_quantity(value: Any) -> int:
    try:
        quantity = int(float(value))
    except (TypeError, ValueError):
        return 1
    return quantity if quantity > 0 else 1


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, confidence))


def parse_response(raw: str) -> VisionResult:
    """Parse the model's reply defensively; never raise on bad output."""
    candidate = _strip_fences(raw)
    if not candidate:
        return VisionResult.unparsable(raw, "empty response")

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        extracted = _first_json_object(candidate)
        if extracted is None:
            return VisionResult.unparsable(raw, "no JSON object found in response")
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError as exc:
            return VisionResult.unparsable(raw, f"invalid JSON: {exc}")

    if not isinstance(data, dict):
        return VisionResult.unparsable(raw, f"expected a JSON object, got {type(data).__name__}")

    raw_direction = str(data.get("direction", "")).strip().upper()
    try:
        direction = Direction(raw_direction)
    except ValueError:
        return VisionResult.unparsable(raw, f"unknown direction {raw_direction!r}")

    item = data.get("item")
    item = str(item).strip() if isinstance(item, str) and item.strip() else None
    category = data.get("category")
    category = str(category).strip().lower() if isinstance(category, str) and category.strip() else None
    reasoning = data.get("reasoning")
    reasoning = str(reasoning).strip() if isinstance(reasoning, str) else ""

    if direction is Direction.NO_ITEM:
        return VisionResult(
            item=None,
            category=None,
            quantity=0,
            direction=direction,
            confidence=0.0,
            reasoning=reasoning or "no identifiable item in the sequence",
            raw_response=raw,
        )

    if item is None:
        # A direction with nothing to attach it to is unusable; let a human look.
        return VisionResult.unparsable(raw, f"direction {direction.value} with no item name")

    return VisionResult(
        item=item,
        category=category if category in CATEGORIES else "other",
        quantity=_coerce_quantity(data.get("quantity", 1)),
        direction=direction,
        confidence=_coerce_confidence(data.get("confidence")),
        reasoning=reasoning,
        raw_response=raw,
    )


def build_message_content(frames: list[Frame]) -> list[dict[str, Any]]:
    """One user message: a preamble, then label/image pairs in clip order."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": USER_PREAMBLE.format(count=len(frames))}
    ]
    for frame in frames:
        content.append({"type": "text", "text": frame.label})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame.as_base64(),
                },
            }
        )
    return content


class VisionAnalyzer:
    """Thin wrapper around the Anthropic client so the CLI can stub it out."""

    def __init__(self, config: Config, client: Anthropic | None = None) -> None:
        self._config = config
        self._client = client or Anthropic(api_key=config.anthropic_api_key)

    def analyze(self, frames: list[Frame]) -> VisionResult:
        if not frames:
            return VisionResult.unparsable("", "no frames to analyze")

        response = self._client.messages.create(
            model=self._config.anthropic_model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_message_content(frames)}],
        )

        raw = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        result = parse_response(raw)

        if not result.parsed:
            log.error(
                "vision_parse_failed",
                extra={"error": result.parse_error, "raw_response": raw[:2000]},
            )
        else:
            log.info(
                "vision_result",
                extra={
                    "item": result.item,
                    "direction": result.direction.value,
                    "confidence": result.confidence,
                    "frames": len(frames),
                    "model": self._config.anthropic_model,
                    "input_tokens": getattr(response.usage, "input_tokens", None),
                    "output_tokens": getattr(response.usage, "output_tokens", None),
                },
            )
        return result
