"""Scoring for `replay-all`: how did this prompt do against hand labels?

The point is to be able to answer "did that prompt change help?" before
shipping it, using real footage rather than a fridge visit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .vision import Direction, VisionResult

_PUNCT = re.compile(r"[^a-z0-9 ]+")
# A shared token has to carry meaning; "of milk" should not match "of eggs".
_STOPWORDS = frozenset({"a", "an", "the", "of", "with", "and"})
# Packaging words describe the container, not the contents. Sharing one proves
# nothing: "egg carton" and "oat milk carton" are different items.
_GENERIC = frozenset(
    {
        "carton", "container", "bottle", "jar", "box", "bag", "packet", "pack",
        "tub", "tray", "can", "jug", "bowl", "plate", "wrapper", "item",
    }
)


@dataclass(frozen=True)
class Expectation:
    """One hand-labelled capture."""

    capture_id: str
    item: str | None
    direction: Direction
    quantity: int | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Score:
    capture_id: str
    expected: Expectation
    result: VisionResult
    applied: bool  # would this have been written straight to inventory?

    @property
    def direction_ok(self) -> bool:
        return self.result.direction is self.expected.direction

    @property
    def item_ok(self) -> bool:
        if self.expected.direction is Direction.NO_ITEM:
            return self.result.direction is Direction.NO_ITEM
        if not self.result.item or not self.expected.item:
            return False
        candidates = (self.expected.item, *self.expected.aliases)
        return any(items_match(self.result.item, c) for c in candidates)

    @property
    def quantity_ok(self) -> bool | None:
        if self.expected.quantity is None:
            return None
        return self.result.quantity == self.expected.quantity

    @property
    def correct(self) -> bool:
        return self.direction_ok and self.item_ok and (self.quantity_ok is not False)


@dataclass(frozen=True)
class Report:
    scores: list[Score]
    missing: list[str]     # expectations with no matching capture directory
    unlabelled: list[str]  # captures with no expectation

    @property
    def total(self) -> int:
        return len(self.scores)

    def _rate(self, predicate) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if predicate(s)) / len(self.scores)

    @property
    def direction_accuracy(self) -> float:
        return self._rate(lambda s: s.direction_ok)

    @property
    def item_accuracy(self) -> float:
        return self._rate(lambda s: s.item_ok)

    @property
    def overall_accuracy(self) -> float:
        return self._rate(lambda s: s.correct)

    @property
    def mean_confidence(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.result.confidence for s in self.scores) / len(self.scores)

    @property
    def false_applies(self) -> list[Score]:
        """Wrong, and confident enough that we would have written it anyway.
        These are the ones that corrupt the user's inventory silently."""
        return [s for s in self.scores if s.applied and not s.correct]

    @property
    def missed_applies(self) -> list[Score]:
        """Right, but below threshold -- unnecessary review-queue noise."""
        return [s for s in self.scores if not s.applied and s.correct]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "direction_accuracy": round(self.direction_accuracy, 4),
            "item_accuracy": round(self.item_accuracy, 4),
            "overall_accuracy": round(self.overall_accuracy, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "false_applies": len(self.false_applies),
            "missed_applies": len(self.missed_applies),
            "missing_captures": self.missing,
            "unlabelled_captures": self.unlabelled,
            "captures": [
                {
                    "capture_id": s.capture_id,
                    "expected_item": s.expected.item,
                    "expected_direction": s.expected.direction.value,
                    "got_item": s.result.item,
                    "got_direction": s.result.direction.value,
                    "confidence": round(s.result.confidence, 4),
                    "direction_ok": s.direction_ok,
                    "item_ok": s.item_ok,
                    "correct": s.correct,
                    "applied": s.applied,
                    "reasoning": s.result.reasoning,
                }
                for s in self.scores
            ],
        }


def _normalize(text: str) -> list[str]:
    cleaned = _PUNCT.sub(" ", text.lower())
    return [t for t in cleaned.split() if t and t not in _STOPWORDS]


def items_match(got: str, expected: str) -> bool:
    """Loose name match.

    Hand labels say "oat milk" where the model says "carton of oat milk"; both
    are right for inventory purposes. Exact string equality would make the
    accuracy number meaningless, so accept containment or a majority of shared
    tokens.
    """
    got_tokens = set(_normalize(got))
    expected_tokens = set(_normalize(expected))
    if not got_tokens or not expected_tokens:
        return False
    if got_tokens == expected_tokens:
        return True

    shared = got_tokens & expected_tokens
    if not shared or not shared - _GENERIC:
        return False
    if got_tokens <= expected_tokens or expected_tokens <= got_tokens:
        return True
    return len(shared) / min(len(got_tokens), len(expected_tokens)) >= 0.5


def load_expectations(path: Path) -> dict[str, Expectation]:
    """Accept either {capture_id: {...}} or [{"capture_id": ..., ...}]."""
    raw = json.loads(path.read_text())

    entries: Iterable[tuple[str, dict[str, Any]]]
    if isinstance(raw, dict):
        entries = raw.items()
    elif isinstance(raw, list):
        entries = (
            (str(e.get("capture_id") or e.get("event_id") or ""), e)
            for e in raw
            if isinstance(e, dict)
        )
    else:
        raise ValueError(f"{path}: expected a JSON object or array")

    expectations: dict[str, Expectation] = {}
    for capture_id, entry in entries:
        if not capture_id:
            raise ValueError(f"{path}: an entry is missing capture_id/event_id")
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry for {capture_id} is not an object")
        direction_raw = str(entry.get("direction", "")).strip().upper()
        try:
            direction = Direction(direction_raw)
        except ValueError as exc:
            raise ValueError(
                f"{path}: {capture_id} has direction {direction_raw!r}; "
                f"expected IN, OUT or NO_ITEM"
            ) from exc
        aliases = entry.get("aliases") or []
        expectations[capture_id] = Expectation(
            capture_id=capture_id,
            item=(str(entry["item"]).strip() if entry.get("item") else None),
            direction=direction,
            quantity=(int(entry["quantity"]) if entry.get("quantity") is not None else None),
            aliases=tuple(str(a) for a in aliases),
        )
    return expectations


def format_report(report: Report) -> str:
    """A terminal-readable table plus the numbers worth watching."""
    lines: list[str] = []
    header = f"{'capture':<34} {'expected':<26} {'got':<26} {'conf':>5}  result"
    lines.append(header)
    lines.append("-" * len(header))

    for score in sorted(report.scores, key=lambda s: (s.correct, s.capture_id)):
        expected = f"{score.expected.direction.value} {score.expected.item or '-'}"
        got = f"{score.result.direction.value} {score.result.item or '-'}"
        if score.correct:
            verdict = "OK" + ("" if score.applied else "  (below threshold)")
        else:
            wrong = []
            if not score.direction_ok:
                wrong.append("direction")
            if not score.item_ok:
                wrong.append("item")
            if score.quantity_ok is False:
                wrong.append("quantity")
            verdict = "WRONG: " + ", ".join(wrong)
            if score.applied:
                verdict += "  <-- would have been applied"
        lines.append(
            f"{score.capture_id[:33]:<34} {expected[:25]:<26} {got[:25]:<26} "
            f"{score.result.confidence:>5.2f}  {verdict}"
        )

    lines.append("")
    lines.append(f"captures scored      {report.total}")
    lines.append(f"direction accuracy   {report.direction_accuracy:.1%}")
    lines.append(f"item accuracy        {report.item_accuracy:.1%}")
    lines.append(f"overall accuracy     {report.overall_accuracy:.1%}")
    lines.append(f"mean confidence      {report.mean_confidence:.2f}")
    lines.append(
        f"false applies        {len(report.false_applies)}"
        "   (wrong AND above threshold -- these corrupt inventory silently)"
    )
    lines.append(
        f"missed applies       {len(report.missed_applies)}"
        "   (right but below threshold -- review-queue noise)"
    )
    if report.missing:
        lines.append(f"labelled but no capture dir: {', '.join(report.missing)}")
    if report.unlabelled:
        lines.append(f"captures with no label:      {', '.join(report.unlabelled)}")
    return "\n".join(lines)
