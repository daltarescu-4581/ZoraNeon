"""Frame sampling has to span the whole clip -- that span is the direction signal."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from fridge_watcher.frame_extractor import (
    FrameExtractionError,
    _target_indices,
    extract_frames,
    load_frames,
    save_frames,
)


def write_clip(path: Path, frames: int = 30, size: tuple[int, int] = (160, 120)) -> Path:
    """A clip where frame N is a solid grey of value N -- so a decoded frame
    tells us exactly which source frame it came from."""
    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((height, width, 3), i * 8 % 256, dtype=np.uint8))
    writer.release()
    return path


def test_target_indices_span_the_clip():
    assert _target_indices(30, 6) == [0, 6, 12, 17, 23, 29]
    # Endpoints included: first and last sightings carry the direction.
    assert _target_indices(30, 6)[0] == 0
    assert _target_indices(30, 6)[-1] == 29


def test_target_indices_handles_short_clips():
    assert _target_indices(3, 6) == [0, 1, 2]
    assert _target_indices(1, 6) == [0]


def test_extracts_requested_number_of_frames(tmp_path: Path):
    clip = write_clip(tmp_path / "clip.mp4", frames=30)
    frames = extract_frames(clip, 6)

    assert len(frames) == 6
    assert [f.position for f in frames] == [1, 2, 3, 4, 5, 6]
    assert all(f.total == 6 for f in frames)
    assert [f.label for f in frames][0] == "Frame 1 of 6"
    assert all(f.jpeg.startswith(b"\xff\xd8") for f in frames)


def test_extracted_frames_are_distinct_moments(tmp_path: Path):
    """Regression guard for keyframe seeking: sampled frames must not collapse
    onto the same picture, or every trajectory looks flat."""
    clip = write_clip(tmp_path / "clip.mp4", frames=40)
    frames = extract_frames(clip, 6)

    decoded = [
        cv2.imdecode(np.frombuffer(f.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        for f in frames
    ]
    means = [float(img.mean()) for img in decoded]
    assert len(set(round(m) for m in means)) == len(means)


def test_clip_shorter_than_frame_count(tmp_path: Path):
    clip = write_clip(tmp_path / "short.mp4", frames=3)
    frames = extract_frames(clip, 6)
    assert 1 <= len(frames) <= 3
    assert all(f.total == len(frames) for f in frames)


def test_missing_clip_raises():
    with pytest.raises(FrameExtractionError):
        extract_frames(Path("/nope/missing.mp4"), 6)


def test_unreadable_clip_raises(tmp_path: Path):
    bogus = tmp_path / "not-a-video.mp4"
    bogus.write_bytes(b"definitely not an mp4")
    with pytest.raises(FrameExtractionError):
        extract_frames(bogus, 6)


def test_save_and_load_round_trip(tmp_path: Path):
    clip = write_clip(tmp_path / "clip.mp4", frames=20)
    frames = extract_frames(clip, 4)

    capture_dir = save_frames(frames, tmp_path / "captures" / "evt1")
    assert sorted(p.name for p in capture_dir.glob("*.jpg")) == [
        "frame_01.jpg", "frame_02.jpg", "frame_03.jpg", "frame_04.jpg",
    ]

    reloaded = load_frames(capture_dir)
    assert [f.jpeg for f in reloaded] == [f.jpeg for f in frames]
    assert [f.position for f in reloaded] == [1, 2, 3, 4]


def test_load_frames_requires_frames(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FrameExtractionError):
        load_frames(empty)
