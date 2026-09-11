"""Pull N evenly spaced frames out of a clip and encode them as JPEG.

Why OpenCV rather than shelling out to ffmpeg:
  * One pip install, no system binary to provision in the Docker image or on
    the Pi -- `opencv-python` ships its own decoders.
  * Frames come back as arrays we resize and JPEG-encode in-process, so there
    is no temp-file dance and no parsing of ffmpeg's stderr to find out
    whether it worked.
  * We need frame *indices* (evenly spaced across the clip), which is a
    counting problem in OpenCV and a timestamp-arithmetic problem in ffmpeg.
The cost is decode speed, which is irrelevant for 5-10 second fridge clips.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Claude downsamples anything larger anyway; 1024px long edge keeps each frame
# around ~1k tokens so a 6-frame sequence stays cheap.
MAX_EDGE_PX = 1024
JPEG_QUALITY = 80


class FrameExtractionError(RuntimeError):
    """The clip could not be opened or contained no decodable frames."""


@dataclass(frozen=True)
class Frame:
    """One sampled still, ready to be sent to the model or written to disk."""

    position: int  # 1-based, chronological
    total: int
    jpeg: bytes
    source_index: int | None = None  # frame number within the clip, if known

    @property
    def label(self) -> str:
        return f"Frame {self.position} of {self.total}"

    def as_base64(self) -> str:
        return base64.standard_b64encode(self.jpeg).decode("ascii")


def _target_indices(total_frames: int, want: int) -> list[int]:
    """Evenly spaced indices spanning the whole clip, endpoints included.

    Endpoints matter: the first and last sightings of an item carry most of
    the direction signal, so we sample the extremes rather than the midpoints
    of equal buckets.
    """
    if want >= total_frames:
        return list(range(total_frames))
    if want == 1:
        return [total_frames // 2]
    step = (total_frames - 1) / (want - 1)
    return sorted({int(round(i * step)) for i in range(want)})


def _encode(image: np.ndarray) -> bytes:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > MAX_EDGE_PX:
        scale = MAX_EDGE_PX / longest
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise FrameExtractionError("cv2.imencode failed on a decoded frame")
    return buffer.tobytes()


def _count_frames(path: Path) -> int:
    """Exact frame count via a grab-only pass.

    CAP_PROP_FRAME_COUNT is a container hint and Frigate's remuxed clips
    routinely disagree with it. `grab()` advances without decoding the
    picture, so counting this way is cheap and always correct.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FrameExtractionError(f"could not open clip: {path}")
    try:
        count = 0
        while capture.grab():
            count += 1
        return count
    finally:
        capture.release()


def extract_frames(video_path: Path | str, count: int) -> list[Frame]:
    """Decode `video_path` and return up to `count` evenly spaced JPEG frames.

    We walk the clip sequentially instead of seeking with CAP_PROP_POS_FRAMES:
    on H.264 clips seeking lands on the nearest keyframe, and a short clip may
    only have one or two -- which collapses several "evenly spaced" samples
    onto the same picture, fatal when the whole signal is movement between
    them. Only the frames we actually want get decoded, so this stays cheap.
    """
    path = Path(video_path)
    if not path.exists():
        raise FrameExtractionError(f"clip not found: {path}")

    total_frames = _count_frames(path)
    if total_frames == 0:
        raise FrameExtractionError(f"clip contained no decodable frames: {path}")

    wanted = _target_indices(total_frames, count)
    last_wanted = wanted[-1]
    wanted_set = set(wanted)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FrameExtractionError(f"could not open clip: {path}")
    picked: list[np.ndarray] = []
    try:
        index = 0
        while index <= last_wanted:
            if not capture.grab():
                break
            if index in wanted_set:
                ok, image = capture.retrieve()
                if ok and image is not None:
                    picked.append(image)
            index += 1
    finally:
        capture.release()

    if not picked:
        raise FrameExtractionError(f"no frames could be decoded from {path}")

    total = len(picked)
    frames = [
        Frame(position=i + 1, total=total, jpeg=_encode(image), source_index=source)
        for i, (image, source) in enumerate(zip(picked, wanted))
    ]
    log.info(
        "frames_extracted",
        extra={"clip": str(path), "frames": total, "source_frames": total_frames},
    )
    return frames


def save_frames(frames: list[Frame], directory: Path) -> Path:
    """Write frames as frame_01.jpg ... so a capture dir can be replayed."""
    directory.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        (directory / f"frame_{frame.position:02d}.jpg").write_bytes(frame.jpeg)
    return directory


def load_frames(directory: Path) -> list[Frame]:
    """Rebuild a frame sequence from a capture directory (replay path)."""
    files = sorted(p for p in directory.glob("frame_*.jpg") if p.is_file())
    if not files:
        raise FrameExtractionError(f"no frame_*.jpg files in {directory}")
    total = len(files)
    return [
        Frame(position=i + 1, total=total, jpeg=path.read_bytes())
        for i, path in enumerate(files)
    ]
