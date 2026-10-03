from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from math import hypot
from typing import Deque


@dataclass(slots=True)
class Detection:
    track_id: int
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(slots=True)
class CountEvent:
    frame_index: int
    track_id: int
    label: str
    object_name: str
    state_name: str
    direction: str
    x: float
    y: float
    confidence: float


class MajorityVote:
    def __init__(self, window: int = 9, min_votes: int = 3):
        self.window = window
        self.min_votes = min_votes
        self._labels: dict[int, Deque[str]] = defaultdict(lambda: deque(maxlen=self.window))
        self._conf: dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, track_id: int, label: str, confidence: float) -> None:
        self._labels[track_id].append(label)
        self._conf[track_id].append(float(confidence))

    def stable(self, track_id: int) -> tuple[str | None, float]:
        labels = self._labels.get(track_id)
        if not labels or len(labels) < self.min_votes:
            return None, 0.0
        counts = Counter(labels)
        label, votes = counts.most_common(1)[0]
        if votes < self.min_votes:
            return None, 0.0
        matching_conf = [
            c for l, c in zip(self._labels[track_id], self._conf[track_id]) if l == label
        ]
        avg_conf = sum(matching_conf) / len(matching_conf) if matching_conf else 0.0
        return label, avg_conf


def _side(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


class DirectionalLineCounter:
    """Counts a track only when its centroid crosses a configured line."""

    def __init__(
        self,
        line_norm: list[list[float]],
        direction: str = "any",
        count_once_per_track: bool = True,
    ):
        self.line_norm = line_norm
        self.direction = direction
        self.count_once_per_track = count_once_per_track
        self._previous_centroids: dict[int, tuple[float, float]] = {}
        self._counted_tracks: set[int] = set()

    def pixel_line(self, width: int, height: int) -> tuple[tuple[float, float], tuple[float, float]]:
        a, b = self.line_norm
        return (a[0] * width, a[1] * height), (b[0] * width, b[1] * height)

    def update(
        self,
        track_id: int,
        centroid: tuple[float, float],
        frame_size: tuple[int, int],
    ) -> str | None:
        width, height = frame_size
        a, b = self.pixel_line(width, height)
        previous = self._previous_centroids.get(track_id)
        self._previous_centroids[track_id] = centroid
        if previous is None:
            return None
        if self.count_once_per_track and track_id in self._counted_tracks:
            return None

        prev_side = _side(previous, a, b)
        curr_side = _side(centroid, a, b)

        # Ignore points exactly on the line until they actually transition sides.
        if prev_side == 0 or curr_side == 0 or (prev_side > 0) == (curr_side > 0):
            return None

        crossing = "positive_to_negative" if prev_side > 0 > curr_side else "negative_to_positive"
        if self.direction != "any" and crossing != self.direction:
            return None

        if self.count_once_per_track:
            self._counted_tracks.add(track_id)
        return crossing


class TrackAggregator:
    def __init__(
        self,
        line_norm: list[list[float]],
        direction: str,
        vote_window: int,
        min_votes: int,
        count_once_per_track: bool,
    ):
        self.votes = MajorityVote(vote_window, min_votes)
        self.counter = DirectionalLineCounter(line_norm, direction, count_once_per_track)
        self.counts: Counter[str] = Counter()
        self.events: list[CountEvent] = []
        self._pending: dict[int, tuple[int, str, tuple[float, float]]] = {}

    @staticmethod
    def split_label(label: str) -> tuple[str, str]:
        if "::" in label:
            return tuple(label.split("::", 1))  # type: ignore[return-value]
        return label, "default"

    def process(
        self,
        frame_index: int,
        frame_size: tuple[int, int],
        detections: list[tuple[Detection, str]],
    ) -> list[CountEvent]:
        new_events: list[CountEvent] = []

        for det, raw_label in detections:
            self.votes.update(det.track_id, raw_label, det.confidence)
            stable_label, stable_conf = self.votes.stable(det.track_id)
            crossing = self.counter.update(det.track_id, det.centroid, frame_size)

            if crossing:
                if stable_label is None:
                    self._pending[det.track_id] = (frame_index, crossing, det.centroid)
                else:
                    new_events.append(
                        self._commit(frame_index, det.track_id, stable_label, crossing, det.centroid, stable_conf)
                    )

            if det.track_id in self._pending and stable_label is not None:
                pending_frame, pending_direction, pending_centroid = self._pending.pop(det.track_id)
                new_events.append(
                    self._commit(
                        pending_frame,
                        det.track_id,
                        stable_label,
                        pending_direction,
                        pending_centroid,
                        stable_conf,
                    )
                )

        return new_events

    def _commit(
        self,
        frame_index: int,
        track_id: int,
        label: str,
        direction: str,
        centroid: tuple[float, float],
        confidence: float,
    ) -> CountEvent:
        object_name, state_name = self.split_label(label)
        self.counts[label] += 1
        event = CountEvent(
            frame_index=frame_index,
            track_id=track_id,
            label=label,
            object_name=object_name,
            state_name=state_name,
            direction=direction,
            x=centroid[0],
            y=centroid[1],
            confidence=confidence,
        )
        self.events.append(event)
        return event
