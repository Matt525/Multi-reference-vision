from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from .tracking import Detection


def _color(key: str) -> tuple[int, int, int]:
    # Deterministic BGR color without external palette dependencies.
    h = abs(hash(key))
    return (70 + h % 160, 70 + (h // 17) % 160, 70 + (h // 37) % 160)


def draw_frame(
    frame: np.ndarray,
    detections: list[tuple[Detection, str, str | None]],
    counts: Counter[str],
    line_px: tuple[tuple[float, float], tuple[float, float]],
    draw_confidence: bool = True,
    draw_track_id: bool = True,
    hud_max_rows: int = 12,
) -> np.ndarray:
    out = frame.copy()
    a, b = line_px
    cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 255, 255), 2)

    for det, raw_label, stable_label in detections:
        label = stable_label or raw_label
        x1, y1, x2, y2 = map(int, det.xyxy)
        color = _color(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        parts = [label.replace("::", "/")]
        if draw_track_id:
            parts.append(f"#{det.track_id}")
        if draw_confidence:
            parts.append(f"{det.confidence:.2f}")
        text = " ".join(parts)
        cv2.putText(out, text, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

    # Compact summary HUD.
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:hud_max_rows]
    panel_h = 42 + 24 * len(rows)
    panel_w = 330
    overlay = out.copy()
    cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.putText(out, "COUNTED", (26, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    y = 66
    for label, count in rows:
        text = f"{label.replace('::', '/'):24s} {count:>4d}"
        cv2.putText(out, text, (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)
        y += 24
    return out
