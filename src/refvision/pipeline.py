from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2

from .config import AppConfig
from .detector import YOLOEReferenceDetector
from .references import ReferenceCatalog
from .render import draw_frame
from .tracking import Detection, TrackAggregator


def _open_capture(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return cap


def _make_writer(path: Path, fps: float, width: int, height: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps if fps > 0 else 30.0, (width, height))


def _parse_result(result, catalog: ReferenceCatalog) -> list[tuple[Detection, str]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    ids = boxes.id.detach().cpu().numpy().astype(int)
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy()

    parsed: list[tuple[Detection, str]] = []
    for box, track_id, class_id, confidence in zip(xyxy, ids, cls, conf):
        if class_id < 0 or class_id >= len(catalog.labels):
            continue
        det = Detection(
            track_id=int(track_id),
            class_id=int(class_id),
            confidence=float(confidence),
            xyxy=tuple(map(float, box)),
        )
        parsed.append((det, catalog.labels[class_id]))
    return parsed


def run_pipeline(cfg: AppConfig) -> dict:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cfg.output_dir / ".reference_cache"
    catalog = ReferenceCatalog.scan(cfg.reference_root)

    detector = YOLOEReferenceDetector(cfg.model, catalog, cache_dir)
    detector.prepare()

    cap = _open_capture(cfg.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    writer = None
    output_video = cfg.output_dir / "annotated.mp4"
    if cfg.render.save_video:
        writer = _make_writer(output_video, fps, width, height)

    aggregator = TrackAggregator(
        line_norm=cfg.counting.line,
        direction=cfg.counting.direction,
        vote_window=cfg.counting.vote_window,
        min_votes=cfg.counting.min_votes,
        count_once_per_track=cfg.counting.count_once_per_track,
    )

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            result_list = detector.track(frame)
            result = result_list[0]
            parsed = _parse_result(result, catalog)

            aggregator.process(frame_index, (width, height), parsed)

            render_detections = []
            for det, raw_label in parsed:
                stable_label, _ = aggregator.votes.stable(det.track_id)
                render_detections.append((det, raw_label, stable_label))

            annotated = draw_frame(
                frame,
                render_detections,
                aggregator.counts,
                aggregator.counter.pixel_line(width, height),
                draw_confidence=cfg.render.draw_confidence,
                draw_track_id=cfg.render.draw_track_id,
                hud_max_rows=cfg.render.hud_max_rows,
            )

            if writer is not None:
                writer.write(annotated)

            if cfg.render.show_window:
                cv2.imshow("Multi-Reference Vision", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if cfg.render.show_window:
            cv2.destroyAllWindows()

    events_path = cfg.output_dir / "events.csv"
    with events_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "track_id",
                "label",
                "object_name",
                "state_name",
                "direction",
                "x",
                "y",
                "confidence",
            ],
        )
        writer_csv.writeheader()
        for event in aggregator.events:
            writer_csv.writerow(asdict(event))

    summary = {
        "frames_processed": frame_index,
        "counts": dict(aggregator.counts),
        "events": [asdict(e) for e in aggregator.events],
        "labels": catalog.labels,
        "output_video": str(output_video) if cfg.render.save_video else None,
        "events_csv": str(events_path),
    }
    summary_path = cfg.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
