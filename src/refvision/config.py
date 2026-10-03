from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModelConfig:
    weights: str = "yoloe-26s-seg.pt"
    device: str = "auto"
    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.50
    tracker: str = "bytetrack.yaml"


@dataclass(slots=True)
class CountingConfig:
    line: list[list[float]] = field(default_factory=lambda: [[0.08, 0.65], [0.92, 0.65]])
    direction: str = "any"
    vote_window: int = 9
    min_votes: int = 3
    count_once_per_track: bool = True


@dataclass(slots=True)
class RenderConfig:
    show_window: bool = True
    save_video: bool = True
    draw_confidence: bool = True
    draw_track_id: bool = True
    hud_max_rows: int = 12


@dataclass(slots=True)
class AppConfig:
    source: str | int
    reference_root: Path
    output_dir: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    config_path: Path | None = None


def _resolve_source(value: Any, base: Path) -> str | int:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit() and len(text) < 3:
        return int(text)
    if "://" in text:
        return text
    path = Path(text)
    return str(path if path.is_absolute() else (base / path).resolve())


def load_config(path: str | Path) -> AppConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base = path.parent
    if "source" not in raw:
        raise ValueError("Config is missing required key: source")
    if "reference_root" not in raw:
        raise ValueError("Config is missing required key: reference_root")

    reference_root = Path(raw["reference_root"])
    if not reference_root.is_absolute():
        reference_root = (base / reference_root).resolve()

    output_dir = Path(raw.get("output_dir", "runs/output"))
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()

    cfg = AppConfig(
        source=_resolve_source(raw["source"], base),
        reference_root=reference_root,
        output_dir=output_dir,
        model=ModelConfig(**(raw.get("model") or {})),
        counting=CountingConfig(**(raw.get("counting") or {})),
        render=RenderConfig(**(raw.get("render") or {})),
        config_path=path,
    )

    if cfg.counting.direction not in {"any", "positive_to_negative", "negative_to_positive"}:
        raise ValueError("counting.direction must be any, positive_to_negative, or negative_to_positive")
    if len(cfg.counting.line) != 2 or any(len(p) != 2 for p in cfg.counting.line):
        raise ValueError("counting.line must contain exactly two [x, y] points")
    if cfg.counting.vote_window < 1 or cfg.counting.min_votes < 1:
        raise ValueError("vote_window and min_votes must be >= 1")
    if cfg.counting.min_votes > cfg.counting.vote_window:
        raise ValueError("min_votes cannot exceed vote_window")
    return cfg
