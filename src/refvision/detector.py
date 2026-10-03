from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .references import ReferenceCatalog, build_reference_sheet


class YOLOEReferenceDetector:
    """YOLOE-26 detector primed from multiple visual reference examples."""

    def __init__(self, cfg: ModelConfig, catalog: ReferenceCatalog, cache_dir: str | Path):
        self.cfg = cfg
        self.catalog = catalog
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    @staticmethod
    def _device_arg(device: str):
        return None if device in {"", "auto", "none", "None"} else device

    def prepare(self) -> None:
        try:
            from ultralytics import YOLOE
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics with YOLOE support is required. Install the project with: pip install -e ."
            ) from exc

        sheet_path, visual_prompts = build_reference_sheet(
            self.catalog, self.cache_dir / "reference_sheet.jpg"
        )
        embedding_path = self.cache_dir / "visual_prompts.npz"

        # Prime a YOLOE-26 model with all reference examples at once. Each object/state
        # pair is a class, and repeated examples of that pair share the same class ID.
        priming_model = YOLOE(self.cfg.weights)
        priming_model.predict(
            source=str(sheet_path),
            refer_image=str(sheet_path),
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self._device_arg(self.cfg.device),
            verbose=False,
            save=False,
        )
        priming_model.save_prompt_embeddings(str(embedding_path))

        self._model = YOLOE(self.cfg.weights)
        self._model.load_prompt_embeddings(str(embedding_path))

    def track(self, frame: np.ndarray):
        if self._model is None:
            self.prepare()
        return self._model.track(
            source=frame,
            persist=True,
            tracker=self.cfg.tracker,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self._device_arg(self.cfg.device),
            verbose=False,
            save=False,
        )
