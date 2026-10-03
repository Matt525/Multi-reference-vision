from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True, slots=True)
class ReferenceExample:
    object_name: str
    state_name: str
    image_path: Path

    @property
    def label(self) -> str:
        return f"{self.object_name}::{self.state_name}"


@dataclass(slots=True)
class ReferenceCatalog:
    examples: list[ReferenceExample]
    labels: list[str]
    label_to_id: dict[str, int]

    @classmethod
    def scan(cls, root: str | Path) -> "ReferenceCatalog":
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Reference folder does not exist: {root}")

        examples: list[ReferenceExample] = []

        # Direct files: each file stem becomes its own object with default state.
        for p in _images(root):
            examples.append(ReferenceExample(p.stem, "default", p))

        for object_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            direct = list(_images(object_dir))
            for p in direct:
                examples.append(ReferenceExample(object_dir.name, "default", p))

            state_dirs = sorted(p for p in object_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
            for state_dir in state_dirs:
                for p in _images(state_dir):
                    examples.append(ReferenceExample(object_dir.name, state_dir.name, p))

        if not examples:
            raise ValueError(
                f"No reference images found under {root}. Expected image files or folders such as "
                "references/apple/normal/*.jpg"
            )

        labels = sorted({e.label for e in examples})
        return cls(examples=examples, labels=labels, label_to_id={label: i for i, label in enumerate(labels)})

    def decode(self, class_id: int) -> tuple[str, str, str]:
        label = self.labels[int(class_id)]
        object_name, state_name = label.split("::", 1)
        return object_name, state_name, label


def _images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


def build_reference_sheet(
    catalog: ReferenceCatalog,
    destination: str | Path,
    tile_size: int = 320,
    columns: int = 4,
) -> tuple[Path, dict[str, np.ndarray]]:
    """Build one montage and visual-prompt boxes for all supplied references.

    Multiple examples with the same object/state label share the same class ID. This lets
    YOLOE build a single visual concept from several reference crops.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    n = len(catalog.examples)
    cols = max(1, min(columns, n))
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_size, rows * tile_size), (114, 114, 114))

    boxes: list[list[float]] = []
    classes: list[int] = []

    for index, example in enumerate(catalog.examples):
        row, col = divmod(index, cols)
        with Image.open(example.image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            # Fit inside the tile without distortion and preserve the exact content box.
            max_inner = int(tile_size * 0.90)
            scale = min(max_inner / im.width, max_inner / im.height)
            width = max(1, int(round(im.width * scale)))
            height = max(1, int(round(im.height * scale)))
            resized = im.resize((width, height), Image.Resampling.LANCZOS)

        x0 = col * tile_size + (tile_size - width) // 2
        y0 = row * tile_size + (tile_size - height) // 2
        sheet.paste(resized, (x0, y0))
        boxes.append([float(x0), float(y0), float(x0 + width), float(y0 + height)])
        classes.append(catalog.label_to_id[example.label])

    sheet.save(destination, quality=95)
    prompts = {
        "bboxes": np.asarray(boxes, dtype=np.float32),
        "cls": np.asarray(classes, dtype=np.int64),
    }
    return destination, prompts
