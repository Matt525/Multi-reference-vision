from pathlib import Path

from PIL import Image

from refvision.references import ReferenceCatalog, build_reference_sheet


def _img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), "white").save(path)


def test_reference_catalog_two_level_and_one_level(tmp_path):
    _img(tmp_path / "apple" / "normal" / "a.jpg")
    _img(tmp_path / "apple" / "damaged" / "b.jpg")
    _img(tmp_path / "box" / "box1.jpg")

    catalog = ReferenceCatalog.scan(tmp_path)
    assert set(catalog.labels) == {"apple::normal", "apple::damaged", "box::default"}

    sheet, prompts = build_reference_sheet(catalog, tmp_path / "sheet.jpg", tile_size=64, columns=2)
    assert sheet.exists()
    assert prompts["bboxes"].shape == (3, 4)
    assert prompts["cls"].shape == (3,)
