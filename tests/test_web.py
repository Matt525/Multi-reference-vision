from pathlib import Path

from PIL import Image

from refvision.web import _build_reference_tree, _label_parts


def test_label_parts_supports_object_and_state():
    assert _label_parts("apple::damaged") == ("apple", "damaged")
    assert _label_parts("shipping box") == ("shipping box", "default")


def test_reference_uploads_are_normalized_to_jpeg(tmp_path: Path):
    source = tmp_path / "phone.png"
    Image.new("RGB", (20, 20), "white").save(source)

    root = tmp_path / "refs"
    labels = _build_reference_tree(root, [("apple::normal", [str(source)])])

    assert labels == ["apple::normal"]
    created = list((root / "apple" / "normal").glob("*.jpg"))
    assert len(created) == 1
