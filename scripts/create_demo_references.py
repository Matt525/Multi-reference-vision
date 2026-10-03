"""Creates placeholder colored reference crops so the folder convention is obvious.

These are NOT meaningful model references; replace them with real, tightly cropped examples.
"""
from pathlib import Path
from PIL import Image, ImageDraw

root = Path("references")
items = [
    ("apple", "normal", (80, 180, 80)),
    ("apple", "damaged", (170, 90, 70)),
    ("orange", "normal", (230, 150, 50)),
    ("orange", "damaged", (140, 90, 40)),
]
for obj, state, color in items:
    folder = root / obj / state
    folder.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(im)
    draw.ellipse((48, 48, 208, 208), fill=color)
    im.save(folder / "replace_me.jpg")
print(f"Created demo structure under {root.resolve()}")
