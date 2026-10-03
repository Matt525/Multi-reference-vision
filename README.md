# Multi-Reference Vision

A generalized **YOLOE-26 + ByteTrack** pipeline for detecting arbitrary visual references, tracking each physical item, smoothing its class/state across frames, and counting it when it crosses a configurable line.

It is designed to reproduce the pattern in the example apple-quality system without hard-coding apples. You can pass references for apples, oranges, packages, parts, defects, tools, products, or other visually distinguishable targets.

## What it does

1. **Accepts multiple visual references** grouped as `object/state`.
2. Builds one reference montage and uses **YOLOE-26 visual prompts** to create prompt embeddings.
3. Detects the prompted objects in each frame.
4. Uses **ByteTrack** to maintain an ID for each physical item.
5. Uses a rolling **majority vote** on every track so state/class does not flicker frame to frame.
6. Counts an item only when its tracked centroid crosses a configured line, optionally in one direction only.
7. Writes an annotated MP4, `events.csv`, and `summary.json`.

## Reference folder format

The easiest input format is a folder tree:

```text
references/
  apple/
    normal/
      apple_good_01.jpg
      apple_good_02.jpg
    damaged/
      apple_bad_01.jpg
  orange/
    normal/
      orange_good_01.jpg
    damaged/
      orange_bad_01.jpg
  shipping_box/
    box_01.jpg
```

`shipping_box/*.jpg` becomes `shipping_box::default`. Each image should be a fairly tight crop of the thing you want the model to recognize. Multiple images in one folder become multiple examples for the same logical class/state.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -e .
```

YOLOE-26 requires a recent Ultralytics release; this project asks for `ultralytics>=8.4.0`.

## Configure

Copy `configs/example.yaml` and edit:

```yaml
source: "input.mp4"
reference_root: "references"
output_dir: "runs/my_test"

model:
  weights: "yoloe-26s-seg.pt"
  device: "auto"
  imgsz: 960
  conf: 0.25
  tracker: "bytetrack.yaml"

counting:
  line: [[0.08, 0.65], [0.92, 0.65]]
  direction: "any"
  vote_window: 9
  min_votes: 3
```

Line coordinates are normalized, so `[0.5, 0.5]` means the center of the frame.

## Inspect the references first

```bash
refvision inspect-refs --config configs/example.yaml
```

This scans the folder, prints the generated classes, and creates a reference sheet under the configured output folder.

## Run

```bash
refvision run --config configs/example.yaml
```

Press **Q** or **Esc** to stop the preview window.

Outputs:

```text
runs/my_test/
  annotated.mp4
  events.csv
  summary.json
  .reference_cache/
    reference_sheet.jpg
    visual_prompts.npz
```

## How generalization works

The detector does **not** contain apple-specific code. Each object/state pair becomes a visual-prompt class:

```text
apple::normal
apple::damaged
orange::normal
orange::damaged
bearing::good
bearing::cracked
...
```

YOLOE receives several reference examples at once. ByteTrack then handles identity over time. The counter only acts on a track when it crosses the line. The final class/state used for the count is the majority-voted label for that track.

This means you can change the inspection problem primarily by changing the reference folders instead of rewriting the pipeline.

## Important practical notes

- **Reference quality matters.** Use tightly cropped, representative examples with several viewing angles when possible.
- **Very subtle defects** may need a trained custom classifier rather than one-shot visual prompting. The current architecture makes that upgrade straightforward because detection/tracking/counting are already separated.
- **Fast conveyor motion** may benefit from a smaller model, lower `imgsz`, a higher shutter-speed camera, and a line placed where objects are well separated.
- **Occlusion-heavy scenes** can cause tracker ID switches. ByteTrack is intentionally the default because it is fast and simple; Ultralytics also supports other trackers if you later need stronger association.
- If an object crosses before `min_votes` has been reached, the system stores a pending crossing and commits it once the track has a stable majority label.

## Directional counting

`direction` can be:

- `any`
- `positive_to_negative`
- `negative_to_positive`

The sign is determined by the order of the two line endpoints. If direction matters, run a short clip once and flip the endpoint order or direction setting if the desired flow is reversed.

## Existing components used

- **Ultralytics YOLOE-26 / YOLO26 ecosystem** for visual-prompt object detection.
- **ByteTrack via Ultralytics tracking mode** for persistent object IDs.
- OpenCV for video I/O and annotation.

The custom portion in this repository is the multi-reference catalog, reference-montage builder, prompt preparation, per-track majority voting, directional line crossing, count aggregation, event logging, and rendering pipeline.

## Licensing note

This repository's original glue code is MIT-licensed. Ultralytics is a separate dependency with its own license (AGPL-3.0 for the open-source package at the time this project was created). Review that license if you distribute this as part of a commercial or hosted product.
