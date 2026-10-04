from __future__ import annotations

import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Iterable

from .config import AppConfig, CountingConfig, ModelConfig, RenderConfig
from .pipeline import run_pipeline

MAX_REFERENCE_GROUPS = 4


def _label_parts(label: str) -> tuple[str, str]:
    """Turn `object::state` into safe cross-platform folder names."""
    label = (label or "").strip()
    if not label:
        raise ValueError("Each reference group with images needs a label.")

    if "::" in label:
        object_name, state_name = (part.strip() for part in label.split("::", 1))
    else:
        object_name, state_name = label, "default"

    if not object_name or not state_name:
        raise ValueError(f"Invalid label {label!r}. Use `object` or `object::state`.")

    def clean(value: str) -> str:
        # Windows-safe while keeping ordinary spaces and punctuation readable.
        value = re.sub(r'[<>:"/\\|?*]+', "_", value).strip(" .")
        value = value.replace("..", "_")
        return value[:80] or "unnamed"

    return clean(object_name), clean(state_name)


def _as_paths(files) -> list[Path]:
    if not files:
        return []
    if isinstance(files, (str, Path)):
        files = [files]
    paths: list[Path] = []
    for item in files:
        value = getattr(item, "name", item)
        if value:
            paths.append(Path(str(value)))
    return paths


def _build_reference_tree(reference_root: Path, groups: Iterable[tuple[str, object]]) -> list[str]:
    labels: list[str] = []
    image_index = 0
    for label, files in groups:
        paths = _as_paths(files)
        if not paths:
            continue
        object_name, state_name = _label_parts(label)
        target_dir = reference_root / object_name / state_name
        target_dir.mkdir(parents=True, exist_ok=True)
        labels.append(f"{object_name}::{state_name}")

        for source in paths:
            if not source.exists():
                raise FileNotFoundError(f"Uploaded reference file is missing: {source}")

            # Normalize phone photos to oriented RGB JPEGs. This prevents HEIC/EXIF
            # surprises when the reference montage is built later.
            try:
                try:
                    from pillow_heif import register_heif_opener

                    register_heif_opener()
                except ImportError:
                    pass

                from PIL import Image, ImageOps

                with Image.open(source) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    destination = target_dir / f"ref_{image_index:03d}.jpg"
                    image.save(destination, "JPEG", quality=95)
            except Exception as exc:
                raise ValueError(
                    f"Could not read reference image {source.name}. "
                    'Install the web extras with: pip install -e ".[web]"'
                ) from exc
            image_index += 1

    if not labels:
        raise ValueError("Add at least one labeled reference image before analyzing.")
    return sorted(set(labels))


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            'The web UI needs imageio-ffmpeg. Install it with: pip install -e ".[web]"'
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _transcode_h264(source: str | Path, destination: str | Path) -> Path:
    """Normalize iPhone/MOV/HEVC input and make output Safari-friendly."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        _ffmpeg_exe(),
        "-y",
        "-i",
        str(source),
        "-map_metadata",
        "0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not destination.exists():
        tail = (completed.stderr or "")[-1800:]
        raise RuntimeError(f"Video conversion failed. ffmpeg said:\n{tail}")
    return destination


def analyze_video(video_path, line_percent, confidence, *reference_values):
    """Gradio callback. Reference values arrive as label/files pairs."""
    if not video_path:
        raise ValueError("Upload a short test video first.")

    if len(reference_values) != MAX_REFERENCE_GROUPS * 2:
        raise ValueError("Reference group inputs are incomplete.")

    groups = [
        (reference_values[i], reference_values[i + 1])
        for i in range(0, len(reference_values), 2)
    ]

    job_dir = Path("runs") / "web" / uuid.uuid4().hex[:12]
    reference_root = job_dir / "references"
    output_dir = job_dir / "output"
    job_dir.mkdir(parents=True, exist_ok=True)

    labels = _build_reference_tree(reference_root, groups)
    normalized_video = _transcode_h264(video_path, job_dir / "input.mp4")

    y = max(0.05, min(0.95, float(line_percent) / 100.0))
    cfg = AppConfig(
        source=str(normalized_video.resolve()),
        reference_root=reference_root.resolve(),
        output_dir=output_dir.resolve(),
        model=ModelConfig(
            weights="yoloe-26s-seg.pt",
            device="auto",
            imgsz=640,
            conf=float(confidence),
            iou=0.50,
            tracker="bytetrack.yaml",
        ),
        counting=CountingConfig(
            line=[[0.05, y], [0.95, y]],
            direction="any",
            vote_window=7,
            min_votes=3,
            count_once_per_track=True,
        ),
        render=RenderConfig(
            show_window=False,
            save_video=True,
            draw_confidence=True,
            draw_track_id=True,
            hud_max_rows=10,
        ),
    )

    summary = run_pipeline(cfg)
    raw_output = Path(summary["output_video"])
    iphone_output = _transcode_h264(raw_output, output_dir / "annotated_iphone.mp4")

    counts = summary.get("counts", {})
    rows = [[label, int(counts.get(label, 0))] for label in summary.get("labels", labels)]
    total = sum(row[1] for row in rows)
    status = (
        f"Done — processed {summary.get('frames_processed', 0)} frames and counted "
        f"{total} crossing{'s' if total != 1 else ''}."
    )
    return status, str(iphone_output), rows, summary.get("events_csv")


def build_app():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError('Install the phone test UI with: pip install -e ".[web]"') from exc

    with gr.Blocks(title="Multi-Reference Vision Test") as demo:
        gr.Markdown(
            "# Multi-Reference Vision Test\n"
            "Upload reference images, upload a short phone video, then analyze it. "
            "Use labels like `apple::normal`, `apple::damaged`, `box::good`, etc."
        )

        reference_inputs = []
        for index in range(MAX_REFERENCE_GROUPS):
            with gr.Row():
                label = gr.Textbox(
                    label=f"Reference {index + 1} label",
                    placeholder="apple::normal" if index == 0 else "Leave blank if unused",
                )
                files = gr.File(
                    label=f"Reference {index + 1} images",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
            reference_inputs.extend([label, files])

        gr.Markdown("### Test video")
        video = gr.Video(
            label="Upload a short video from your iPhone",
            sources=["upload"],
            format="mp4",
        )
        line_percent = gr.Slider(
            minimum=10,
            maximum=90,
            value=65,
            step=1,
            label="Horizontal counting line (% down from top)",
        )
        confidence = gr.Slider(
            minimum=0.10,
            maximum=0.80,
            value=0.25,
            step=0.05,
            label="Detection confidence",
        )
        analyze = gr.Button("Analyze video", variant="primary")

        status = gr.Markdown()
        output_video = gr.Video(label="Annotated result")
        counts = gr.Dataframe(
            headers=["Class / state", "Count"],
            datatype=["str", "number"],
            interactive=False,
            label="Counts",
        )
        events = gr.File(label="Event log (CSV)")

        analyze.click(
            fn=analyze_video,
            inputs=[video, line_percent, confidence, *reference_inputs],
            outputs=[status, output_video, counts, events],
        )

        gr.Markdown(
            "**Tip:** For the first test, use a 5–20 second clip with objects moving clearly "
            "across the selected counting line. Each ByteTrack ID is counted at most once."
        )

    return demo


def launch(share: bool = False) -> None:
    demo = build_app()
    user = os.getenv("REFVISION_USER")
    password = os.getenv("REFVISION_PASSWORD")
    auth = (user, password) if user and password else None

    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        share=share,
        auth=auth,
        show_error=True,
    )
