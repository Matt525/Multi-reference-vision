from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline
from .references import ReferenceCatalog, build_reference_sheet


def _cmd_run(args) -> int:
    cfg = load_config(args.config)
    summary = run_pipeline(cfg)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_inspect_refs(args) -> int:
    cfg = load_config(args.config)
    catalog = ReferenceCatalog.scan(cfg.reference_root)
    cache_dir = cfg.output_dir / ".reference_cache"
    sheet, prompts = build_reference_sheet(catalog, cache_dir / "reference_sheet.jpg")
    print(f"Reference sheet: {sheet}")
    print(f"Classes ({len(catalog.labels)}):")
    for idx, label in enumerate(catalog.labels):
        examples = sum(1 for e in catalog.examples if e.label == label)
        print(f"  {idx:>3}: {label} ({examples} reference image{'s' if examples != 1 else ''})")
    print(f"Prompt boxes: {len(prompts['bboxes'])}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="refvision")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run detection, tracking, voting, and directional counting")
    run.add_argument("--config", required=True, type=Path)
    run.set_defaults(func=_cmd_run)

    inspect_refs = sub.add_parser("inspect-refs", help="Scan references and build the YOLOE visual prompt sheet")
    inspect_refs.add_argument("--config", required=True, type=Path)
    inspect_refs.set_defaults(func=_cmd_inspect_refs)

    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
