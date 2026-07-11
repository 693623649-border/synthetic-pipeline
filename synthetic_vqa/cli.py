from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .adapters import DetectionReplay, YoloDetector
from .pipeline import run_reference_pipeline, run_smoke_pipeline
from .reference import ReferenceBuildConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic VQA smoke pipeline")
    subparsers = parser.add_subparsers(dest="command")
    smoke = subparsers.add_parser("smoke", help="Generate and validate the 20-row local smoke dataset")
    smoke.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    smoke.add_argument("--seed", type=int, default=42)
    smoke.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Zoo-Bus scene_gen/src asset directory containing background.jpeg and sprites",
    )
    smoke.add_argument(
        "--strict-assets",
        action="store_true",
        help="Fail when --asset-root is missing instead of using the local compatibility pack",
    )
    build = subparsers.add_parser(
        "build",
        help="Run the full Zoo-Bus-compatible sampling, rendering, detection, filtering, QA, split, and balancing pipeline",
    )
    build.add_argument("--output", type=Path, default=Path("artifacts/reference"))
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--num-scenes", type=int, default=120)
    build.add_argument("--per-answer", type=int, default=8, help="Maximum retained rows per split/question-type/answer; 0 means unlimited")
    build.add_argument("--per-question-type", type=int, default=24, help="Maximum retained rows per split/question type; 0 means unlimited")
    build.add_argument("--max-rows-per-source", type=int, default=29)
    build.add_argument("--strict-balance", action="store_true", help="Fail instead of exporting when any requested balance bucket has a shortfall")
    build.add_argument("--asset-root", type=Path, default=None, help="Zoo-Bus scene_gen/src asset directory")
    build.add_argument("--strict-assets", action="store_true", help="Require supplied reference assets instead of compatibility sprites")
    build.add_argument(
        "--detector",
        choices=("ideal", "replay", "yolo"),
        default="ideal",
        help="ideal uses generator-matched detections locally; replay reads prior detector JSONL; yolo runs supplied Ultralytics weights",
    )
    build.add_argument("--detection-replay", type=Path, default=None, help="JSONL with source_id and detections for --detector replay")
    build.add_argument("--yolo-weights", type=Path, default=None, help="Local Ultralytics YOLO weight file for --detector yolo; never downloaded by this CLI")
    build.add_argument("--yolo-confidence", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "smoke"
    if command == "smoke":
        summary = run_smoke_pipeline(
            output_dir=getattr(args, "output", Path("artifacts/smoke")),
            seed=getattr(args, "seed", 42),
            asset_root=getattr(args, "asset_root", None),
            strict_assets=getattr(args, "strict_assets", False),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if command == "build":
        if args.detector == "replay":
            if args.detection_replay is None:
                parser.error("--detector replay requires --detection-replay")
            detector = DetectionReplay(args.detection_replay)
        elif args.detector == "yolo":
            detector = YoloDetector("" if args.yolo_weights is None else str(args.yolo_weights), confidence=args.yolo_confidence)
        else:
            detector = None
        config = ReferenceBuildConfig(
            num_scenes=args.num_scenes,
            seed=args.seed,
            per_answer=args.per_answer,
            per_question_type=args.per_question_type,
            max_rows_per_source=args.max_rows_per_source,
            strict_balance=args.strict_balance,
        )
        summary = run_reference_pipeline(
            args.output,
            config=config,
            detector=detector,
            asset_root=args.asset_root,
            strict_assets=args.strict_assets,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    parser.error("Unknown command: %s" % command)
    return 2
