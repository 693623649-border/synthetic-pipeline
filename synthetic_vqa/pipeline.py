from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .adapters import (
    DetectionInput,
    Detector,
    DeterministicMockEvaluator,
    Evaluator,
    OracleDetector,
    score_samples,
    smoke_mock_plan,
)
from .exporters import export_dataset
from .scene import generate_smoke_samples
from .selection import SMOKE_BO3_QUOTA, bo3_counts, select_by_bo3
from .reference import (
    HELDOUT_QUESTION_TYPES,
    QUESTION_TYPES,
    ReferenceBuildConfig,
    build_reference_samples,
)
from .validation import validate_samples, validate_smoke_shape


def run_smoke_pipeline(
    output_dir: Path,
    seed: int = 42,
    detector: Optional[Detector] = None,
    evaluator: Optional[Evaluator] = None,
    asset_root: Optional[Path] = None,
    strict_assets: bool = False,
) -> Dict[str, object]:
    samples = generate_smoke_samples(seed=seed, asset_root=asset_root, strict_assets=strict_assets)
    detector = detector or OracleDetector({sample.source_id: sample.scene for sample in samples})
    evaluator = evaluator or DeterministicMockEvaluator(smoke_mock_plan(samples))
    validate_smoke_shape(samples)

    for sample in samples:
        sample.detections = detector.detect(DetectionInput(image_bytes=sample.image_bytes, source_id=sample.source_id))
    validate_samples(samples, require_scores=False, detector_name=detector.name)

    score_samples(samples, evaluator)
    validate_samples(samples, require_scores=True, detector_name=detector.name)

    selected = select_by_bo3(samples, SMOKE_BO3_QUOTA, seed=seed)
    if bo3_counts(selected) != SMOKE_BO3_QUOTA:
        raise RuntimeError("Selected BO3 distribution does not match the smoke quota")
    final_validation = validate_samples(selected, require_scores=True, detector_name=detector.name)
    validate_smoke_shape(selected)
    return export_dataset(
        selected,
        Path(output_dir),
        final_validation,
        detector_name=detector.name,
        evaluator_name=evaluator.name,
        evaluation_kind=evaluator.evaluation_kind,
    )


def run_reference_pipeline(
    output_dir: Path,
    *,
    config: Optional[ReferenceBuildConfig] = None,
    detector: Optional[Detector] = None,
    asset_root: Optional[Path] = None,
    strict_assets: bool = False,
) -> Dict[str, object]:
    """Run the complete Zoo-Bus-compatible construction pipeline.

    Unlike ``run_smoke_pipeline``, this does not invent BO3/model-API scores.
    Its answers are detector-and-geometry-derived labels intended for training
    data construction, and the output audit records exactly why candidates were
    rejected or removed during split/balancing.
    """

    config = config or ReferenceBuildConfig()
    samples, audit, detector_name = build_reference_samples(
        config,
        asset_root=asset_root,
        strict_assets=strict_assets,
        detector=detector,
    )
    output_dir = Path(output_dir)
    summary = export_dataset(
        samples,
        output_dir,
        audit["validation"],
        detector_name=detector_name,
        evaluator_name="not-run",
        evaluation_kind="not_scored",
        allow_unscored=True,
        extra_summary={
            "pipeline": {
                "kind": "zoo_bus_compatible_reference",
                "stages": [
                    "scene_parameter_sampling",
                    "2d_sprite_compositing",
                    "object_detection",
                    "detection_and_geometry_stability_filtering",
                    "question_template_answering",
                    "image_text_qa_generation",
                    "source_id_split",
                    "question_type_and_answer_rebalancing",
                ],
                "question_type_catalog": list(QUESTION_TYPES),
                "heldout_question_types": sorted(HELDOUT_QUESTION_TYPES),
                "source_id_policy": "all QA rows of one source_id share one split; held-out types are removed from train",
            },
            "balance": audit["balance"],
            "detector_provenance": audit["detector_provenance"],
            "split_policy": "deterministic source_id-level train/evaluation/test split (seed 42 by default); no source image can cross splits",
        },
    )
    audit_path = output_dir / "construction_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evaluation_inputs_path = output_dir / "evaluation_inputs.jsonl"
    with evaluation_inputs_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            if sample.split not in {"evaluation", "test"}:
                continue
            # This is the public/blind-evaluation view.  Labels, proofs,
            # annotations and generator state remain in the audit artifact.
            record = {
                "id": sample.id,
                "source_id": sample.source_id,
                "split": sample.split,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "images/%s" % sample.source_id},
                            {"type": "text", "text": sample.question},
                        ],
                    }
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary["artifacts"]["construction_audit"] = str(audit_path)
    summary["artifacts"]["evaluation_inputs"] = str(evaluation_inputs_path)
    summary["audit_visibility"] = {
        "dataset_parquet": "restricted audit/training artifact; it contains labels and geometry evidence",
        "evaluation_inputs_jsonl": "public/blind-evaluation inputs without answers, annotations, proof, or scene state",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
