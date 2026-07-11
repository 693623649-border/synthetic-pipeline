from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
from datasets import Dataset, Image as DatasetImage

from .models import Sample
from .selection import bo3_counts


ANNOTATION_TYPE = pa.struct(
    [
        pa.field("area", pa.float64()),
        pa.field("bbox", pa.list_(pa.float64())),
        pa.field("category", pa.string()),
        pa.field("category_id", pa.int64()),
        pa.field("iscrowd", pa.int64()),
        pa.field("score", pa.float64()),
    ]
)

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
        pa.field("annotations", pa.list_(ANNOTATION_TYPE)),
        pa.field("detections", pa.list_(ANNOTATION_TYPE)),
        pa.field("question", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("question_type", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("id", pa.int64()),
        pa.field("split", pa.string()),
        pa.field("capability", pa.string()),
        pa.field("group_id", pa.string()),
        pa.field("scene_family_id", pa.string()),
        pa.field("variant_role", pa.string()),
        pa.field("scene_state_json", pa.string()),
        pa.field("difficulty_axes_json", pa.string()),
        pa.field("proof_json", pa.string()),
        pa.field("bo3", pa.int64()),
        pa.field("attempts_json", pa.string()),
        pa.field("seed", pa.int64()),
        pa.field("generator_version", pa.string()),
        pa.field("detector_backend", pa.string()),
        pa.field("evaluation_kind", pa.string()),
    ]
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parquet_row(
    sample: Sample, *, detector_name: str, evaluation_kind: str, allow_unscored: bool = False
) -> Dict[str, object]:
    if sample.bo3 is None and not allow_unscored:
        raise ValueError("Cannot export an unscored sample")
    return {
        "image": {"bytes": sample.image_bytes, "path": None},
        "annotations": sample.annotations,
        "detections": sample.detections,
        "question": sample.question,
        "answer": sample.answer,
        "question_type": sample.question_type,
        "source_id": sample.source_id,
        "id": sample.id,
        "split": sample.split,
        "capability": sample.capability,
        "group_id": sample.group_id,
        "scene_family_id": sample.scene_family_id,
        "variant_role": sample.variant_role,
        "scene_state_json": _json(sample.scene.to_dict()),
        "difficulty_axes_json": _json(sample.difficulty_axes),
        "proof_json": _json(sample.proof),
        "bo3": sample.bo3,
        "attempts_json": _json([item.to_dict() for item in sample.attempts]),
        "seed": sample.seed,
        "generator_version": sample.generator_version,
        "detector_backend": detector_name,
        "evaluation_kind": evaluation_kind,
    }


def messages_record(sample: Sample, relative_image_path: str) -> Dict[str, object]:
    return {
        "id": sample.id,
        "source_id": sample.source_id,
        "split": sample.split,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": relative_image_path},
                    {"type": "text", "text": sample.question},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": sample.answer}]},
        ],
    }


def _safe_image_path(image_dir: Path, source_id: str) -> Path:
    relative = Path(source_id)
    if relative.is_absolute() or relative.name != source_id or ".." in relative.parts:
        raise ValueError("Unsafe source_id: %s" % source_id)
    root = image_dir.resolve()
    candidate = (image_dir / relative.name).resolve()
    if candidate.parent != root:
        raise ValueError("Image path escapes output directory: %s" % source_id)
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_dataset(
    samples: List[Sample],
    output_dir: Path,
    validation: Dict[str, object],
    *,
    detector_name: str,
    evaluator_name: str,
    evaluation_kind: str,
    allow_unscored: bool = False,
    extra_summary: Dict[str, object] = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    written_images: Dict[str, bytes] = {}
    for sample in samples:
        existing = written_images.get(sample.source_id)
        if existing is not None and existing != sample.image_bytes:
            raise ValueError("Rows with source_id %s have different image bytes" % sample.source_id)
        if existing is None:
            _safe_image_path(image_dir, sample.source_id).write_bytes(sample.image_bytes)
            written_images[sample.source_id] = sample.image_bytes

    parquet_path = output_dir / "dataset.parquet"
    table = pa.Table.from_pylist(
        [
            parquet_row(
                sample,
                detector_name=detector_name,
                evaluation_kind=evaluation_kind,
                allow_unscored=allow_unscored,
            )
            for sample in samples
        ],
        schema=PARQUET_SCHEMA,
    )
    dataset = Dataset(table).cast_column("image", DatasetImage(decode=True))
    dataset.to_parquet(str(parquet_path))

    jsonl_path = output_dir / "messages.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = messages_record(sample, "images/%s" % sample.source_id)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "status": "ok",
        "rows": len(samples),
        "capabilities": dict(sorted(Counter(item.capability for item in samples).items())),
        "question_types": dict(sorted(Counter(item.question_type for item in samples).items())),
        "splits": dict(sorted(Counter(item.split for item in samples).items())),
        "variant_roles": dict(sorted(Counter(item.variant_role for item in samples).items())),
        "rendering": {
            "renderer": samples[0].scene.metadata.get("renderer") if samples else None,
            "reference_assets": all(bool(item.scene.metadata.get("reference_assets")) for item in samples),
            "asset_roots": sorted({root for root in (item.scene.metadata.get("asset_root") for item in samples) if root is not None}),
            "native_size": samples[0].scene.metadata.get("native_size") if samples else None,
            "output_size": samples[0].scene.metadata.get("output_size") if samples else None,
        },
        "bo3": None if allow_unscored else {str(key): value for key, value in bo3_counts(samples).items()},
        "source_images": len(written_images),
        "validation": validation,
        "artifacts": {
            "parquet": str(parquet_path),
            "messages_jsonl": str(jsonl_path),
            "images": str(image_dir),
            "summary": str(output_dir / "summary.json"),
        },
        "checksums": {
            "parquet_sha256": _sha256_file(parquet_path),
            "messages_jsonl_sha256": _sha256_file(jsonl_path),
        },
        "adapters": {
            "detector": detector_name,
            "evaluator": evaluator_name,
            "evaluation_kind": evaluation_kind,
            "is_real_evaluation": evaluation_kind == "model_api",
            "yolo_status": "configured_by_caller" if detector_name == "yolo" else "not_used_in_this_export",
            "model_api_status": "reserved_for_development_machine",
        },
        "split_policy": "single smoke split; not an unbiased benchmark split",
    }
    if extra_summary:
        summary.update(extra_summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
