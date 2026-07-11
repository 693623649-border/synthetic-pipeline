from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set

from .adapters import normalize_answer
from .models import Sample, annotation_from_object
from .qa import CAPABILITIES, RuleQAGenerator
from .scene import VARIANT_ROLES


class DatasetValidationError(ValueError):
    pass


def _canonical_scene(sample: Sample) -> str:
    value = sample.scene.to_dict()
    value["metadata"] = {key: val for key, val in value["metadata"].items() if key != "counterfactual_change"}
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _bbox_iou(left: List[float], right: List[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    lx, ly, lw, lh = map(float, left)
    rx, ry, rw, rh = map(float, right)
    x0, y0 = max(lx, rx), max(ly, ry)
    x1, y1 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = lw * lh + rw * rh - intersection
    return 0.0 if union <= 0 else intersection / union


def _validate_detection_sanity(sample: Sample, errors: List[str]) -> None:
    for index, detection in enumerate(sample.detections):
        try:
            x, y, width, height = map(float, detection["bbox"])
            score = float(detection["score"])
            if width <= 0 or height <= 0 or x < 0 or y < 0:
                raise ValueError("invalid dimensions")
            if x + width > sample.scene.width or y + height > sample.scene.height:
                raise ValueError("out of bounds")
            if not 0.0 <= score <= 1.0:
                raise ValueError("invalid score")
            if not isinstance(detection["category"], str):
                raise ValueError("invalid category")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append("%s detection %d is malformed: %s" % (sample.source_id, index, exc))


def _validate_detection_coverage(sample: Sample, errors: List[str]) -> None:
    # Custom marker/obstacle classes are not expected from a standard COCO YOLO.
    required: Set[str] = {"clock", "bench", "person", "stop sign", "zebra", "elephant", "giraffe"}
    for gold in sample.annotations:
        if gold["category"] not in required:
            continue
        matches = [
            detection
            for detection in sample.detections
            if detection.get("category") == gold["category"]
            and float(detection.get("score", 0.0)) >= 0.2
            and _bbox_iou(gold["bbox"], detection.get("bbox", [])) >= 0.3
        ]
        if not matches:
            errors.append(
                "%s has no detector match for gold object %s (%s) at IoU>=0.3 and score>=0.2"
                % (sample.source_id, gold.get("category_id"), gold["category"])
            )


def validate_samples(
    samples: List[Sample], *, require_scores: bool = False, detector_name: str = "oracle"
) -> Dict[str, object]:
    errors: List[str] = []
    qa = RuleQAGenerator()
    ids = [sample.id for sample in samples]
    source_ids = [sample.source_id for sample in samples]
    if len(ids) != len(set(ids)):
        errors.append("sample ids are not unique")
    if len(source_ids) != len(set(source_ids)):
        errors.append("source_ids are not unique")

    for sample in samples:
        source_path = Path(sample.source_id)
        if source_path.is_absolute() or source_path.name != sample.source_id or ".." in source_path.parts:
            errors.append("%s is not a safe plain source filename" % sample.source_id)
        for obj in sample.scene.objects:
            if not obj.bbox.within(sample.scene.width, sample.scene.height):
                errors.append("%s has out-of-bounds object %s" % (sample.source_id, obj.uid))
        expected = qa.generate(sample.scene, sample.capability)
        if (sample.question_type, sample.question, sample.answer) != (
            expected.question_type,
            expected.question,
            expected.answer,
        ):
            errors.append("%s QA does not match scene-derived truth" % sample.source_id)
        expected_annotations = [annotation_from_object(obj) for obj in sample.scene.objects]
        if sample.annotations != expected_annotations:
            errors.append("%s gold annotations do not match scene objects" % sample.source_id)
        _validate_detection_sanity(sample, errors)
        if detector_name == "oracle":
            if sample.detections != expected_annotations:
                errors.append("%s Oracle detections do not match gold annotations" % sample.source_id)
        else:
            _validate_detection_coverage(sample, errors)
        if require_scores:
            if sample.bo3 is None or sample.bo3 not in range(4):
                errors.append("%s has invalid BO3" % sample.source_id)
            if len(sample.attempts) != 3:
                errors.append("%s does not have three attempts" % sample.source_id)
            else:
                if [item.attempt for item in sample.attempts] != [0, 1, 2]:
                    errors.append("%s attempts are not exactly [0, 1, 2]" % sample.source_id)
                recomputed_correct = []
                expected_answer = normalize_answer(sample.answer)
                for item in sample.attempts:
                    normalized = normalize_answer(item.raw_answer) if item.raw_answer else ""
                    if item.error is not None:
                        errors.append("%s has an invalid evaluator attempt: %s" % (sample.source_id, item.error))
                    if normalized != item.normalized_answer:
                        errors.append("%s stored normalized answer is inconsistent" % sample.source_id)
                    correct = normalized == expected_answer and item.error is None
                    if correct != item.correct:
                        errors.append("%s stored correctness is inconsistent" % sample.source_id)
                    recomputed_correct.append(correct)
                if sum(recomputed_correct) != sample.bo3:
                    errors.append("%s BO3 disagrees with recomputed attempts" % sample.source_id)

    grouped: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group_id].append(sample)
    for group_id, rows in grouped.items():
        roles = {row.variant_role for row in rows}
        expected_roles = set(VARIANT_ROLES)
        role_counts = Counter(row.variant_role for row in rows)
        if len(rows) != 4 or roles != expected_roles or any(role_counts[role] != 1 for role in expected_roles):
            errors.append("%s must contain each of the four variant roles exactly once" % group_id)
        else:
            by_role = {row.variant_role: row for row in rows}
            base = by_role["base"]
            for role in ("nuisance_brightness", "nuisance_compression"):
                nuisance = by_role[role]
                if nuisance.answer != base.answer:
                    errors.append("%s %s changed the answer" % (group_id, role))
                if _canonical_scene(nuisance) != _canonical_scene(base):
                    errors.append("%s %s changed scene state" % (group_id, role))
                if nuisance.image_bytes == base.image_bytes:
                    errors.append("%s %s did not change pixels" % (group_id, role))
            counterfactual = by_role["counterfactual"]
            if counterfactual.answer == base.answer:
                errors.append("%s counterfactual did not change the answer" % group_id)
            if "counterfactual_change" not in counterfactual.scene.metadata:
                errors.append("%s counterfactual lacks change metadata" % group_id)
        splits = {row.split for row in rows}
        if len(splits) != 1:
            errors.append("%s leaks across splits" % group_id)

    family_splits: Dict[str, Set[str]] = defaultdict(set)
    for sample in samples:
        family_splits[sample.scene_family_id].add(sample.split)
    leaking_families = {family: sorted(splits) for family, splits in family_splits.items() if len(splits) > 1}
    if leaking_families:
        errors.append("scene families leak across splits: %s" % json.dumps(leaking_families, sort_keys=True))

    if errors:
        raise DatasetValidationError("\n".join(errors))

    return {
        "rows": len(samples),
        "groups": len(grouped),
        "capabilities": dict(Counter(sample.capability for sample in samples)),
        "splits": dict(Counter(sample.split for sample in samples)),
        "group_leakage": False,
        "scene_family_leakage": False,
        "detector_backend": detector_name,
        "scores_validated": require_scores,
    }


def validate_smoke_shape(samples: List[Sample]) -> None:
    if len(samples) != 20:
        raise DatasetValidationError("Smoke dataset must contain 20 rows")
    capability_counts = Counter(sample.capability for sample in samples)
    if capability_counts != Counter({capability: 4 for capability in CAPABILITIES}):
        raise DatasetValidationError("Each smoke capability must contain four rows")
    group_counts = Counter(sample.group_id for sample in samples)
    if len(group_counts) != 5 or any(count != 4 for count in group_counts.values()):
        raise DatasetValidationError("Smoke dataset must contain five complete four-row groups")
    if {sample.split for sample in samples} != {"smoke"}:
        raise DatasetValidationError("Smoke rows must use the non-benchmark 'smoke' split")
