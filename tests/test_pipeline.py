from __future__ import annotations

import json
import copy
import io
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from dataclasses import fields
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from synthetic_vqa.adapters import (
    AdapterUnavailableError,
    ApiEvaluator,
    DetectionInput,
    EvaluationInput,
    OracleDetector,
    YoloDetector,
    score_samples,
    smoke_mock_plan,
    DeterministicMockEvaluator,
)
from synthetic_vqa.pipeline import run_smoke_pipeline
from synthetic_vqa.pipeline import run_reference_pipeline
from synthetic_vqa.reference import (
    HELDOUT_QUESTION_TYPES,
    QUESTION_TYPES,
    DetectionGeometryFilter,
    ReferenceBuildConfig,
    ZooBusTemplateGenerator,
    build_reference_samples,
)
from synthetic_vqa.scene import generate_smoke_samples
from synthetic_vqa.selection import SMOKE_BO3_QUOTA, bo3_counts
from synthetic_vqa.validation import validate_samples, validate_smoke_shape
from synthetic_vqa.validation import DatasetValidationError
from synthetic_vqa.models import BBox, Scene, SceneObject, annotation_from_object


class GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = generate_smoke_samples(seed=42)
        detector = OracleDetector({sample.source_id: sample.scene for sample in self.samples})
        for sample in self.samples:
            sample.detections = detector.detect(DetectionInput(sample.image_bytes, sample.source_id))

    def test_smoke_shape_and_ground_truth(self) -> None:
        validate_smoke_shape(self.samples)
        summary = validate_samples(self.samples)
        self.assertEqual(summary["rows"], 20)
        self.assertEqual(summary["groups"], 5)
        self.assertEqual(Counter(sample.variant_role for sample in self.samples), Counter({
            "base": 5,
            "nuisance_brightness": 5,
            "nuisance_compression": 5,
            "counterfactual": 5,
        }))

    def test_generation_is_byte_deterministic(self) -> None:
        repeated = generate_smoke_samples(seed=42)
        self.assertEqual(
            [(item.source_id, item.answer, item.image_bytes, item.scene.to_dict()) for item in self.samples],
            [(item.source_id, item.answer, item.image_bytes, item.scene.to_dict()) for item in repeated],
        )

    def test_reference_render_state_preserves_native_and_output_boxes(self) -> None:
        sample = self.samples[0]
        self.assertEqual((sample.scene.width, sample.scene.height), (1280, 960))
        self.assertEqual(sample.scene.metadata["native_size"], [4032, 3024])
        self.assertEqual(sample.scene.metadata["output_size"], [1280, 960])
        self.assertEqual(sample.scene.metadata["renderer"], "zoo_bus_sprite_compositor_v1")
        bench = next(obj for obj in sample.scene.objects if obj.category == "bench")
        self.assertIsNotNone(bench.source_bbox)
        self.assertAlmostEqual(bench.bbox.width, bench.source_bbox.width * sample.scene.metadata["scale"])
        self.assertTrue(sample.source_id.endswith(".jpg"))

    def test_strict_asset_mode_requires_reference_asset_root(self) -> None:
        with self.assertRaises(FileNotFoundError):
            generate_smoke_samples(seed=42, strict_assets=True)

    def test_mock_bo3_matches_requested_distribution(self) -> None:
        evaluator = DeterministicMockEvaluator(smoke_mock_plan(self.samples))
        score_samples(self.samples, evaluator)
        validate_samples(self.samples, require_scores=True)
        self.assertEqual(bo3_counts(self.samples), SMOKE_BO3_QUOTA)
        for sample in self.samples:
            valid_answers = {sample.answer}
            valid_answers.update(item.raw_answer for item in sample.attempts)
            self.assertNotIn("__mock_wrong__", valid_answers)

    def test_validation_rejects_incomplete_or_duplicate_group_roles(self) -> None:
        missing = copy.deepcopy(self.samples)
        missing[0].variant_role = "nuisance_brightness"
        with self.assertRaises(DatasetValidationError):
            validate_samples(missing)

        duplicate = copy.deepcopy(self.samples)
        duplicate[1].variant_role = "base"
        with self.assertRaises(DatasetValidationError):
            validate_samples(duplicate)

    def test_validation_recomputes_attempt_correctness_and_indices(self) -> None:
        evaluator = DeterministicMockEvaluator(smoke_mock_plan(self.samples))
        score_samples(self.samples, evaluator)
        forged = copy.deepcopy(self.samples)
        first = forged[0]
        first.attempts = [
            replace(item, raw_answer="definitely wrong", normalized_answer="definitely wrong", correct=True)
            for item in first.attempts
        ]
        first.bo3 = 3
        with self.assertRaises(DatasetValidationError):
            validate_samples(forged, require_scores=True)

        repeated = copy.deepcopy(self.samples)
        repeated[0].attempts = [replace(item, attempt=0) for item in repeated[0].attempts]
        with self.assertRaises(DatasetValidationError):
            validate_samples(repeated, require_scores=True)

    def test_validation_rejects_unsafe_source_id(self) -> None:
        unsafe = copy.deepcopy(self.samples)
        unsafe[0].source_id = "../../escape.png"
        with self.assertRaises(DatasetValidationError):
            validate_samples(unsafe)


class AdapterTests(unittest.TestCase):
    def test_unconfigured_adapters_fail_clearly(self) -> None:
        sample = generate_smoke_samples(seed=1)[0]
        with self.assertRaises(AdapterUnavailableError):
            YoloDetector().detect(DetectionInput(sample.image_bytes, sample.source_id))
        with self.assertRaises(AdapterUnavailableError):
            ApiEvaluator().answer(EvaluationInput(sample.image_bytes, sample.question), attempt=0, seed=1)

    def test_yolo_postprocess_can_recover_visible_heading_dot_without_scene_state(self) -> None:
        sample = generate_smoke_samples(seed=42)[0]
        clock = next(item for item in sample.annotations if item["category"] == "clock")
        with Image.open(io.BytesIO(sample.image_bytes)) as image:
            marker = YoloDetector._detect_heading_marker(image.convert("RGB"), [clock])
        self.assertIsNotNone(marker)
        self.assertEqual(marker["category"], "heading marker")
        expected = next(item for item in sample.annotations if item["category"] == "heading marker")
        marker_center = (marker["bbox"][0] + marker["bbox"][2] / 2.0, marker["bbox"][1] + marker["bbox"][3] / 2.0)
        expected_center = (expected["bbox"][0] + expected["bbox"][2] / 2.0, expected["bbox"][1] + expected["bbox"][3] / 2.0)
        self.assertLess(((marker_center[0] - expected_center[0]) ** 2 + (marker_center[1] - expected_center[1]) ** 2) ** 0.5, 6.0)

    def test_evaluation_input_cannot_carry_gold_linkable_fields(self) -> None:
        self.assertEqual({item.name for item in fields(EvaluationInput)}, {"image_bytes", "question"})


class EndToEndTests(unittest.TestCase):
    def test_pipeline_exports_readable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "smoke"
            summary = run_smoke_pipeline(output, seed=42)
            self.assertEqual(summary["rows"], 20)
            self.assertEqual(summary["bo3"], {"0": 2, "1": 6, "2": 8, "3": 4})
            self.assertEqual(summary["splits"], {"smoke": 20})
            self.assertEqual(summary["adapters"]["evaluation_kind"], "synthetic_mock")
            self.assertFalse(summary["adapters"]["is_real_evaluation"])

            table = pq.read_table(output / "dataset.parquet")
            self.assertEqual(table.num_rows, 20)
            first_image = table.column("image")[0].as_py()["bytes"]
            image_path = output / "_readback.jpg"
            image_path.write_bytes(first_image)
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (1280, 960))

            records = [json.loads(line) for line in (output / "messages.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 20)
            for record in records:
                self.assertEqual([message["role"] for message in record["messages"]], ["user", "assistant"])
                image_ref = record["messages"][0]["content"][0]["image"]
                self.assertTrue((output / image_ref).is_file())
                serialized = json.dumps(record, ensure_ascii=False)
                self.assertNotIn("bo3", serialized.lower())
                self.assertNotIn("proof", serialized.lower())

            persisted = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted, summary)

    def test_pipeline_reports_injected_backend_provenance(self) -> None:
        reference = generate_smoke_samples(seed=42)
        oracle = OracleDetector({sample.source_id: sample.scene for sample in reference})

        class NamedDetector:
            name = "custom-detector"

            def detect(self, request):
                return oracle.detect(request)

        evaluator = DeterministicMockEvaluator(smoke_mock_plan(reference))
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_smoke_pipeline(
                Path(temp_dir) / "custom",
                seed=42,
                detector=NamedDetector(),
                evaluator=evaluator,
            )
        self.assertEqual(summary["adapters"]["detector"], "custom-detector")
        self.assertEqual(summary["adapters"]["evaluator"], "deterministic-mock")


class ReferencePipelineTests(unittest.TestCase):
    def test_public_question_catalog_has_all_29_types(self) -> None:
        self.assertEqual(len(QUESTION_TYPES), 29)
        self.assertEqual(len(set(QUESTION_TYPES)), 29)
        self.assertEqual(len(HELDOUT_QUESTION_TYPES), 6)
        self.assertTrue(HELDOUT_QUESTION_TYPES.issubset(set(QUESTION_TYPES)))

    def test_reference_pipeline_keeps_source_ids_in_one_split_and_holds_out_compositions(self) -> None:
        config = ReferenceBuildConfig(num_scenes=12, seed=42, per_answer=1, per_question_type=2)
        rows, audit, detector_name = build_reference_samples(config)
        self.assertEqual(detector_name, "ideal-detection-replay")
        self.assertGreater(len(rows), 0)
        by_source = {}
        for row in rows:
            by_source.setdefault(row.source_id, []).append(row)
            self.assertEqual(row.proof["evidence_source"], "filtered_detections")
            self.assertIsNone(row.bo3)
            self.assertEqual(row.attempts, [])
            if row.split == "train":
                self.assertNotIn(row.question_type, HELDOUT_QUESTION_TYPES)
        self.assertTrue(any(len(values) > 1 for values in by_source.values()))
        for source_rows in by_source.values():
            self.assertEqual(len({row.split for row in source_rows}), 1)
            self.assertEqual(len({row.image_bytes for row in source_rows}), 1)
        self.assertFalse(audit["validation"]["source_id_split_leakage"])
        self.assertFalse(audit["validation"]["heldout_train_leakage"])
        self.assertGreater(audit["balance"]["heldout_removed_from_train"], 0)

    def test_detector_filter_drops_duplicate_and_nonmatching_geometry(self) -> None:
        config = ReferenceBuildConfig(num_scenes=1, seed=7, per_answer=0, per_question_type=0)
        source_rows, _, _ = build_reference_samples(config)
        source = source_rows[0]
        # Omit the first bench and replace it with a far-away false box.  A
        # second retained gold box separately exercises class-NMS.
        duplicate = dict(source.annotations[1])
        detections = [dict(item) for item in source.annotations[1:]] + [duplicate]
        detections.append(
            {
                "category": "bench",
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "score": 0.99,
            }
        )
        state = DetectionGeometryFilter(min_score=0.25, identity_iou=0.30).prepare(source.scene, detections)
        rejected = state.report["rejected_detections"]
        self.assertGreaterEqual(rejected.get("duplicate_nms", 0), 1)
        self.assertGreaterEqual(rejected.get("identity_iou", 0), 1)
        self.assertLess(len(state.retained_detections), len(detections))

    def test_direction_stability_keeps_sector_center_and_rejects_22_5_boundary(self) -> None:
        clock = SceneObject("clock", "clock", BBox(450, 450, 100, 100), "white")
        marker = SceneObject("marker", "heading marker", BBox(590, 490, 20, 20), "red")
        # Its center is 22.5 degrees north of east from the clock center.
        bench = SceneObject("bench", "bench", BBox(760, 325.7, 80, 80), "brown", number=1)
        scene = Scene(width=1280, height=960, objects=[clock, marker, bench], heading=(1, 0))
        state = DetectionGeometryFilter(min_score=0.25, identity_iou=0.30).prepare(
            scene, [annotation_from_object(item) for item in scene.objects]
        )
        templates = ZooBusTemplateGenerator(ReferenceBuildConfig(num_scenes=1))
        self.assertEqual(templates._heading(state)[2], "East")
        with self.assertRaises(ValueError):
            templates._direction(state, state.objects("bench")[0])

    def test_filter_rejects_nonfinite_detector_values(self) -> None:
        source = generate_smoke_samples(seed=5)[0]
        detections = [dict(item) for item in source.annotations]
        detections.append({"category": "bench", "bbox": [0.0, 0.0, 1.0, 1.0], "score": float("nan")})
        state = DetectionGeometryFilter(min_score=0.25, identity_iou=0.30).prepare(source.scene, detections)
        self.assertEqual(state.report["rejected_detections"].get("non_finite_detection"), 1)

    def test_filter_requires_complete_and_stable_visible_anchor_numbering(self) -> None:
        source = generate_smoke_samples(seed=6)[0]
        leftmost = min(source.scene.find(category="bench"), key=lambda item: item.bbox.center[0])
        removed = annotation_from_object(leftmost)
        detections = [item for item in source.annotations if item != removed]
        missing_state = DetectionGeometryFilter(min_score=0.25, identity_iou=0.30).prepare(source.scene, detections)
        self.assertEqual(missing_state.objects("bench"), [])
        self.assertGreater(missing_state.report["rejected_detections"].get("incomplete_visible_number_set", 0), 0)

        clock = SceneObject("clock", "clock", BBox(500, 100, 100, 100), "white")
        bench_a = SceneObject("bench-a", "bench", BBox(250, 250, 120, 50), "brown", number=1)
        # Different y positions avoid sprite overlap while their horizontal
        # centers are too close to infer the image's left-to-right IDs safely.
        bench_b = SceneObject("bench-b", "bench", BBox(320, 500, 120, 50), "brown", number=2)
        close_scene = Scene(width=1280, height=960, objects=[clock, bench_a, bench_b], heading=(1, 0))
        close_state = DetectionGeometryFilter(min_score=0.25, identity_iou=0.30).prepare(
            close_scene, [annotation_from_object(item) for item in close_scene.objects]
        )
        self.assertEqual(close_state.objects("bench"), [])
        self.assertGreater(close_state.report["rejected_detections"].get("unstable_visible_number_order", 0), 0)

    def test_strict_balance_refuses_unattainable_small_build(self) -> None:
        with self.assertRaises(RuntimeError):
            build_reference_samples(
                ReferenceBuildConfig(num_scenes=3, seed=8, per_answer=1, per_question_type=2, strict_balance=True)
            )

    def test_reference_export_has_shared_images_and_construction_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "reference"
            summary = run_reference_pipeline(
                output,
                config=ReferenceBuildConfig(num_scenes=10, seed=9, per_answer=1, per_question_type=2),
            )
            self.assertEqual(summary["pipeline"]["kind"], "zoo_bus_compatible_reference")
            self.assertEqual(summary["adapters"]["detector"], "ideal-detection-replay")
            self.assertIsNone(summary["bo3"])
            self.assertTrue((output / "construction_audit.json").is_file())
            self.assertTrue((output / "evaluation_inputs.jsonl").is_file())
            records = [json.loads(line) for line in (output / "messages.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), summary["rows"])
            image_refs = {record["source_id"]: record["messages"][0]["content"][0]["image"] for record in records}
            self.assertEqual(len(image_refs), summary["source_images"])
            for image_ref in image_refs.values():
                self.assertTrue((output / image_ref).is_file())
            evaluation_inputs = [
                json.loads(line)
                for line in (output / "evaluation_inputs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(evaluation_inputs)
            self.assertTrue(all(len(record["messages"]) == 1 for record in evaluation_inputs))
            self.assertTrue(all(record["messages"][0]["role"] == "user" for record in evaluation_inputs))


if __name__ == "__main__":
    unittest.main()
