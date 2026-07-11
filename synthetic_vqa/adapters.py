from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Protocol

from PIL import Image

from .models import CATEGORY_IDS, Prediction, Sample, Scene, annotation_from_object


class AdapterUnavailableError(RuntimeError):
    """Raised when an adapter intentionally requires the later dev machine."""


@dataclass(frozen=True)
class DetectionInput:
    image_bytes: bytes
    source_id: str


@dataclass(frozen=True)
class EvaluationInput:
    image_bytes: bytes
    question: str


class Detector(Protocol):
    name: str

    def detect(self, request: DetectionInput) -> List[Dict[str, object]]:
        ...


class Evaluator(Protocol):
    name: str
    evaluation_kind: str

    def answer(self, request: EvaluationInput, *, attempt: int, seed: int) -> str:
        ...


class OracleDetector:
    name = "oracle"

    def __init__(self, scenes: Dict[str, Scene]) -> None:
        self.scenes = scenes

    def detect(self, request: DetectionInput) -> List[Dict[str, object]]:
        scene = self.scenes[request.source_id]
        with Image.open(io.BytesIO(request.image_bytes)) as image:
            if image.size != (scene.width, scene.height):
                raise ValueError("Rendered image size does not match scene state")
            image.verify()
        return [annotation_from_object(obj) for obj in scene.objects]


class IdealDetectionReplay(OracleDetector):
    """Explicit local stand-in for a saved perfect detector pass.

    It is deliberately named differently from YOLO in exported provenance, so
    local fixture runs cannot be confused with a real detector experiment.
    """

    name = "ideal-detection-replay"

    def provenance(self) -> Dict[str, object]:
        return {"backend": self.name, "kind": "local_perfect_fixture", "is_real_yolo": False}


class YoloDetector:
    """Ultralytics YOLO adapter, imported only when weights are supplied.

    The public Zoo-Bus pipeline uses an Ultralytics detector before GRAID QA
    generation.  Keeping the import lazy lets the compatibility/test runner
    exercise the same protocol without downloading a model or adding the
    heavyweight runtime to this environment.
    """

    name = "yolo"

    def __init__(
        self,
        model_path: str = "",
        *,
        confidence: float = 0.25,
        class_map: Mapping[str, str] = None,
    ) -> None:
        self.model_path = model_path
        self.confidence = float(confidence)
        self.class_map = dict(
            class_map
            or {
                "person": "person",
                "bench": "bench",
                "stop sign": "stop sign",
                "elephant": "elephant",
                "zebra": "zebra",
                "giraffe": "giraffe",
                "clock": "clock",
            }
        )
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("YOLO confidence must be in (0, 1]")

    def detect(self, request: DetectionInput) -> List[Dict[str, object]]:
        if not self.model_path:
            raise AdapterUnavailableError(
                "YOLO weights are not configured. Supply --yolo-weights on the development machine; "
                "the local build intentionally uses the ideal/replay detector instead."
            )
        weights = Path(self.model_path).expanduser()
        if not weights.is_file():
            raise AdapterUnavailableError("YOLO weights do not exist: %s" % weights)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise AdapterUnavailableError(
                "Ultralytics is not installed. Install it on the development machine before using --detector yolo."
            ) from exc
        with Image.open(io.BytesIO(request.image_bytes)) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            result = YOLO(str(weights))(rgb, conf=self.confidence, verbose=False)[0]
        names = result.names
        detections: List[Dict[str, object]] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            raw_name = str(names[class_id])
            category = self.class_map.get(raw_name)
            if category is None:
                continue
            x0, y0, x1, y1 = (float(value) for value in box.xyxy[0].tolist())
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(float(width), x1), min(float(height), y1)
            if x1 <= x0 or y1 <= y0:
                continue
            detections.append(
                {
                    "area": (x1 - x0) * (y1 - y0),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "category": category,
                    "category_id": CATEGORY_IDS[category],
                    "iscrowd": 0,
                    "score": float(box.conf.item()),
                }
            )
        marker = self._detect_heading_marker(rgb, detections)
        if marker is not None:
            detections.append(marker)
        return detections

    def provenance(self) -> Dict[str, object]:
        weights = Path(self.model_path).expanduser() if self.model_path else None
        digest = None
        if weights is not None and weights.is_file():
            hasher = hashlib.sha256()
            with weights.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        return {
            "backend": self.name,
            "kind": "ultralytics_yolo_plus_image_heading_dot",
            "weights": None if weights is None else str(weights),
            "weights_sha256": digest,
            "confidence": self.confidence,
            "class_map": self.class_map,
            "is_real_yolo": True,
        }

    @staticmethod
    def _detect_heading_marker(
        image: Image.Image, detections: List[Dict[str, object]]
    ) -> Optional[Dict[str, object]]:
        """Recover the rendered red heading dot near a detected clock.

        Zoo-Bus encodes heading with a synthetic red dot rather than a COCO
        class.  Standard YOLO can detect ``clock`` but not that custom marker,
        so this small deterministic post-process makes the otherwise-visible
        geometric cue available to the same filtering contract.  It only
        searches compact windows around clock detections and never has access
        to generator coordinates.
        """

        clocks = [item for item in detections if item["category"] == "clock"]
        if not clocks:
            return None
        pixels = image.load()
        for clock in clocks:
            x, y, width, height = (float(value) for value in clock["bbox"])
            cx, cy = x + width / 2.0, y + height / 2.0
            radius = max(width, height) * 1.55
            left, top = max(0, int(cx - radius)), max(0, int(cy - radius))
            right, bottom = min(image.width, int(cx + radius)), min(image.height, int(cy + radius))
            seen: set[tuple[int, int]] = set()
            candidates: List[tuple[float, float, int, int, int, int]] = []
            for py in range(top, bottom):
                for px in range(left, right):
                    if (px, py) in seen:
                        continue
                    red, green, blue = pixels[px, py]
                    if not (red >= 170 and green <= 105 and blue <= 105 and red >= green * 1.8):
                        continue
                    stack = [(px, py)]
                    seen.add((px, py))
                    area = 0
                    min_x = max_x = px
                    min_y = max_y = py
                    while stack:
                        sx, sy = stack.pop()
                        area += 1
                        min_x, max_x = min(min_x, sx), max(max_x, sx)
                        min_y, max_y = min(min_y, sy), max(max_y, sy)
                        for nx, ny in ((sx - 1, sy), (sx + 1, sy), (sx, sy - 1), (sx, sy + 1)):
                            if nx < left or nx >= right or ny < top or ny >= bottom or (nx, ny) in seen:
                                continue
                            rr, gg, bb = pixels[nx, ny]
                            if rr >= 170 and gg <= 105 and bb <= 105 and rr >= gg * 1.8:
                                seen.add((nx, ny))
                                stack.append((nx, ny))
                    component_cx, component_cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
                    distance = ((component_cx - cx) ** 2 + (component_cy - cy) ** 2) ** 0.5
                    if 80 <= area <= 2200 and max(width, height) * 0.35 <= distance <= max(width, height) * 1.25:
                        candidates.append((abs(area - 650), component_cx, component_cy, min_x, min_y, max(max_x - min_x, max_y - min_y)))
            if candidates:
                _, _, _, min_x, min_y, size = min(candidates, key=lambda item: item[0])
                return {
                    "area": float(size * size),
                    "bbox": [float(min_x), float(min_y), float(size), float(size)],
                    "category": "heading marker",
                    "category_id": CATEGORY_IDS["heading marker"],
                    "iscrowd": 0,
                    "score": 0.90,
                }
        return None


class DetectionReplay:
    """Read detector candidates from JSONL for deterministic offline replay.

    Every line has ``source_id``, ``image_sha256``, ``image_size`` and
    ``detections`` fields.  Binding replay candidates to exact rendered bytes
    prevents a stale YOLO run from silently relabeling a different scene.
    """

    name = "detection-replay"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError("Detection replay file does not exist: %s" % self.path)
        self.by_source: Dict[str, Dict[str, object]] = {}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite JSON constant %s" % value)))
            source_id = str(row["source_id"])
            detections = row["detections"]
            digest = row.get("image_sha256")
            image_size = row.get("image_size")
            if (
                source_id in self.by_source
                or not isinstance(detections, list)
                or not isinstance(digest, str)
                or len(digest) != 64
                or not isinstance(image_size, list)
                or len(image_size) != 2
                or not all(isinstance(value, int) and value > 0 for value in image_size)
            ):
                raise ValueError("Invalid replay row %d" % line_number)
            self.by_source[source_id] = {
                "detections": [dict(item) for item in detections],
                "image_sha256": digest.lower(),
                "image_size": tuple(image_size),
            }

    def validate_sources(self, source_ids: List[str]) -> None:
        expected, actual = set(source_ids), set(self.by_source)
        if expected != actual:
            raise ValueError(
                "Detection replay source_id set differs from generated images; missing=%s extra=%s"
                % (sorted(expected - actual), sorted(actual - expected))
            )

    def provenance(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "kind": "sha256_bound_detection_replay",
            "replay_path": str(self.path),
            "source_count": len(self.by_source),
            "is_real_yolo": False,
        }

    def detect(self, request: DetectionInput) -> List[Dict[str, object]]:
        if request.source_id not in self.by_source:
            raise KeyError("No replay detections for %s" % request.source_id)
        row = self.by_source[request.source_id]
        if hashlib.sha256(request.image_bytes).hexdigest() != row["image_sha256"]:
            raise ValueError("Detection replay image_sha256 mismatch for %s" % request.source_id)
        with Image.open(io.BytesIO(request.image_bytes)) as image:
            if image.size != row["image_size"]:
                raise ValueError("Detection replay image_size mismatch for %s" % request.source_id)
        return [dict(item) for item in row["detections"]]  # type: ignore[index]


class ApiEvaluator:
    """Reserved integration point for the model API supplied later."""

    name = "api"
    evaluation_kind = "model_api"

    def __init__(self, endpoint: str = "") -> None:
        self.endpoint = endpoint

    def answer(self, request: EvaluationInput, *, attempt: int, seed: int) -> str:
        del request, attempt, seed
        raise AdapterUnavailableError(
            "The model API is not configured. Supply the endpoint and request "
            "contract on the later development machine."
        )


@dataclass(frozen=True)
class MockPlan:
    answer_key: Dict[str, str]
    wrong_answer_key: Dict[str, str]
    desired_bo3: Dict[str, int]


def evaluation_key(request: EvaluationInput) -> str:
    digest = hashlib.sha256()
    digest.update(request.image_bytes)
    digest.update(b"\x00")
    digest.update(request.question.encode("utf-8"))
    return digest.hexdigest()


class DeterministicMockEvaluator:
    """Simulates BO3 outcomes without pretending to be a real vision model."""

    name = "deterministic-mock"
    evaluation_kind = "synthetic_mock"

    def __init__(self, plan: MockPlan) -> None:
        self.plan = plan

    @staticmethod
    def _slot(key: str) -> int:
        digest = hashlib.sha256(key.encode("ascii")).digest()
        return digest[0] % 3

    def answer(self, request: EvaluationInput, *, attempt: int, seed: int) -> str:
        del seed
        key = evaluation_key(request)
        desired = self.plan.desired_bo3[key]
        slot = self._slot(key)
        if desired == 3:
            correct = True
        elif desired == 2:
            correct = attempt != slot
        elif desired == 1:
            correct = attempt == slot
        elif desired == 0:
            correct = False
        else:
            raise ValueError("BO3 target must be in [0, 3]")
        return self.plan.answer_key[key] if correct else self.plan.wrong_answer_key[key]


def normalize_answer(value: str) -> str:
    return " ".join(value.strip().lower().split())


def score_samples(samples: List[Sample], evaluator: Evaluator) -> None:
    for sample in samples:
        predictions: List[Prediction] = []
        expected = normalize_answer(sample.answer)
        request = EvaluationInput(
            image_bytes=sample.image_bytes,
            question=sample.question,
        )
        invalid = False
        for attempt in range(3):
            seed = sample.seed * 1000 + sample.id * 10 + attempt
            try:
                raw = evaluator.answer(request, attempt=attempt, seed=seed)
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError("Evaluator returned an empty or non-string answer")
                normalized = normalize_answer(raw)
                error = None
            except Exception as exc:  # adapters may fail per sample without corrupting BO3
                raw = ""
                normalized = ""
                error = "%s: %s" % (type(exc).__name__, exc)
                invalid = True
            predictions.append(
                Prediction(
                    attempt=attempt,
                    raw_answer=raw,
                    normalized_answer=normalized,
                    correct=normalized == expected,
                    seed=seed,
                    error=error,
                )
            )
        sample.attempts = predictions
        sample.bo3 = None if invalid else sum(1 for item in predictions if item.correct)


def smoke_mock_plan(samples: List[Sample]) -> MockPlan:
    role_order = ("base", "nuisance_brightness", "nuisance_compression", "counterfactual")
    capability_patterns = {
        "perception": (0, 1, 2, 3),
        "counting": (0, 1, 2, 3),
        "spatial": (1, 1, 2, 3),
        "heading": (1, 2, 2, 3),
        "obstacle": (1, 2, 2, 2),
    }
    answers: Dict[str, str] = {}
    wrong_answers: Dict[str, str] = {}
    targets: Dict[str, int] = {}
    for sample in samples:
        key = evaluation_key(EvaluationInput(image_bytes=sample.image_bytes, question=sample.question))
        answers[key] = sample.answer
        if sample.capability == "perception":
            wrong_answers[key] = "green"
        elif sample.capability == "counting":
            wrong_answers[key] = str(int(sample.answer) + 1)
        elif sample.capability == "spatial":
            wrong_answers[key] = "2" if sample.answer == "1" else "1"
        elif sample.capability == "heading":
            wrong_answers[key] = "South" if sample.answer != "South" else "North"
        else:
            choices = ("keep straight", "turn left", "turn right")
            wrong_answers[key] = next(choice for choice in choices if choice != sample.answer)
        targets[key] = capability_patterns[sample.capability][role_order.index(sample.variant_role)]
    return MockPlan(answer_key=answers, wrong_answer_key=wrong_answers, desired_bo3=targets)
