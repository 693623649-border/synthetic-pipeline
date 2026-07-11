"""Executable, Zoo-Bus-compatible synthetic VQA construction pipeline.

The public Zoo-Bus repository documents the stage ordering but does not ship
its private ``scene_gen`` assets or GRAID question-class implementation.  This
module reproduces the observable contract: sampled scene parameters are
rendered as 2D sprites; detections (not generator coordinates) drive QA;
ambiguous detections/geometries are rejected; QA rows share an image-level
``source_id``; then split and answer-aware balancing happen on retained rows.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .adapters import AdapterUnavailableError, DetectionInput, Detector, IdealDetectionReplay
from .models import BBox, QAPair, Sample, Scene, SceneObject, annotation_from_object
from .rendering import AssetPack, NATIVE_SIZE, render_reference_scene


QUESTION_TYPES: Tuple[str, ...] = (
    "CountPeople",
    "CountAnimals",
    "CountPeopleAtBench",
    "CountAnimalsAtStopSign",
    "ListBenchesWithAtLeastKPeople",
    "ListStopSignsWithAtLeastKAnimals",
    "ArrivedAtBench",
    "ArrivedAtAnimalsAroundStopSigns",
    "ClosestBench",
    "ClosestStopSign",
    "PairwiseCloserBench",
    "PairwiseCloserStopSign",
    "ClosestToFurthestBenches",
    "ClosestToFurthestStopSigns",
    "GeometricDirectionToBench",
    "GeometricDirectionToStopSign",
    "AvoidObstacleToReachBench",
    "AvoidObstacleToReachStopSign",
    "BusHeadingDirection",
    "TurnDirectionToBench",
    "TurnDirectionToStopSign",
    "BenchRelativeToHeading",
    "StopSignRelativeToHeading",
    "CountPersonAtClosestBench",
    "ClosestBenchWithPerson",
    "AvoidObstacleToReachClosestBench",
    "AvoidObstacleToReachClosestStopSign",
    "DirectionToClosestBench",
    "DirectionToClosestStopSign",
)

HELDOUT_QUESTION_TYPES = frozenset(
    {
        "CountPersonAtClosestBench",
        "ClosestBenchWithPerson",
        "AvoidObstacleToReachClosestBench",
        "AvoidObstacleToReachClosestStopSign",
        "DirectionToClosestBench",
        "DirectionToClosestStopSign",
    }
)

ANIMAL_CATEGORIES = frozenset({"zebra", "elephant", "giraffe"})
ANCHOR_CATEGORIES = frozenset({"bench", "stop sign"})
DETECTED_CATEGORIES = frozenset(
    {"clock", "bench", "stop sign", "person", "zebra", "elephant", "giraffe", "heading marker"}
)
DIRECTION_LABELS = ("East", "Northeast", "North", "Northwest", "West", "Southwest", "South", "Southeast")


class GeometryRejected(ValueError):
    """Raised for incomplete or ambiguous detection geometry."""


@dataclass(frozen=True)
class ReferenceBuildConfig:
    num_scenes: int = 120
    seed: int = 42
    train_ratio: float = 0.8
    evaluation_ratio: float = 0.1
    test_ratio: float = 0.1
    min_detector_score: float = 0.25
    identity_iou: float = 0.30
    nearest_margin_ratio: float = 0.035
    association_margin_ratio: float = 0.025
    direction_boundary_degrees: float = 6.0
    per_answer: int = 8
    per_question_type: int = 24
    max_rows_per_source: int = 29
    keep_heldout_out_of_train: bool = True
    strict_balance: bool = False
    generator_version: str = "zoo-bus-compatible-v1"

    def __post_init__(self) -> None:
        if self.num_scenes <= 0:
            raise ValueError("num_scenes must be positive")
        if any(value < 0.0 for value in (self.train_ratio, self.evaluation_ratio, self.test_ratio)):
            raise ValueError("split ratios cannot be negative")
        if not math.isclose(self.train_ratio + self.evaluation_ratio + self.test_ratio, 1.0, abs_tol=1e-9):
            raise ValueError("train/evaluation/test ratios must sum to 1")
        if self.per_answer < 0 or self.per_question_type < 0 or self.max_rows_per_source <= 0:
            raise ValueError("balancing caps must be non-negative and max rows positive")


@dataclass(frozen=True)
class SourceScene:
    source_id: str
    image_bytes: bytes
    scene: Scene
    annotations: List[Dict[str, Any]]
    seed: int


@dataclass(frozen=True)
class ObservedObject:
    source: SceneObject
    bbox: BBox
    score: float

    @property
    def uid(self) -> str:
        return self.source.uid

    @property
    def category(self) -> str:
        return self.source.category

    @property
    def number(self) -> Optional[int]:
        return self.source.number

    @property
    def center(self) -> Tuple[float, float]:
        return self.bbox.center


@dataclass
class GeometryState:
    scene: Scene
    observations: List[ObservedObject]
    retained_detections: List[Dict[str, Any]]
    report: Dict[str, Any]
    diagonal: float

    def objects(self, category: str) -> List[ObservedObject]:
        rows = [item for item in self.observations if item.category == category]
        return sorted(rows, key=lambda item: (item.number is None, item.number or 0, item.uid))

    def one(self, category: str) -> ObservedObject:
        rows = self.objects(category)
        if len(rows) != 1:
            raise GeometryRejected("requires exactly one detected %s, found %d" % (category, len(rows)))
        return rows[0]


@dataclass
class _BalanceReport:
    candidates: int
    selected: int
    heldout_removed: int
    per_split_type_before: Dict[str, int]
    per_split_type_after: Dict[str, int]
    answers_before: Dict[str, Dict[str, int]]
    answers_after: Dict[str, Dict[str, int]]
    shortfalls: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": self.candidates,
            "selected": self.selected,
            "heldout_removed_from_train": self.heldout_removed,
            "per_split_question_type_before": self.per_split_type_before,
            "per_split_question_type_after": self.per_split_type_after,
            "answers_before": self.answers_before,
            "answers_after": self.answers_after,
            "shortfalls": self.shortfalls,
        }


def _obj(
    uid: str,
    category: str,
    bbox: BBox,
    *,
    color: str,
    number: Optional[int] = None,
    anchor_uid: Optional[str] = None,
    z_index: int = 0,
) -> SceneObject:
    return SceneObject(
        uid=uid,
        category=category,
        bbox=bbox,
        color=color,
        number=number,
        sprite_name=category,
        anchor_uid=anchor_uid,
        z_index=z_index,
    )


def _centered_box(center: Tuple[float, float], size: Tuple[int, int]) -> BBox:
    return BBox(center[0] - size[0] / 2.0, center[1] - size[1] / 2.0, float(size[0]), float(size[1]))


def _distance(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _iou(left: BBox, right: BBox) -> float:
    x0, y0 = max(left.x, right.x), max(left.y, right.y)
    x1, y1 = min(left.x + left.width, right.x + right.width), min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = left.area + right.area - intersection
    return 0.0 if union <= 0 else intersection / union


def _point_segment_distance(point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return _distance(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator))
    return _distance(point, (start[0] + t * dx, start[1] + t * dy))


class SceneParameterSampler:
    """Samples concrete native-space scene parameters before sprite rendering."""

    def __init__(self, assets: AssetPack, seed: int) -> None:
        self.assets = assets
        self.seed = seed
        self.width, self.height = NATIVE_SIZE

    def _free(self, bbox: BBox, occupied: Iterable[BBox], *, margin: float = 35.0) -> bool:
        if not bbox.within(self.width, self.height):
            return False
        for other in occupied:
            separated = (
                bbox.x + bbox.width + margin <= other.x
                or other.x + other.width + margin <= bbox.x
                or bbox.y + bbox.height + margin <= other.y
                or other.y + other.height + margin <= bbox.y
            )
            if not separated:
                return False
        return True

    def _place_random(self, rng: random.Random, sprite: str, occupied: List[BBox], *, inset: int = 260) -> BBox:
        size = self.assets.size(sprite)
        for _ in range(800):
            center = (
                rng.uniform(inset + size[0] / 2.0, self.width - inset - size[0] / 2.0),
                rng.uniform(inset + size[1] / 2.0, self.height - inset - size[1] / 2.0),
            )
            bbox = _centered_box(center, size)
            if self._free(bbox, occupied, margin=80.0):
                return bbox
        raise RuntimeError("Could not sample a non-overlapping %s placement" % sprite)

    def _place_numbered_anchor(
        self, rng: random.Random, sprite: str, index: int, total: int, occupied: List[BBox]
    ) -> BBox:
        """Place same-class numbered anchors with an x-order safety margin."""

        size = self.assets.size(sprite)
        min_x = 260 + size[0] / 2.0
        max_x = self.width - 260 - size[0] / 2.0
        fraction = 0.5 if total == 1 else 0.04 + 0.92 * index / float(total - 1)
        center_x = min_x + (max_x - min_x) * fraction
        for _ in range(800):
            center_y = rng.uniform(260 + size[1] / 2.0, self.height - 260 - size[1] / 2.0)
            bbox = _centered_box((center_x, center_y), size)
            if self._free(bbox, occupied, margin=80.0):
                return bbox
        raise RuntimeError("Could not sample a safely numbered %s placement" % sprite)

    def _place_associated(
        self,
        rng: random.Random,
        sprite: str,
        anchor: SceneObject,
        occupied: List[BBox],
    ) -> BBox:
        size = self.assets.size(sprite)
        ax, ay = anchor.bbox.center
        for _ in range(300):
            angle = rng.uniform(0.0, math.tau)
            radius = rng.uniform(300.0, 470.0)
            bbox = _centered_box((ax + radius * math.cos(angle), ay + radius * math.sin(angle)), size)
            if self._free(bbox, occupied, margin=25.0):
                return bbox
        raise RuntimeError("Could not sample an associated %s placement" % sprite)

    def _clock_render_safe(self, bbox: BBox) -> bool:
        """Reserve room for rotated clock corners and the 215px heading dot."""

        cx, cy = bbox.center
        guard = 270.0
        return guard <= cx <= self.width - guard and guard <= cy <= self.height - guard

    def sample(self, index: int) -> SourceScene:
        # A source-specific RNG makes sampling independent of filtering order.
        rng = random.Random((self.seed << 20) + index)
        for restart in range(30):
            try:
                objects: List[SceneObject] = []
                occupied: List[BBox] = []
                benches: List[SceneObject] = []
                stops: List[SceneObject] = []
                bench_total = rng.randint(3, 5)
                for position in range(bench_total):
                    bbox = self._place_numbered_anchor(rng, "bench", position, bench_total, occupied)
                    bench = _obj("bench-p%d" % position, "bench", bbox, color="brown", z_index=20)
                    benches.append(bench)
                    objects.append(bench)
                    occupied.append(bbox)
                stop_total = rng.randint(2, 4)
                for position in range(stop_total):
                    bbox = self._place_numbered_anchor(rng, "stop sign", position, stop_total, occupied)
                    stop = _obj("stop-p%d" % position, "stop sign", bbox, color="red", z_index=20)
                    stops.append(stop)
                    objects.append(stop)
                    occupied.append(bbox)
                # IDs are visible overlays, so their sort order is part of the sampled state.
                for number, item in enumerate(sorted(benches, key=lambda obj: obj.bbox.center[0]), start=1):
                    objects[objects.index(item)] = _obj(item.uid, item.category, item.bbox, color=item.color, number=number, z_index=item.z_index)
                benches = [item for item in objects if item.category == "bench"]
                for number, item in enumerate(sorted(stops, key=lambda obj: obj.bbox.center[0]), start=1):
                    objects[objects.index(item)] = _obj(item.uid, item.category, item.bbox, color=item.color, number=number, z_index=item.z_index)
                stops = [item for item in objects if item.category == "stop sign"]

                people_counts: Dict[str, int] = {}
                person_index = 0
                for bench in benches:
                    count = rng.randint(0, 3)
                    people_counts[str(bench.number)] = count
                    for _ in range(count):
                        bbox = self._place_associated(rng, "person", bench, occupied)
                        person = _obj("person-p%d" % person_index, "person", bbox, color="blue", anchor_uid=bench.uid, z_index=30)
                        person_index += 1
                        objects.append(person)
                        occupied.append(bbox)
                animal_counts: Dict[str, int] = {}
                animal_index = 0
                animal_names = ("zebra", "elephant", "giraffe")
                for stop in stops:
                    count = rng.randint(0, 3)
                    animal_counts[str(stop.number)] = count
                    for _ in range(count):
                        category = rng.choice(animal_names)
                        bbox = self._place_associated(rng, category, stop, occupied)
                        animal = _obj("animal-p%d" % animal_index, category, bbox, color=category, anchor_uid=stop.uid, z_index=25)
                        animal_index += 1
                        objects.append(animal)
                        occupied.append(bbox)

                # Every fourth scene deliberately creates an arrival-positive candidate.
                arrival_mode = (index + restart) % 4
                clock_bbox: Optional[BBox] = None
                if arrival_mode == 0 and benches:
                    target = rng.choice(benches)
                    cx = target.bbox.x + target.bbox.width + self.assets.size("clock")[0] / 2.0 + rng.uniform(25.0, 60.0)
                    cy = target.bbox.center[1]
                    candidate = _centered_box((cx, cy), self.assets.size("clock"))
                    if self._free(candidate, occupied, margin=10.0) and self._clock_render_safe(candidate):
                        clock_bbox = candidate
                elif arrival_mode == 1:
                    animals = [item for item in objects if item.category in ANIMAL_CATEGORIES]
                    if animals:
                        target = rng.choice(animals)
                        cx = target.bbox.x + target.bbox.width + self.assets.size("clock")[0] / 2.0 + rng.uniform(20.0, 50.0)
                        candidate = _centered_box((cx, target.bbox.center[1]), self.assets.size("clock"))
                        if self._free(candidate, occupied, margin=10.0) and self._clock_render_safe(candidate):
                            clock_bbox = candidate
                if clock_bbox is None:
                    clock_bbox = self._place_random(rng, "clock", occupied, inset=360)
                # Avoidance prompts explicitly state that the red dot faces
                # their target.  Select that target at sampling time, then
                # encode the exact visible direction into the rendered dot.
                heading_mode = index % 4
                if heading_mode == 0:
                    heading_target = min(benches, key=lambda item: _distance(item.bbox.center, clock_bbox.center))
                elif heading_mode == 1:
                    # Dedicated source variants make the held-out
                    # closest-stop-sign avoidance template attainable.
                    heading_target = min(stops, key=lambda item: _distance(item.bbox.center, clock_bbox.center))
                elif heading_mode == 2:
                    heading_target = rng.choice(benches)
                else:
                    heading_target = rng.choice(stops)
                vector = (
                    heading_target.bbox.center[0] - clock_bbox.center[0],
                    heading_target.bbox.center[1] - clock_bbox.center[1],
                )
                vector_norm = math.hypot(*vector)
                heading = (vector[0] / vector_norm, vector[1] / vector_norm)
                heading_angle = math.degrees(math.atan2(-heading[1], heading[0])) % 360.0
                clock = _obj("clock-agent", "clock", clock_bbox, color="white", z_index=50)
                objects.append(clock)
                scene = Scene(
                    width=self.width,
                    height=self.height,
                    objects=objects,
                    heading=heading,
                    metadata={
                        "scene_family": "zoo-bus-parameter-sampled-v1",
                        "sampled_parameters": {
                            "scene_index": index,
                            "restart": restart,
                            "people_per_bench": people_counts,
                            "animals_per_stop_sign": animal_counts,
                            "arrival_mode": ("bench_near", "animal_near", "free", "free")[arrival_mode],
                            "heading_degrees": heading_angle,
                            "heading_target": {"category": heading_target.category, "number": heading_target.number},
                            "heading_target_mode": ("closest_bench", "closest_stop_sign", "bench", "stop_sign")[heading_mode],
                            "anchor_counts": {"benches": len(benches), "stop_signs": len(stops)},
                        },
                    },
                )
                image_bytes, output_scene = render_reference_scene(scene, self.assets)
                source_id = "zoo_bus_%08x_%05d.jpg" % (self.seed & 0xFFFFFFFF, index)
                return SourceScene(
                    source_id=source_id,
                    image_bytes=image_bytes,
                    scene=output_scene,
                    annotations=[annotation_from_object(item) for item in output_scene.objects],
                    seed=(self.seed << 16) + index,
                )
            except RuntimeError:
                continue
        raise RuntimeError("Failed to sample scene %d after 30 restarts" % index)


class DetectionGeometryFilter:
    """Cleans detector output and binds stable synthetic identities for QA."""

    def __init__(self, *, min_score: float, identity_iou: float) -> None:
        self.min_score = min_score
        self.identity_iou = identity_iou

    def prepare(self, scene: Scene, detections: Sequence[Mapping[str, Any]]) -> GeometryState:
        rejects: Counter[str] = Counter()
        candidates: List[Dict[str, Any]] = []
        for raw in detections:
            try:
                category = str(raw["category"])
                bbox = BBox(*[float(value) for value in raw["bbox"]])
                score = float(raw["score"])
            except (KeyError, TypeError, ValueError):
                rejects["malformed_detection"] += 1
                continue
            if not all(math.isfinite(value) for value in (bbox.x, bbox.y, bbox.width, bbox.height, score)):
                rejects["non_finite_detection"] += 1
            elif not 0.0 <= score <= 1.0:
                rejects["invalid_score"] += 1
            elif category not in DETECTED_CATEGORIES:
                rejects["unsupported_category"] += 1
            elif score < self.min_score:
                rejects["low_confidence"] += 1
            elif not bbox.within(scene.width, scene.height):
                rejects["out_of_bounds"] += 1
            else:
                candidates.append({"category": category, "bbox": bbox, "score": score})

        retained: List[Dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
            if any(
                candidate["category"] == existing["category"] and _iou(candidate["bbox"], existing["bbox"]) >= 0.75
                for existing in retained
            ):
                rejects["duplicate_nms"] += 1
                continue
            retained.append(candidate)

        sources_by_category: Dict[str, List[SceneObject]] = defaultdict(list)
        for item in scene.objects:
            sources_by_category[item.category].append(item)
        assigned: set[str] = set()
        identity_checked: List[Tuple[Dict[str, Any], SceneObject]] = []
        for detection in retained:
            possible = [item for item in sources_by_category[detection["category"]] if item.uid not in assigned]
            if not possible:
                rejects["unmatched_category"] += 1
                continue
            source, overlap = max(((item, _iou(detection["bbox"], item.bbox)) for item in possible), key=lambda item: item[1])
            if overlap < self.identity_iou:
                rejects["identity_iou"] += 1
                continue
            assigned.add(source.uid)
            # The generator box is a detector-quality gate only.  Question
            # geometry below is reconstructed from detector boxes and visible
            # target numbering (our renderer numbers anchors left-to-right),
            # never from this matched source object.
            identity_checked.append((detection, source))

        observations: List[ObservedObject] = []
        serialized: List[Dict[str, Any]] = []
        by_category: Dict[str, List[Tuple[Dict[str, Any], SceneObject]]] = defaultdict(list)
        for detection, source in identity_checked:
            by_category[detection["category"]].append((detection, source))
        for category, rows in sorted(by_category.items()):
            ordered = sorted(rows, key=lambda item: (item[0]["bbox"].center[0], item[0]["bbox"].center[1]))
            expected_order = sorted(rows, key=lambda item: (item[1].bbox.center[0], item[1].bbox.center[1]))
            incomplete_number_set = category in ANCHOR_CATEGORIES and len(rows) != len(sources_by_category[category])
            crossed_number_order = category in ANCHOR_CATEGORIES and [item[1].uid for item in ordered] != [item[1].uid for item in expected_order]
            unstable_number_order = category in ANCHOR_CATEGORIES and any(
                right[0]["bbox"].center[0] - left[0]["bbox"].center[0]
                < 1.25 * max(left[0]["bbox"].width, right[0]["bbox"].width)
                for left, right in zip(ordered, ordered[1:])
            )
            if incomplete_number_set or crossed_number_order or unstable_number_order:
                # Anchor number overlays are defined left-to-right.  When two
                # detector centers are close enough to exchange under normal
                # box jitter, no synthetic ID should be inferred from order.
                # Reject the whole category rather than renumbering a visible
                # ``bench 2`` to an invisible/incorrect ``bench 1``.
                reason = (
                    "incomplete_visible_number_set"
                    if incomplete_number_set
                    else "crossed_visible_number_order"
                    if crossed_number_order
                    else "unstable_visible_number_order"
                )
                rejects[reason] += len(ordered)
                continue
            for index, (detection, _) in enumerate(ordered, start=1):
                number = index if category in ANCHOR_CATEGORIES else None
                detected_identity = SceneObject(
                    uid="det:%s:%d" % (category.replace(" ", "_"), index),
                    category=category,
                    bbox=detection["bbox"],
                    color=category,
                    number=number,
                    sprite_name=category,
                )
                observations.append(ObservedObject(source=detected_identity, bbox=detection["bbox"], score=detection["score"]))
                serialized.append(
                    {
                        "area": detection["bbox"].area,
                        "bbox": detection["bbox"].to_list(),
                        "category": category,
                        "category_id": annotation_from_object(detected_identity)["category_id"],
                        "iscrowd": 0,
                        "score": detection["score"],
                    }
                )
        return GeometryState(
            scene=scene,
            observations=observations,
            retained_detections=serialized,
            report={
                "raw_detections": len(detections),
                "accepted_detections": len(observations),
                "rejected_detections": dict(sorted(rejects.items())),
            },
            diagonal=math.hypot(scene.width, scene.height),
        )


class ZooBusTemplateGenerator:
    """All 29 public Zoo-Bus templates evaluated over filtered detections."""

    def __init__(self, config: ReferenceBuildConfig) -> None:
        self.config = config

    def _margin(self, state: GeometryState, ratio: float) -> float:
        return state.diagonal * ratio

    def _targets(self, state: GeometryState, category: str) -> List[ObservedObject]:
        values = state.objects(category)
        if not values:
            raise GeometryRejected("no detected %s" % category)
        if any(item.number is None for item in values):
            raise GeometryRejected("target identifiers are unavailable")
        return values

    @staticmethod
    def _proof(rule: str, **values: Any) -> Dict[str, Any]:
        return {"rule": rule, **values}

    def _closest(self, state: GeometryState, targets: Sequence[ObservedObject]) -> Tuple[ObservedObject, Dict[str, float]]:
        clock = state.one("clock")
        values = {str(item.number): _distance(clock.center, item.center) for item in targets}
        ranked = sorted(targets, key=lambda item: values[str(item.number)])
        if len(ranked) > 1 and values[str(ranked[1].number)] - values[str(ranked[0].number)] < self._margin(state, self.config.nearest_margin_ratio):
            raise GeometryRejected("nearest-target margin is too small")
        return ranked[0], values

    def _ranked(self, state: GeometryState, targets: Sequence[ObservedObject]) -> Tuple[List[ObservedObject], Dict[str, float]]:
        clock = state.one("clock")
        values = {str(item.number): _distance(clock.center, item.center) for item in targets}
        ranked = sorted(targets, key=lambda item: values[str(item.number)])
        margin = self._margin(state, self.config.nearest_margin_ratio)
        if any(values[str(right.number)] - values[str(left.number)] < margin for left, right in zip(ranked, ranked[1:])):
            raise GeometryRejected("ranking distance margin is too small")
        return ranked, values

    def _associations(self, state: GeometryState, member_categories: Iterable[str], anchor_category: str) -> Dict[int, List[ObservedObject]]:
        anchors = self._targets(state, anchor_category)
        result: Dict[int, List[ObservedObject]] = {int(anchor.number): [] for anchor in anchors}
        margin = self._margin(state, self.config.association_margin_ratio)
        for category in member_categories:
            for member in state.objects(category):
                distances = sorted((_distance(member.center, anchor.center), anchor) for anchor in anchors)
                if len(distances) > 1 and distances[1][0] - distances[0][0] < margin:
                    raise GeometryRejected("%s-to-%s association is ambiguous" % (category, anchor_category))
                result[int(distances[0][1].number)].append(member)
        return result

    def _direction(self, state: GeometryState, target: ObservedObject) -> Tuple[str, float]:
        clock = state.one("clock")
        dx, dy = target.center[0] - clock.center[0], target.center[1] - clock.center[1]
        angle = math.degrees(math.atan2(-dy, dx)) % 360.0
        # Direction labels are centered at 0, 45, ... degrees; their decision
        # boundaries are offset by 22.5 degrees.
        sector_position = (angle + 22.5) % 45.0
        boundary_distance = min(sector_position, 45.0 - sector_position)
        if boundary_distance < self.config.direction_boundary_degrees:
            raise GeometryRejected("direction is too close to an 8-way boundary")
        return DIRECTION_LABELS[int((angle + 22.5) // 45.0) % 8], angle

    def _heading(self, state: GeometryState) -> Tuple[Tuple[float, float], float, str]:
        clock = state.one("clock")
        marker = state.one("heading marker")
        vector = (marker.center[0] - clock.center[0], marker.center[1] - clock.center[1])
        if math.hypot(*vector) < 1e-6:
            raise GeometryRejected("heading marker overlaps clock")
        angle = math.degrees(math.atan2(-vector[1], vector[0])) % 360.0
        sector_position = (angle + 22.5) % 45.0
        boundary_distance = min(sector_position, 45.0 - sector_position)
        if boundary_distance < self.config.direction_boundary_degrees:
            raise GeometryRejected("heading is too close to an 8-way boundary")
        return vector, angle, DIRECTION_LABELS[int((angle + 22.5) // 45.0) % 8]

    def _heading_aligned_target(self, state: GeometryState, category: str) -> ObservedObject:
        """Return the unique numbered target visually faced by the red dot."""

        clock = state.one("clock")
        marker = state.one("heading marker")
        heading_angle = math.degrees(math.atan2(-(marker.center[1] - clock.center[1]), marker.center[0] - clock.center[0])) % 360.0
        matches: List[ObservedObject] = []
        for target in self._targets(state, category):
            target_angle = math.degrees(math.atan2(-(target.center[1] - clock.center[1]), target.center[0] - clock.center[0])) % 360.0
            difference = abs((target_angle - heading_angle + 180.0) % 360.0 - 180.0)
            if difference <= self.config.direction_boundary_degrees:
                matches.append(target)
        if len(matches) != 1:
            raise GeometryRejected("red heading dot is not uniquely aligned to a %s" % category)
        return matches[0]

    def _arrival(self, state: GeometryState, target: ObservedObject) -> Tuple[str, Dict[str, float]]:
        clock = state.one("clock")
        center_distance = _distance(clock.center, target.center)
        surface_gap = max(0.0, center_distance - math.hypot(clock.bbox.width, clock.bbox.height) / 2.0 - math.hypot(target.bbox.width, target.bbox.height) / 2.0)
        threshold = max(clock.bbox.width, clock.bbox.height) * 0.22
        ambiguity = self._margin(state, 0.012)
        if abs(surface_gap - threshold) < ambiguity:
            raise GeometryRejected("arrival is too close to threshold")
        return ("Yes" if surface_gap < threshold else "No"), {"surface_gap": surface_gap, "threshold": threshold}

    def _avoidance(self, state: GeometryState, target: ObservedObject) -> Tuple[str, Dict[str, Any]]:
        clock = state.one("clock")
        blockers = [
            item
            for item in state.observations
            if item.uid not in {clock.uid, target.uid, "heading-marker"} and item.category != "heading marker"
        ]
        if not blockers:
            return "keep straight", {"blocker": None}
        candidates: List[Tuple[float, ObservedObject, float]] = []
        for item in blockers:
            radius = max(item.bbox.width, item.bbox.height) / 2.0
            distance = _point_segment_distance(item.center, clock.center, target.center)
            candidates.append((distance - radius, item, radius))
        clearance, obstacle, radius = min(candidates, key=lambda item: item[0])
        ambiguity = self._margin(state, 0.012)
        if abs(clearance) < ambiguity:
            raise GeometryRejected("obstacle path clearance is too close to threshold")
        if clearance > 0.0:
            return "keep straight", {"blocker": None, "nearest_clearance": clearance}
        heading = (target.center[0] - clock.center[0], target.center[1] - clock.center[1])
        relative = (obstacle.center[0] - clock.center[0], obstacle.center[1] - clock.center[1])
        cross = heading[0] * relative[1] - heading[1] * relative[0]
        if abs(cross) < state.diagonal * 12.0:
            raise GeometryRejected("obstacle is head-on and detour side is ambiguous")
        return (
            "turn left" if cross > 0 else "turn right",
            {"blocker": obstacle.uid, "blocker_category": obstacle.category, "path_clearance": clearance, "blocker_radius": radius, "cross_product_screen": cross},
        )

    def _turn(self, state: GeometryState, target: ObservedObject) -> Tuple[str, Dict[str, Any]]:
        clock = state.one("clock")
        heading, _, _ = self._heading(state)
        desired = (target.center[0] - clock.center[0], target.center[1] - clock.center[1])
        cross = heading[0] * desired[1] - heading[1] * desired[0]
        dot = heading[0] * desired[0] + heading[1] * desired[1]
        angle = abs(math.degrees(math.atan2(cross, dot)))
        if angle < self.config.direction_boundary_degrees:
            action = "keep straight"
        elif abs(abs(angle) - 180.0) < self.config.direction_boundary_degrees:
            raise GeometryRejected("turn is exactly behind the current heading")
        else:
            # Positive cross in image coordinates places the target right of heading.
            action = "turn right" if cross > 0 else "turn left"
        return action, {"cross_product_screen": cross, "dot_product": dot, "turn_angle_degrees": angle}

    def _relative_heading(self, state: GeometryState, target: ObservedObject) -> Tuple[str, Dict[str, Any]]:
        _, heading_angle, _ = self._heading(state)
        _, target_angle = self._direction(state, target)
        relative = (target_angle - heading_angle) % 360.0
        labels = ("front", "front-left", "left", "back-left", "back", "back-right", "right", "front-right")
        return labels[int((relative + 22.5) // 45.0) % 8], {"heading_degrees": heading_angle, "target_degrees": target_angle, "relative_degrees": relative}

    def generate_all(self, state: GeometryState) -> Tuple[List[QAPair], Dict[str, str]]:
        outputs: Dict[str, QAPair] = {}
        rejected: Dict[str, str] = {}

        def add(question_type: str, question: str, answer: str, proof: Dict[str, Any]) -> None:
            outputs[question_type] = QAPair(question_type, question, answer, {"evidence_source": "filtered_detections", **proof})

        def attempt(question_type: str, operation) -> None:
            try:
                operation()
            except GeometryRejected as exc:
                rejected[question_type] = str(exc)

        def count_people() -> None:
            people = state.objects("person")
            add("CountPeople", "How many people are visible in the scene?", str(len(people)), self._proof("count_category", category="person", count=len(people)))
        attempt("CountPeople", count_people)

        def count_animals() -> None:
            animals = [item for category in ANIMAL_CATEGORIES for item in state.objects(category)]
            add("CountAnimals", "How many animals are visible in the scene?", str(len(animals)), self._proof("count_categories", categories=sorted(ANIMAL_CATEGORIES), count=len(animals)))
        attempt("CountAnimals", count_animals)

        def people_at_bench() -> None:
            benches = self._targets(state, "bench")
            associations = self._associations(state, ("person",), "bench")
            target = benches[0]
            count = len(associations[int(target.number)])
            add("CountPeopleAtBench", "How many people are at bench %d?" % target.number, str(count), self._proof("stable_nearest_anchor_count", bench=target.number, count=count))
        attempt("CountPeopleAtBench", people_at_bench)

        def animals_at_stop() -> None:
            stops = self._targets(state, "stop sign")
            associations = self._associations(state, ANIMAL_CATEGORIES, "stop sign")
            target = stops[0]
            count = len(associations[int(target.number)])
            add("CountAnimalsAtStopSign", "How many animals are around stop sign %d?" % target.number, str(count), self._proof("stable_nearest_anchor_count", stop_sign=target.number, count=count))
        attempt("CountAnimalsAtStopSign", animals_at_stop)

        def list_benches() -> None:
            associations = self._associations(state, ("person",), "bench")
            k = 2 if any(len(values) >= 2 for values in associations.values()) else 1
            answer = [number for number, values in associations.items() if len(values) >= k]
            add("ListBenchesWithAtLeastKPeople", "Which benches have at least %d people?" % k, ", ".join(map(str, answer)) if answer else "none", self._proof("filter_anchor_counts", k=k, benches=answer))
        attempt("ListBenchesWithAtLeastKPeople", list_benches)

        def list_stops() -> None:
            associations = self._associations(state, ANIMAL_CATEGORIES, "stop sign")
            k = 2 if any(len(values) >= 2 for values in associations.values()) else 1
            answer = [number for number, values in associations.items() if len(values) >= k]
            add("ListStopSignsWithAtLeastKAnimals", "Which stop signs have at least %d animals?" % k, ", ".join(map(str, answer)) if answer else "none", self._proof("filter_anchor_counts", k=k, stop_signs=answer))
        attempt("ListStopSignsWithAtLeastKAnimals", list_stops)

        def arrived_bench() -> None:
            target = self._targets(state, "bench")[0]
            answer, metrics = self._arrival(state, target)
            add("ArrivedAtBench", "Has the clock arrived at bench %d?" % target.number, answer, self._proof("surface_distance_threshold", bench=target.number, **metrics))
        attempt("ArrivedAtBench", arrived_bench)

        def arrived_animals() -> None:
            stops = self._targets(state, "stop sign")
            associations = self._associations(state, ANIMAL_CATEGORIES, "stop sign")
            target = stops[0]
            animals = associations[int(target.number)]
            if not animals:
                raise GeometryRejected("target stop sign has no detected animals")
            outcomes = [self._arrival(state, animal) for animal in animals]
            answer = "Yes" if any(item[0] == "Yes" for item in outcomes) else "No"
            add("ArrivedAtAnimalsAroundStopSigns", "Has the clock arrived at an animal around stop sign %d?" % target.number, answer, self._proof("any_arrival_in_stable_group", stop_sign=target.number, animal_outcomes=[item[0] for item in outcomes]))
        attempt("ArrivedAtAnimalsAroundStopSigns", arrived_animals)

        def closest_bench() -> None:
            target, distances = self._closest(state, self._targets(state, "bench"))
            add("ClosestBench", "Which numbered bench is closest to the clock?", str(target.number), self._proof("minimum_center_distance", distances=distances, winner=target.number))
        attempt("ClosestBench", closest_bench)

        def closest_stop() -> None:
            target, distances = self._closest(state, self._targets(state, "stop sign"))
            add("ClosestStopSign", "Which numbered stop sign is closest to the clock?", str(target.number), self._proof("minimum_center_distance", distances=distances, winner=target.number))
        attempt("ClosestStopSign", closest_stop)

        def pairwise(category: str, qtype: str, noun: str) -> None:
            targets = self._targets(state, category)
            if len(targets) < 2:
                raise GeometryRejected("requires two %s targets" % noun)
            first, second = targets[0], targets[1]
            clock = state.one("clock")
            one, two = _distance(clock.center, first.center), _distance(clock.center, second.center)
            if abs(one - two) < self._margin(state, self.config.nearest_margin_ratio):
                raise GeometryRejected("pairwise distance margin is too small")
            answer = first if one < two else second
            add(qtype, "Which is closer to the clock, %s %d or %s %d?" % (noun, first.number, noun, second.number), str(answer.number), self._proof("pairwise_center_distance", first=first.number, second=second.number, distances={str(first.number): one, str(second.number): two}))
        attempt("PairwiseCloserBench", lambda: pairwise("bench", "PairwiseCloserBench", "bench"))
        attempt("PairwiseCloserStopSign", lambda: pairwise("stop sign", "PairwiseCloserStopSign", "stop sign"))

        def ranking(category: str, qtype: str, noun: str) -> None:
            targets, distances = self._ranked(state, self._targets(state, category))
            answer = ", ".join(str(item.number) for item in targets)
            add(qtype, "List all %ss from closest to farthest relative to the clock." % noun, answer, self._proof("stable_distance_ranking", distances=distances, order=[item.number for item in targets]))
        attempt("ClosestToFurthestBenches", lambda: ranking("bench", "ClosestToFurthestBenches", "bench"))
        attempt("ClosestToFurthestStopSigns", lambda: ranking("stop sign", "ClosestToFurthestStopSigns", "stop sign"))

        def geometric_direction(category: str, qtype: str, noun: str) -> None:
            target = self._targets(state, category)[0]
            answer, angle = self._direction(state, target)
            add(qtype, "What is the geometric direction from the clock to %s %d?" % (noun, target.number), answer, self._proof("quantize_relative_vector_to_8_directions", target=target.number, angle_degrees=angle, direction=answer))
        attempt("GeometricDirectionToBench", lambda: geometric_direction("bench", "GeometricDirectionToBench", "bench"))
        attempt("GeometricDirectionToStopSign", lambda: geometric_direction("stop sign", "GeometricDirectionToStopSign", "stop sign"))

        def avoidance(category: str, qtype: str, noun: str) -> None:
            target = self._heading_aligned_target(state, category)
            answer, metrics = self._avoidance(state, target)
            add(qtype, "With the red heading dot aligned to %s %d, how should the clock move while avoiding obstacles?" % (noun, target.number), answer, self._proof("segment_blockage_then_opposite_side_detour", target=target.number, **metrics))
        attempt("AvoidObstacleToReachBench", lambda: avoidance("bench", "AvoidObstacleToReachBench", "bench"))
        attempt("AvoidObstacleToReachStopSign", lambda: avoidance("stop sign", "AvoidObstacleToReachStopSign", "stop sign"))

        def heading_direction() -> None:
            _, angle, answer = self._heading(state)
            add("BusHeadingDirection", "Which compass direction is the clock currently facing?", answer, self._proof("heading_marker_vector_to_8_directions", angle_degrees=angle, direction=answer))
        attempt("BusHeadingDirection", heading_direction)

        def turn(category: str, qtype: str, noun: str) -> None:
            target = self._targets(state, category)[0]
            answer, metrics = self._turn(state, target)
            add(qtype, "How should the clock turn to face %s %d?" % (noun, target.number), answer, self._proof("heading_to_target_signed_turn", target=target.number, **metrics))
        attempt("TurnDirectionToBench", lambda: turn("bench", "TurnDirectionToBench", "bench"))
        attempt("TurnDirectionToStopSign", lambda: turn("stop sign", "TurnDirectionToStopSign", "stop sign"))

        def relative(category: str, qtype: str, noun: str) -> None:
            target = self._targets(state, category)[0]
            answer, metrics = self._relative_heading(state, target)
            add(qtype, "Where is %s %d relative to the clock's current heading?" % (noun, target.number), answer, self._proof("heading_relative_8_sector", target=target.number, **metrics))
        attempt("BenchRelativeToHeading", lambda: relative("bench", "BenchRelativeToHeading", "bench"))
        attempt("StopSignRelativeToHeading", lambda: relative("stop sign", "StopSignRelativeToHeading", "stop sign"))

        def people_at_closest() -> None:
            target, distances = self._closest(state, self._targets(state, "bench"))
            associations = self._associations(state, ("person",), "bench")
            answer = len(associations[int(target.number)])
            add("CountPersonAtClosestBench", "How many people are at the bench closest to the clock?", str(answer), self._proof("closest_then_stable_anchor_count", closest_bench=target.number, distances=distances, count=answer))
        attempt("CountPersonAtClosestBench", people_at_closest)

        def closest_with_person() -> None:
            associations = self._associations(state, ("person",), "bench")
            candidates = [item for item in self._targets(state, "bench") if associations[int(item.number)]]
            if not candidates:
                raise GeometryRejected("no bench has a detected person")
            target, distances = self._closest(state, candidates)
            add("ClosestBenchWithPerson", "Which bench with at least one person is closest to the clock?", str(target.number), self._proof("filter_people_then_minimum_distance", distances=distances, winner=target.number))
        attempt("ClosestBenchWithPerson", closest_with_person)

        def avoidance_closest(category: str, qtype: str, noun: str) -> None:
            target, distances = self._closest(state, self._targets(state, category))
            aligned = self._heading_aligned_target(state, category)
            if aligned.number != target.number:
                raise GeometryRejected("red heading dot is not aligned to the closest %s" % noun)
            answer, metrics = self._avoidance(state, target)
            add(qtype, "With the red heading dot aligned to the closest %s, how should the clock move while avoiding obstacles?" % noun, answer, self._proof("closest_then_segment_blockage", closest_target=target.number, distances=distances, **metrics))
        attempt("AvoidObstacleToReachClosestBench", lambda: avoidance_closest("bench", "AvoidObstacleToReachClosestBench", "bench"))
        attempt("AvoidObstacleToReachClosestStopSign", lambda: avoidance_closest("stop sign", "AvoidObstacleToReachClosestStopSign", "stop sign"))

        def direction_closest(category: str, qtype: str, noun: str) -> None:
            target, distances = self._closest(state, self._targets(state, category))
            answer, angle = self._direction(state, target)
            add(qtype, "What is the geometric direction to the closest %s?" % noun, answer, self._proof("closest_then_quantize_direction", closest_target=target.number, distances=distances, angle_degrees=angle, direction=answer))
        attempt("DirectionToClosestBench", lambda: direction_closest("bench", "DirectionToClosestBench", "bench"))
        attempt("DirectionToClosestStopSign", lambda: direction_closest("stop sign", "DirectionToClosestStopSign", "stop sign"))

        return [outputs[name] for name in QUESTION_TYPES if name in outputs], rejected


def _split_source_ids(source_ids: Sequence[str], config: ReferenceBuildConfig) -> Dict[str, str]:
    ordered = list(sorted(source_ids))
    rng = random.Random(config.seed)
    rng.shuffle(ordered)
    total = len(ordered)
    train_end = int(round(total * config.train_ratio))
    evaluation_end = train_end + int(round(total * config.evaluation_ratio))
    # With small diagnostic builds, retain non-empty evaluation/test when possible.
    if total >= 3:
        train_end = min(max(train_end, 1), total - 2)
        evaluation_end = min(max(evaluation_end, train_end + 1), total - 1)
    result: Dict[str, str] = {}
    for index, source_id in enumerate(ordered):
        result[source_id] = "train" if index < train_end else "evaluation" if index < evaluation_end else "test"
    return result


def _stable_key(sample: Sample, seed: int) -> str:
    return hashlib.sha256((str(seed) + "\0" + sample.source_id + "\0" + sample.question_type + "\0" + sample.answer).encode("utf-8")).hexdigest()


def _distribution(samples: Sequence[Sample], *, include_answer: bool) -> Dict[str, Any]:
    if include_answer:
        values: Dict[str, Counter[str]] = defaultdict(Counter)
        for item in samples:
            values["%s/%s" % (item.split, item.question_type)][item.answer] += 1
        return {key: dict(sorted(counter.items())) for key, counter in sorted(values.items())}
    return dict(sorted(Counter("%s/%s" % (item.split, item.question_type) for item in samples).items()))


def rebalance_rows(candidates: Sequence[Sample], config: ReferenceBuildConfig) -> Tuple[List[Sample], _BalanceReport]:
    before_type = _distribution(candidates, include_answer=False)
    before_answers = _distribution(candidates, include_answer=True)
    grouped: Dict[Tuple[str, str], Dict[str, List[Sample]]] = defaultdict(lambda: defaultdict(list))
    for item in candidates:
        grouped[(item.split, item.question_type)][item.answer].append(item)
    selected: List[Sample] = []
    shortfalls: Dict[str, int] = {}
    for (split, qtype), answer_buckets in sorted(grouped.items()):
        key = "%s/%s" % (split, qtype)
        answer_order = sorted(answer_buckets)
        prepared: Dict[str, List[Sample]] = {}
        for answer in answer_order:
            bucket = sorted(answer_buckets[answer], key=lambda item: _stable_key(item, config.seed))
            cap = config.per_answer or len(bucket)
            prepared[answer] = bucket[:cap]
            if config.per_answer and len(bucket) < config.per_answer:
                shortfalls["%s/%s" % (key, answer)] = config.per_answer - len(bucket)
        # Answer round-robin avoids a dominant answer claiming the whole type cap.
        type_cap = config.per_question_type or sum(len(values) for values in prepared.values())
        used_sources: set[str] = set()
        positions = {answer: 0 for answer in answer_order}
        while len([item for item in selected if item.split == split and item.question_type == qtype]) < type_cap:
            progressed = False
            for answer in answer_order:
                bucket = prepared[answer]
                while positions[answer] < len(bucket) and bucket[positions[answer]].source_id in used_sources:
                    positions[answer] += 1
                if positions[answer] >= len(bucket):
                    continue
                item = bucket[positions[answer]]
                positions[answer] += 1
                selected.append(item)
                used_sources.add(item.source_id)
                progressed = True
                if len([row for row in selected if row.split == split and row.question_type == qtype]) >= type_cap:
                    break
            if not progressed:
                break
        available = sum(len(values) for values in prepared.values())
        if config.per_question_type and available < config.per_question_type:
            shortfalls[key] = config.per_question_type - available

    # A final per-image cap protects against question-template concentration.
    source_counts: Counter[str] = Counter()
    limited: List[Sample] = []
    for item in sorted(selected, key=lambda row: (row.source_id, row.question_type, row.id)):
        if source_counts[item.source_id] < config.max_rows_per_source:
            source_counts[item.source_id] += 1
            limited.append(item)
    limited.sort(key=lambda item: item.id)
    after_type = _distribution(limited, include_answer=False)
    after_answers = _distribution(limited, include_answer=True)
    # Report the full requested split/type matrix, including types with no
    # candidate at all.  Train intentionally excludes compositional hold-outs.
    if config.per_question_type:
        for split in ("train", "evaluation", "test"):
            for qtype in QUESTION_TYPES:
                if split == "train" and config.keep_heldout_out_of_train and qtype in HELDOUT_QUESTION_TYPES:
                    continue
                key = "%s/%s" % (split, qtype)
                actual = int(after_type.get(key, 0))
                if actual < config.per_question_type:
                    shortfalls[key] = max(shortfalls.get(key, 0), config.per_question_type - actual)
    if config.per_answer:
        for key, answer_counts in before_answers.items():
            kept = after_answers.get(key, {})
            for answer, available in answer_counts.items():
                target = min(config.per_answer, int(available))
                actual = int(kept.get(answer, 0))
                if actual < target:
                    shortfalls["%s/%s" % (key, answer)] = max(
                        shortfalls.get("%s/%s" % (key, answer), 0), target - actual
                    )
    return limited, _BalanceReport(
        candidates=len(candidates),
        selected=len(limited),
        heldout_removed=0,
        per_split_type_before=before_type,
        per_split_type_after=after_type,
        answers_before=before_answers,
        answers_after=after_answers,
        shortfalls=dict(sorted(shortfalls.items())),
    )


def validate_reference_rows(
    rows: Sequence[Sample], source_splits: Mapping[str, str], config: ReferenceBuildConfig
) -> Dict[str, Any]:
    errors: List[str] = []
    ids = [item.id for item in rows]
    if len(ids) != len(set(ids)):
        errors.append("QA row ids are not unique")
    source_rows: Dict[str, List[Sample]] = defaultdict(list)
    pairs: set[Tuple[str, str]] = set()
    for item in rows:
        source_rows[item.source_id].append(item)
        if source_splits.get(item.source_id) != item.split:
            errors.append("%s has inconsistent source_id split" % item.source_id)
        if item.split == "train" and item.question_type in HELDOUT_QUESTION_TYPES:
            errors.append("held-out question type leaked into train: %s" % item.question_type)
        if item.question_type not in QUESTION_TYPES:
            errors.append("unsupported question type: %s" % item.question_type)
        if not item.proof.get("evidence_source") == "filtered_detections":
            errors.append("%d does not record detector-grounded proof" % item.id)
        if (item.source_id, item.question_type) in pairs:
            errors.append("duplicate question type for source %s: %s" % (item.source_id, item.question_type))
        pairs.add((item.source_id, item.question_type))
        if item.bo3 is not None or item.attempts:
            errors.append("reference build rows must not fabricate model scores")
        if not all(obj.bbox.within(item.scene.width, item.scene.height) for obj in item.scene.objects):
            errors.append("%s contains an output-state box out of bounds" % item.source_id)
    for source_id, samples in source_rows.items():
        if len({item.split for item in samples}) != 1:
            errors.append("%s leaks across splits" % source_id)
        if len({item.image_bytes for item in samples}) != 1:
            errors.append("%s QA rows do not share the same image bytes" % source_id)
        # Recompute retained question rows from the exported filtered boxes.
        # This prevents a stale/manual QA label from passing merely because its
        # proof claims detector provenance.
        state = DetectionGeometryFilter(
            min_score=config.min_detector_score, identity_iou=config.identity_iou
        ).prepare(samples[0].scene, samples[0].detections)
        recomputed, _ = ZooBusTemplateGenerator(config).generate_all(state)
        by_type = {item.question_type: item for item in recomputed}
        for item in samples:
            expected = by_type.get(item.question_type)
            if expected is None or (item.question, item.answer, item.proof) != (
                expected.question,
                expected.answer,
                expected.proof,
            ):
                errors.append("%s/%s does not recompute from exported detections" % (source_id, item.question_type))
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "rows": len(rows),
        "source_images": len(source_rows),
        "source_id_split_leakage": False,
        "heldout_train_leakage": False,
        "question_types": dict(sorted(Counter(item.question_type for item in rows).items())),
        "splits": dict(sorted(Counter(item.split for item in rows).items())),
    }


def build_reference_samples(
    config: ReferenceBuildConfig,
    *,
    asset_root: Optional[Path] = None,
    strict_assets: bool = False,
    detector: Optional[Detector] = None,
) -> Tuple[List[Sample], Dict[str, Any], str]:
    """Execute sampling -> render -> detect/filter -> QA -> split -> rebalance."""

    assets = AssetPack.from_root(asset_root, allow_compatibility=not strict_assets)
    sampler = SceneParameterSampler(assets, config.seed)
    sources = [sampler.sample(index) for index in range(config.num_scenes)]
    detector = detector or IdealDetectionReplay({source.source_id: source.scene for source in sources})
    validate_sources = getattr(detector, "validate_sources", None)
    if callable(validate_sources):
        validate_sources([source.source_id for source in sources])
    filterer = DetectionGeometryFilter(min_score=config.min_detector_score, identity_iou=config.identity_iou)
    templates = ZooBusTemplateGenerator(config)
    source_splits = _split_source_ids([source.source_id for source in sources], config)
    candidates: List[Sample] = []
    stage_rejections: Counter[str] = Counter()
    source_audit: List[Dict[str, Any]] = []
    row_id = 0
    for source in sources:
        try:
            raw_detections = detector.detect(DetectionInput(source.image_bytes, source.source_id))
        except AdapterUnavailableError:
            raise
        except Exception as exc:
            stage_rejections["detector_error"] += 1
            source_audit.append(
                {
                    "source_id": source.source_id,
                    "split": source_splits[source.source_id],
                    "sampled_parameters": source.scene.metadata.get("sampled_parameters", {}),
                    "detector_error": "%s: %s" % (type(exc).__name__, exc),
                    "qa_generated": [],
                    "qa_rejected": {},
                }
            )
            continue
        state = filterer.prepare(source.scene, raw_detections)
        pairs, rejected = templates.generate_all(state)
        stage_rejections.update(rejected.keys())
        source_audit.append(
            {
                "source_id": source.source_id,
                "split": source_splits[source.source_id],
                "sampled_parameters": source.scene.metadata.get("sampled_parameters", {}),
                "detector_filter": state.report,
                "qa_generated": [item.question_type for item in pairs],
                "qa_rejected": rejected,
            }
        )
        for pair in pairs:
            candidates.append(
                Sample(
                    id=row_id,
                    image_bytes=source.image_bytes,
                    annotations=source.annotations,
                    detections=state.retained_detections,
                    question=pair.question,
                    answer=pair.answer,
                    question_type=pair.question_type,
                    source_id=source.source_id,
                    split=source_splits[source.source_id],
                    capability="zoo_bus_reference",
                    group_id="source:%s" % source.source_id,
                    scene_family_id=source.source_id,
                    variant_role="source",
                    scene=source.scene,
                    difficulty_axes={
                        "filtering": "detector_and_geometry_stable",
                        "has_heading_marker_detection": bool(state.objects("heading marker")),
                    },
                    proof=pair.proof,
                    seed=source.seed,
                    generator_version=config.generator_version,
                )
            )
            row_id += 1

    heldout_removed = 0
    allowed: List[Sample] = []
    for item in candidates:
        if config.keep_heldout_out_of_train and item.split == "train" and item.question_type in HELDOUT_QUESTION_TYPES:
            heldout_removed += 1
            continue
        allowed.append(item)
    selected, balance = rebalance_rows(allowed, config)
    balance.heldout_removed = heldout_removed
    if not selected:
        raise RuntimeError("No stable QA rows were retained; inspect detector/filter rejection records")
    if config.strict_balance and balance.shortfalls:
        raise RuntimeError(
            "Requested balance is not attainable with the generated candidates; %d shortfall buckets remain"
            % len(balance.shortfalls)
        )
    # Reassign dense ids after filtering/rebalancing, just as final dataset exporters do.
    for identifier, item in enumerate(selected):
        item.id = identifier
    validation = validate_reference_rows(selected, source_splits, config)
    audit = {
        "config": {
            "num_scenes": config.num_scenes,
            "seed": config.seed,
            "min_detector_score": config.min_detector_score,
            "identity_iou": config.identity_iou,
            "per_answer": config.per_answer,
            "per_question_type": config.per_question_type,
            "max_rows_per_source": config.max_rows_per_source,
            "strict_balance": config.strict_balance,
            "heldout_question_types": sorted(HELDOUT_QUESTION_TYPES),
        },
        "generated_source_images": len(sources),
        "detector": detector.name,
        "detector_provenance": detector.provenance() if callable(getattr(detector, "provenance", None)) else {"backend": detector.name},
        "source_id_splits": dict(sorted(source_splits.items())),
        "filter_rejected_question_types": dict(sorted(stage_rejections.items())),
        "source_audit": source_audit,
        "balance": balance.to_dict(),
        "validation": validation,
    }
    return selected, audit, detector.name
