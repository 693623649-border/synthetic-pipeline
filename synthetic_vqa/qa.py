from __future__ import annotations

import math
from typing import Tuple

from .models import QAPair, Scene, SceneObject


CAPABILITIES = ("perception", "counting", "spatial", "heading", "obstacle")


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_segment_distance(
    point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]
) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return _distance(point, start)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
    return _distance(point, (x1 + t * dx, y1 + t * dy))


class RuleQAGenerator:
    """Computes questions, answers, and auditable proofs from scene state."""

    def generate(self, scene: Scene, capability: str) -> QAPair:
        if capability == "perception":
            return QAPair(
                question_type="HeadingMarkerColor",
                question="What color is the dot that marks the clock's heading?",
                answer=scene.heading_marker_color,
                proof={
                    "rule": "read_scene_attribute",
                    "attribute": "heading_marker_color",
                    "value": scene.heading_marker_color,
                },
            )
        if capability == "counting":
            people = scene.find(category="person")
            return QAPair(
                question_type="CountPeople",
                question="How many people are visible in the scene?",
                answer=str(len(people)),
                proof={
                    "rule": "count_category",
                    "category": "person",
                    "object_ids": [obj.uid for obj in people],
                    "count": len(people),
                },
            )
        if capability == "spatial":
            clock = self._one(scene, category="clock")
            benches = scene.find(category="bench")
            distances = {str(obj.number): _distance(clock.bbox.center, obj.bbox.center) for obj in benches}
            closest = min(benches, key=lambda obj: distances[str(obj.number)])
            ordered = sorted(distances.values())
            margin = ordered[1] - ordered[0] if len(ordered) > 1 else float("inf")
            return QAPair(
                question_type="ClosestBench",
                question="Which numbered bench is closest to the clock?",
                answer=str(closest.number),
                proof={
                    "rule": "minimum_center_distance",
                    "clock_center": list(clock.bbox.center),
                    "bench_distances": distances,
                    "winner": closest.number,
                    "nearest_margin": margin,
                },
            )
        if capability == "heading":
            answer, angle = self._heading_label(scene.heading)
            return QAPair(
                question_type="BusHeadingDirection",
                question="Which compass direction is the clock currently facing?",
                answer=answer,
                proof={
                    "rule": "quantize_heading_to_8_directions",
                    "heading_vector": list(scene.heading),
                    "angle_degrees": angle,
                    "direction": answer,
                },
            )
        if capability == "obstacle":
            return self._obstacle_question(scene)
        raise ValueError("Unsupported capability: %s" % capability)

    @staticmethod
    def _one(scene: Scene, *, category: str = None, role: str = None) -> SceneObject:
        matches = scene.find(category=category, role=role)
        if len(matches) != 1:
            raise ValueError("Expected one object for category=%r role=%r, found %d" % (category, role, len(matches)))
        return matches[0]

    @staticmethod
    def _heading_label(heading: Tuple[float, float]) -> Tuple[str, float]:
        dx, dy = heading
        if dx == 0 and dy == 0:
            raise ValueError("Heading vector cannot be zero")
        angle = math.degrees(math.atan2(-dy, dx)) % 360.0
        labels = ("East", "Northeast", "North", "Northwest", "West", "Southwest", "South", "Southeast")
        return labels[int((angle + 22.5) // 45.0) % 8], angle

    def _obstacle_question(self, scene: Scene) -> QAPair:
        clock = self._one(scene, category="clock")
        target = self._one(scene, role="target")
        obstacle = self._one(scene, role="blocker")
        start, end, center = clock.bbox.center, target.bbox.center, obstacle.bbox.center
        path_distance = _point_segment_distance(center, start, end)
        blocking_radius = max(obstacle.bbox.width, obstacle.bbox.height) / 2.0
        if path_distance > blocking_radius:
            answer = "keep straight"
            cross = None
        else:
            heading = (end[0] - start[0], end[1] - start[1])
            relative = (center[0] - start[0], center[1] - start[1])
            cross = heading[0] * relative[1] - heading[1] * relative[0]
            if abs(cross) < 1e-9:
                raise ValueError("Ambiguous head-on obstacle")
            # Screen coordinates have positive y down. An obstacle on the right
            # side of travel requires a left detour, and vice versa.
            answer = "turn left" if cross > 0 else "turn right"
        return QAPair(
            question_type="AvoidObstacleToReachBench",
            question="How should the clock move to reach target bench 1 while avoiding the obstacle?",
            answer=answer,
            proof={
                "rule": "segment_blockage_then_opposite_side_detour",
                "clock_center": list(start),
                "target_center": list(end),
                "obstacle_center": list(center),
                "path_distance": path_distance,
                "blocking_radius": blocking_radius,
                "cross_product_screen": cross,
                "decision": answer,
            },
        )
