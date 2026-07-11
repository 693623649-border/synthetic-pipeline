from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


CATEGORY_IDS = {
    "person": 0,
    "bench": 13,
    "stop sign": 11,
    "elephant": 20,
    "zebra": 22,
    "giraffe": 23,
    "clock": 74,
    "obstacle": 1001,
    "heading marker": 1002,
}


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def within(self, width: int, height: int) -> bool:
        return (
            self.width > 0
            and self.height > 0
            and self.x >= 0
            and self.y >= 0
            and self.x + self.width <= width
            and self.y + self.height <= height
        )

    def to_list(self) -> List[float]:
        return [float(self.x), float(self.y), float(self.width), float(self.height)]

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "BBox":
        return cls(
            x=float(value["x"]),
            y=float(value["y"]),
            width=float(value["width"]),
            height=float(value["height"]),
        )


@dataclass(frozen=True)
class SceneObject:
    uid: str
    category: str
    bbox: BBox
    color: str
    number: Optional[int] = None
    role: Optional[str] = None
    sprite_name: Optional[str] = None
    anchor_uid: Optional[str] = None
    source_bbox: Optional[BBox] = None
    z_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "category": self.category,
            "bbox": self.bbox.to_dict(),
            "color": self.color,
            "number": self.number,
            "role": self.role,
            "sprite_name": self.sprite_name,
            "anchor_uid": self.anchor_uid,
            "source_bbox": None if self.source_bbox is None else self.source_bbox.to_dict(),
            "z_index": int(self.z_index),
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SceneObject":
        return cls(
            uid=str(value["uid"]),
            category=str(value["category"]),
            bbox=BBox.from_dict(value["bbox"]),
            color=str(value["color"]),
            number=None if value.get("number") is None else int(value["number"]),
            role=value.get("role"),
            sprite_name=value.get("sprite_name"),
            anchor_uid=value.get("anchor_uid"),
            source_bbox=None if value.get("source_bbox") is None else BBox.from_dict(value["source_bbox"]),
            z_index=int(value.get("z_index", 0)),
        )


@dataclass
class Scene:
    width: int
    height: int
    objects: List[SceneObject]
    heading: Tuple[float, float]
    heading_marker_color: str = "red"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def find(self, *, category: Optional[str] = None, role: Optional[str] = None) -> List[SceneObject]:
        return [
            obj
            for obj in self.objects
            if (category is None or obj.category == category)
            and (role is None or obj.role == role)
        ]

    def get(self, uid: str) -> SceneObject:
        for obj in self.objects:
            if obj.uid == uid:
                return obj
        raise KeyError(uid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "objects": [obj.to_dict() for obj in self.objects],
            "heading": [float(self.heading[0]), float(self.heading[1])],
            "heading_marker_color": self.heading_marker_color,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Scene":
        return cls(
            width=int(value["width"]),
            height=int(value["height"]),
            objects=[SceneObject.from_dict(item) for item in value["objects"]],
            heading=(float(value["heading"][0]), float(value["heading"][1])),
            heading_marker_color=str(value["heading_marker_color"]),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class QAPair:
    question_type: str
    question: str
    answer: str
    proof: Dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    attempt: int
    raw_answer: str
    normalized_answer: str
    correct: bool
    seed: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt": self.attempt,
            "raw_answer": self.raw_answer,
            "normalized_answer": self.normalized_answer,
            "correct": self.correct,
            "seed": self.seed,
            "error": self.error,
        }


@dataclass
class Sample:
    id: int
    image_bytes: bytes
    annotations: List[Dict[str, Any]]
    detections: List[Dict[str, Any]]
    question: str
    answer: str
    question_type: str
    source_id: str
    split: str
    capability: str
    group_id: str
    scene_family_id: str
    variant_role: str
    scene: Scene
    difficulty_axes: Dict[str, Any]
    proof: Dict[str, Any]
    seed: int
    generator_version: str = "smoke-v1"
    bo3: Optional[int] = None
    attempts: List[Prediction] = field(default_factory=list)


def annotation_from_object(obj: SceneObject) -> Dict[str, Any]:
    return {
        "area": float(obj.bbox.area),
        "bbox": obj.bbox.to_list(),
        "category": obj.category,
        "category_id": int(CATEGORY_IDS[obj.category]),
        "iscrowd": 0,
        "score": 1.0,
    }
