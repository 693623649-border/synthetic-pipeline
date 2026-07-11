from __future__ import annotations

import copy
import io
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter

from .models import BBox, Sample, Scene, SceneObject, annotation_from_object
from .qa import CAPABILITIES, RuleQAGenerator
from .rendering import AssetPack, render_reference_scene


WIDTH = 4032
HEIGHT = 3024
VARIANT_ROLES = ("base", "nuisance_brightness", "nuisance_compression", "counterfactual")


def _obj(
    uid: str,
    category: str,
    bbox: Tuple[float, float, float, float],
    color: str,
    number: int = None,
    role: str = None,
    sprite_name: str = None,
    anchor_uid: str = None,
    z_index: int = 0,
) -> SceneObject:
    return SceneObject(
        uid,
        category,
        BBox(*bbox),
        color,
        number,
        role,
        sprite_name or category,
        anchor_uid,
        None,
        z_index,
    )


def _replace_object(scene: Scene, uid: str, replacement: SceneObject) -> None:
    scene.objects = [replacement if obj.uid == uid else obj for obj in scene.objects]


def _sync_heading_marker(scene: Scene) -> None:
    clock = scene.find(category="clock")[0]
    cx, cy = clock.bbox.center
    dx, dy = scene.heading
    norm = math.hypot(dx, dy)
    mx, my = cx + dx / norm * 215.0, cy + dy / norm * 215.0
    marker = _obj(
        "heading-marker",
        "heading marker",
        (mx - 45.0, my - 45.0, 90.0, 90.0),
        scene.heading_marker_color,
        role="heading",
        sprite_name="heading marker",
        z_index=10_000,
    )
    if scene.find(category="heading marker"):
        _replace_object(scene, "heading-marker", marker)
    else:
        scene.objects.append(marker)


def _centered_bbox(assets: AssetPack, sprite_name: str, cx: float, cy: float) -> Tuple[float, float, float, float]:
    width, height = assets.size(sprite_name)
    return (cx - width / 2.0, cy - height / 2.0, float(width), float(height))


def base_scene(capability: str, assets: AssetPack) -> Scene:
    scene = Scene(
        width=WIDTH,
        height=HEIGHT,
        heading=(1.0, 0.0),
        heading_marker_color="red",
        metadata={"capability": capability},
        objects=[
            _obj("clock", "clock", _centered_bbox(assets, "clock", 2000, 1500), "white", role="agent", z_index=50),
            _obj("bench-1", "bench", _centered_bbox(assets, "bench", 1150, 850), "brown", number=1, z_index=20),
            _obj("bench-2", "bench", _centered_bbox(assets, "bench", 2500, 850), "brown", number=2, z_index=20),
            _obj("bench-3", "bench", _centered_bbox(assets, "bench", 3050, 2250), "brown", number=3, z_index=20),
            _obj("person-1", "person", _centered_bbox(assets, "person", 950, 1200), "blue", anchor_uid="bench-1", z_index=30),
            _obj("person-2", "person", _centered_bbox(assets, "person", 1350, 1200), "green", anchor_uid="bench-1", z_index=30),
            _obj("person-3", "person", _centered_bbox(assets, "person", 2450, 1200), "purple", anchor_uid="bench-2", z_index=30),
            _obj("stop-1", "stop sign", _centered_bbox(assets, "stop sign", 1050, 2200), "red", number=1, z_index=20),
            _obj("stop-2", "stop sign", _centered_bbox(assets, "stop sign", 3250, 1600), "red", number=2, z_index=20),
            _obj("animal-1", "zebra", _centered_bbox(assets, "zebra", 1450, 2200), "striped", anchor_uid="stop-1", z_index=25),
            _obj("animal-2", "elephant", _centered_bbox(assets, "elephant", 800, 1700), "gray", anchor_uid="stop-1", z_index=25),
            _obj("animal-3", "giraffe", _centered_bbox(assets, "giraffe", 3000, 1050), "yellow", anchor_uid="stop-2", z_index=25),
            _obj("animal-4", "zebra", _centered_bbox(assets, "zebra", 3500, 2200), "striped", anchor_uid="stop-2", z_index=25),
        ],
    )
    if capability == "obstacle":
        _replace_object(
            scene,
            "bench-1",
            _obj("bench-1", "bench", _centered_bbox(assets, "bench", 3050, 1500), "brown", number=1, role="target", z_index=20),
        )
        _replace_object(
            scene,
            "stop-1",
            _obj("stop-1", "stop sign", _centered_bbox(assets, "stop sign", 2500, 1580), "red", number=1, role="blocker", z_index=20),
        )
    _sync_heading_marker(scene)
    return scene


def counterfactual_scene(scene: Scene, capability: str, assets: AssetPack) -> Scene:
    changed = copy.deepcopy(scene)
    if capability == "perception":
        changed.heading_marker_color = "blue"
        changed.metadata["counterfactual_change"] = "heading_marker_color:red->blue"
        _sync_heading_marker(changed)
    elif capability == "counting":
        changed.objects.append(
            _obj("person-4", "person", _centered_bbox(assets, "person", 1750, 2200), "purple", anchor_uid="bench-3", z_index=30)
        )
        changed.metadata["counterfactual_change"] = "add_person"
    elif capability == "spatial":
        _replace_object(
            changed,
            "bench-1",
            _obj("bench-1", "bench", _centered_bbox(assets, "bench", 1950, 1300), "brown", number=1, z_index=20),
        )
        changed.metadata["counterfactual_change"] = "move_bench_1_closer"
    elif capability == "heading":
        changed.heading = (0.0, -1.0)
        changed.metadata["counterfactual_change"] = "heading:east->north"
        _sync_heading_marker(changed)
    elif capability == "obstacle":
        blocker = changed.get("stop-1")
        _replace_object(
            changed,
            "stop-1",
            _obj(
                "stop-1",
                "stop sign",
                _centered_bbox(assets, "stop sign", blocker.bbox.center[0], 1420),
                "red",
                number=1,
                role="blocker",
                z_index=20,
            ),
        )
        changed.metadata["counterfactual_change"] = "move_blocker_across_path"
    else:
        raise ValueError(capability)
    return changed


def apply_nuisance(image_bytes: bytes, role: str) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    if role == "nuisance_brightness":
        image = ImageEnhance.Brightness(image).enhance(0.72)
    elif role == "nuisance_compression":
        jpeg = io.BytesIO()
        image.save(jpeg, format="JPEG", quality=38)
        jpeg.seek(0)
        with Image.open(jpeg) as compressed:
            image = compressed.convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.65))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def generate_smoke_samples(
    seed: int = 42,
    asset_root: Optional[Path] = None,
    strict_assets: bool = False,
) -> List[Sample]:
    assets = AssetPack.from_root(asset_root, allow_compatibility=not strict_assets)
    qa = RuleQAGenerator()
    samples: List[Sample] = []
    sample_id = 0
    for capability_index, capability in enumerate(CAPABILITIES):
        group_id = "%s-000" % capability
        original = base_scene(capability, assets)
        counterfactual = counterfactual_scene(original, capability, assets)
        for role in VARIANT_ROLES:
            scene = counterfactual if role == "counterfactual" else copy.deepcopy(original)
            image_bytes, rendered_scene = render_reference_scene(scene, assets)
            pair = qa.generate(rendered_scene, capability)
            image_bytes = apply_nuisance(image_bytes, role)
            axes: Dict[str, object] = {
                "pixel_perturbation": role if role.startswith("nuisance_") else "none",
                "counterfactual": role == "counterfactual",
            }
            if role == "counterfactual":
                axes["changed_variable"] = scene.metadata["counterfactual_change"]
            samples.append(
                Sample(
                    id=sample_id,
                    image_bytes=image_bytes,
                    annotations=[annotation_from_object(obj) for obj in rendered_scene.objects],
                    detections=[],
                    question=pair.question,
                    answer=pair.answer,
                    question_type=pair.question_type,
                    source_id="%s__%s.jpg" % (group_id, role),
                    split="smoke",
                    capability=capability,
                    group_id=group_id,
                    scene_family_id="zoo-bus-geometric-smoke-v1",
                    variant_role=role,
                    scene=rendered_scene,
                    difficulty_axes=axes,
                    proof=pair.proof,
                    seed=seed + capability_index * 100,
                )
            )
            sample_id += 1
    return samples
