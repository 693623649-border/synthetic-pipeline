"""Reference-style Zoo-Bus scene rendering.

The released Zoo-Bus data was rendered by compositing a background and
transparent sprites, then resizing the native scene and adding object IDs to
the resized image.  This module keeps that contract explicit.  The reference
asset pack is intentionally an input: the original Hugging Face dataset does
not publish its private ``scene_gen/src`` assets, so the smoke runner creates
an in-memory compatibility pack when no asset root is supplied.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .models import BBox, Scene, SceneObject


NATIVE_SIZE = (4032, 3024)
OUTPUT_LONGEST_SIDE = 1280
HEADING_DOT_OFFSET = 215.0
HEADING_DOT_RADIUS = 45.0
HEADING_DOT_COLOR = (220, 30, 30, 255)

ASSET_FILENAMES = {
    "background": "background.jpeg",
    "bench": "bench.png",
    "person": "person.png",
    "stop sign": "stopSign.png",
    "zebra": "zebra.png",
    "elephant": "elephant.png",
    "giraffe": "giraffe.png",
    "clock": "clock.png",
}


def _rgba_sprite(size: Tuple[int, int], fill: Tuple[int, int, int, int], label: str) -> Image.Image:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x, y, w, h = 8, 8, size[0] - 16, size[1] - 16
    if label == "bench":
        draw.rounded_rectangle((x, y, x + w, y + h), radius=28, fill=fill, outline=(45, 25, 10, 255), width=10)
        draw.line((x + 25, y + h * 0.55, x + w - 25, y + h * 0.55), fill=(220, 155, 90, 255), width=12)
    elif label == "person":
        draw.ellipse((size[0] * 0.25, 8, size[0] * 0.75, size[0] * 0.28), fill=(226, 173, 135, 255), outline=(55, 35, 25, 255), width=5)
        draw.rounded_rectangle((size[0] * 0.18, size[1] * 0.23, size[0] * 0.82, size[1] - 8), radius=18, fill=fill, outline=(30, 40, 50, 255), width=6)
    elif label == "stop sign":
        cx, cy = size[0] / 2, size[1] * 0.4
        r = min(size[0], size[1]) * 0.36
        points = [(cx + r * math.cos(math.radians(22.5 + i * 45)), cy + r * math.sin(math.radians(22.5 + i * 45))) for i in range(8)]
        draw.polygon(points, fill=fill, outline=(80, 10, 10, 255))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, int(r * 0.35)))
        except OSError:
            font = ImageFont.load_default()
        draw.text((cx, cy), "STOP", font=font, fill="white", anchor="mm")
        draw.rectangle((cx - 12, cy + r, cx + 12, size[1] - 8), fill=(90, 90, 90, 255))
    elif label in {"zebra", "elephant", "giraffe"}:
        draw.ellipse((x, y + h * 0.20, x + w, y + h * 0.85), fill=fill, outline=(70, 55, 20, 255), width=8)
        draw.ellipse((x + w * 0.72, y, x + w, y + h * 0.42), fill=fill, outline=(70, 55, 20, 255), width=8)
        for leg_x in (0.25, 0.55, 0.78):
            draw.rectangle((x + w * leg_x, y + h * 0.72, x + w * leg_x + 22, y + h), fill=(75, 55, 25, 255))
    elif label == "clock":
        draw.ellipse((x, y, x + w, y + w), fill=fill, outline=(35, 40, 45, 255), width=12)
        cx, cy = size[0] / 2, size[1] / 2
        draw.line((cx, cy, cx, cy - size[1] * 0.22), fill=(25, 25, 25, 255), width=10)
        draw.line((cx, cy, cx + size[0] * 0.20, cy), fill=(25, 25, 25, 255), width=10)
    return image


def _compatibility_assets() -> Dict[str, Image.Image]:
    """Create a deterministic in-memory pack for local smoke tests.

    On the development machine, pass the actual Zoo-Bus ``scene_gen/src``
    directory to ``AssetPack``.  This fallback preserves the reference
    compositing/resizing/state contract but cannot reproduce the private asset
    pixels.
    """

    background = Image.new("RGBA", NATIVE_SIZE, (207, 213, 197, 255))
    draw = ImageDraw.Draw(background)
    for x in range(0, NATIVE_SIZE[0], 220):
        draw.line((x, 0, x, NATIVE_SIZE[1]), fill=(192, 201, 183, 255), width=4)
    for y in range(0, NATIVE_SIZE[1], 220):
        draw.line((0, y, NATIVE_SIZE[0], y), fill=(192, 201, 183, 255), width=4)
    return {
        "background": background,
        "bench": _rgba_sprite((520, 230), (130, 78, 38, 255), "bench"),
        "person": _rgba_sprite((180, 420), (45, 110, 205, 255), "person"),
        "stop sign": _rgba_sprite((180, 260), (205, 40, 38, 255), "stop sign"),
        "zebra": _rgba_sprite((300, 210), (225, 225, 225, 255), "zebra"),
        "elephant": _rgba_sprite((330, 250), (150, 155, 165, 255), "elephant"),
        "giraffe": _rgba_sprite((250, 360), (220, 170, 45, 255), "giraffe"),
        "clock": _rgba_sprite((300, 300), (235, 235, 235, 255), "clock"),
    }


@dataclass
class AssetPack:
    """Reference asset loader using the filenames from the Zoo-Bus generator."""

    images: Dict[str, Image.Image]
    root: Optional[Path] = None
    is_reference: bool = False

    @classmethod
    def from_root(cls, root: Optional[Path], *, allow_compatibility: bool = True) -> "AssetPack":
        if root is not None:
            root = Path(root).expanduser().resolve()
            missing = [filename for filename in ASSET_FILENAMES.values() if not (root / filename).is_file()]
            if missing:
                raise FileNotFoundError("Missing Zoo-Bus scene assets under %s: %s" % (root, ", ".join(missing)))
            images = {
                name: Image.open(root / filename).convert("RGBA")
                for name, filename in ASSET_FILENAMES.items()
            }
            return cls(images=images, root=root, is_reference=True)
        if not allow_compatibility:
            raise FileNotFoundError(
                "Zoo-Bus scene assets are required. Pass --asset-root pointing to scene_gen/src."
            )
        return cls(images=_compatibility_assets(), root=None, is_reference=False)

    def image(self, name: str) -> Image.Image:
        return self.images[name]

    def size(self, name: str) -> Tuple[int, int]:
        image = self.image(name)
        return image.width, image.height


def _scaled_bbox(bbox: BBox, scale: float) -> BBox:
    return BBox(bbox.x * scale, bbox.y * scale, bbox.width * scale, bbox.height * scale)


def _centered_bbox(center: Tuple[float, float], size: Tuple[int, int]) -> BBox:
    return BBox(center[0] - size[0] / 2.0, center[1] - size[1] / 2.0, float(size[0]), float(size[1]))


def rotate_clock(sprite: Image.Image, heading: Tuple[float, float]) -> Image.Image:
    hx, hy = heading
    angle = math.degrees(math.atan2(-hx, -hy))
    return sprite.rotate(angle, expand=True)


def _draw_ids(image: Image.Image, objects: Iterable[SceneObject], scale: float) -> None:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(12, int(48 * scale)))
    except OSError:
        font = ImageFont.load_default()
    for obj in objects:
        if obj.number is None or obj.category not in {"bench", "stop sign"}:
            continue
        x, y, w, h = obj.bbox.to_list()
        draw.text((x + w / 2.0, y + h / 2.0), str(obj.number), font=font, fill=(255, 255, 255), anchor="mm")


def render_reference_scene(scene: Scene, assets: AssetPack) -> Tuple[bytes, Scene]:
    """Composite sprites, resize to 1280-long-side, and return JPEG + output state."""

    native = assets.image("background").copy().convert("RGBA")
    if native.size != NATIVE_SIZE:
        native = native.resize(NATIVE_SIZE, Image.Resampling.LANCZOS)
    rendered_objects = []
    for obj in sorted(scene.objects, key=lambda item: item.z_index):
        if obj.category == "heading marker":
            continue
        sprite = assets.image(obj.sprite_name or obj.category)
        if obj.category == "clock":
            sprite = rotate_clock(sprite, scene.heading)
            bbox = _centered_bbox(obj.bbox.center, sprite.size)
        else:
            bbox = obj.bbox
        native.alpha_composite(sprite, (int(round(bbox.x)), int(round(bbox.y))))
        rendered_objects.append(
            SceneObject(
                uid=obj.uid,
                category=obj.category,
                bbox=bbox,
                color=obj.color,
                number=obj.number,
                role=obj.role,
                sprite_name=obj.sprite_name or obj.category,
                anchor_uid=obj.anchor_uid,
                source_bbox=obj.source_bbox or obj.bbox,
                z_index=obj.z_index,
            )
        )

    clock = next(obj for obj in rendered_objects if obj.category == "clock")
    hx, hy = scene.heading
    cx, cy = clock.bbox.center
    dot_center = (cx + hx * HEADING_DOT_OFFSET, cy + hy * HEADING_DOT_OFFSET)
    dot_color = HEADING_DOT_COLOR if scene.heading_marker_color == "red" else (30, 90, 220, 255)
    ImageDraw.Draw(native).ellipse(
        [
            dot_center[0] - HEADING_DOT_RADIUS,
            dot_center[1] - HEADING_DOT_RADIUS,
            dot_center[0] + HEADING_DOT_RADIUS,
            dot_center[1] + HEADING_DOT_RADIUS,
        ],
        fill=dot_color,
    )
    rendered_objects.append(
        SceneObject(
            uid="heading-marker",
            category="heading marker",
            bbox=BBox(dot_center[0] - HEADING_DOT_RADIUS, dot_center[1] - HEADING_DOT_RADIUS, HEADING_DOT_RADIUS * 2, HEADING_DOT_RADIUS * 2),
            color=scene.heading_marker_color,
            role="heading",
            source_bbox=BBox(dot_center[0] - HEADING_DOT_RADIUS, dot_center[1] - HEADING_DOT_RADIUS, HEADING_DOT_RADIUS * 2, HEADING_DOT_RADIUS * 2),
            z_index=10_000,
        )
    )

    scale = OUTPUT_LONGEST_SIDE / max(native.size)
    output_size = (int(round(native.width * scale)), int(round(native.height * scale)))
    output = native.resize(output_size, Image.Resampling.LANCZOS)
    output_objects = [
        SceneObject(
            uid=obj.uid,
            category=obj.category,
            bbox=_scaled_bbox(obj.bbox, scale),
            color=obj.color,
            number=obj.number,
            role=obj.role,
            sprite_name=obj.sprite_name,
            anchor_uid=obj.anchor_uid,
            source_bbox=obj.source_bbox or obj.bbox,
            z_index=obj.z_index,
        )
        for obj in rendered_objects
    ]
    _draw_ids(output, output_objects, scale)
    buffer = io.BytesIO()
    output.convert("RGB").save(buffer, format="JPEG", quality=95)
    output_scene = Scene(
        width=output.width,
        height=output.height,
        objects=output_objects,
        heading=scene.heading,
        heading_marker_color=scene.heading_marker_color,
        metadata={
            **scene.metadata,
            "renderer": "zoo_bus_sprite_compositor_v1",
            "asset_root": None if assets.root is None else str(assets.root),
            "reference_assets": assets.is_reference,
            "native_size": list(native.size),
            "output_size": list(output.size),
            "scale": scale,
            "heading_dot": {
                "offset_native": HEADING_DOT_OFFSET,
                "radius_native": HEADING_DOT_RADIUS,
                "color": list(dot_color[:3]),
            },
            "id_overlay": "bench_and_stop_sign_numbers_after_resize",
        },
    )
    return buffer.getvalue(), output_scene
