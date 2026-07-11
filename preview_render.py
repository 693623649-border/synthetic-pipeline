"""渲染预览：验证真实精灵加载 + 场景合成效果。

数据流：AssetPack.from_root(assets/) 加载 8 类真实精灵 →
SceneParameterSampler 随机采样场景参数 → render_reference_scene 合成 →
输出场景预览图与精灵对照表到 artifacts/。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from synthetic_vqa.rendering import AssetPack, NATIVE_SIZE
from synthetic_vqa.reference import SceneParameterSampler


def main() -> None:
    assets_dir = Path("assets").resolve()
    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    pack = AssetPack.from_root(assets_dir)
    print("is_reference:", pack.is_reference, "root:", pack.root)
    for name in ("background", "bench", "person", "stop sign", "zebra",
                 "elephant", "giraffe", "clock"):
        print("  %-10s %s" % (name, pack.size(name)))

    # 精灵对照表：把所有精灵贴到一张背景上，视觉检查风格一致性
    sheet = pack.image("background").convert("RGBA").resize((1280, 960), Image.Resampling.LANCZOS)
    layout = [
        ("bench", 40, 120), ("stop sign", 620, 90), ("person", 1100, 60),
        ("zebra", 40, 480), ("elephant", 420, 470), ("giraffe", 860, 430),
    ]
    for name, x, y in layout:
        spr = pack.image(name)
        # 缩放精灵到对照表可读尺寸（保留长边 ~260）
        scale = 260.0 / max(spr.size)
        spr_s = spr.resize((int(spr.width * scale), int(spr.height * scale)), Image.Resampling.LANCZOS)
        sheet.alpha_composite(spr_s, (x, y))
    sheet.convert("RGB").save(out / "sprite_sheet.jpg", quality=92)
    print("wrote", out / "sprite_sheet.jpg")

    # 采样 3 个随机场景，渲染预览
    sampler = SceneParameterSampler(pack, seed=42)
    for i in range(3):
        src = sampler.sample(i)
        (out / ("preview_scene_%d.jpg" % i)).write_bytes(src.image_bytes)
        sp = src.scene.metadata.get("sampled_parameters", {})
        print("scene %d: %s | %dx%d | %d objects | heading %.0f° | benches=%s stops=%s"
              % (i, src.source_id, src.scene.width, src.scene.height, len(src.scene.objects),
                 sp.get("heading_degrees", 0), sp.get("anchor_counts", {}).get("benches"),
                 sp.get("anchor_counts", {}).get("stop_signs")))


if __name__ == "__main__":
    main()
