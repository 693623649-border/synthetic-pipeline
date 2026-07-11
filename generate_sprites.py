"""Procedurally generate the 6 missing Zoo-Bus sprites into assets/.

风格对齐 Kenney Animal Pack (CC0) 的 outline 卡通风格：
  - 深灰褐粗外描边（描边完全包围图形外侧，由 MaxFilter 膨胀 alpha 通道实现）
  - 低饱和纯色填，无阴影/高光
  - 全圆角，高度简化

生成的 8 类精灵齐全后，rendering.AssetPack.from_root(assets/) 即返回
is_reference=True，reference.py 的整条复现管线（采样→渲染→检测→几何过滤→
29 种问题模板→分割→平衡→导出）即可走真实精灵合成路径。

依赖：仅 Pillow，与 requirements.txt 一致。
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


# ---------------------------------------------------------------------------
# 统一风格常量（对齐 Kenney outline）
# ---------------------------------------------------------------------------
OUTLINE = (58, 52, 48)          # 中性深灰褐外描边（对齐 Kenney，减少暖偏移）
OUTLINE_HALF = 6                # 描边半宽（像素），实际描边宽度约 2*OUTLINE_HALF

# 低饱和纯色调色板（对齐 Kenney 的低饱和卡通风格）
WOOD = (142, 92, 52)            # 长椅木色
WOOD_DARK = (104, 64, 36)       # 长椅阴影木纹（仅作细线，不算高光）
STOP_RED = (174, 60, 54)        # 停止标志砖红（降饱和）
STOP_TEXT = (248, 246, 240)     # STOP 文字米白
POLE = (116, 116, 122)          # 标志杆灰
CLOCK_FACE = (246, 244, 236)    # 钟面米白
CLOCK_INK = (48, 42, 40)        # 钟面墨色（指针/数字）
SKIN = (224, 172, 136)          # 人肤色
SHIRT = (74, 110, 154)          # 人上衣柔和蓝（降饱和）
PANTS = (46, 56, 92)            # 人裤子深蓝
ZEBRA_WHITE = (246, 246, 246)   # 斑马白
ZEBRA_BLACK = (44, 42, 42)      # 斑马黑条纹


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _b(xa: float, ya: float, xb: float, yb: float):
    """返回 [min, min, max, max]，防止椭圆/弧线坐标翻转。"""
    return [min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)]


def finish_outline(img: Image.Image, color=OUTLINE, half: int = OUTLINE_HALF) -> Image.Image:
    """对透明背景 RGBA 图层的 alpha 外轮廓做描边。

    数据流：取 alpha → MaxFilter 膨胀（半径=half）→ 减原 alpha 得描边带 →
    贴描边色 → 再贴原图层。这样描边完全包围图形外侧，匹配 Kenney 风格。
    """
    img = img.convert("RGBA")
    alpha = img.getchannel("A")
    window = half * 2 + 1
    dilated = alpha.filter(ImageFilter.MaxFilter(window))
    outline_mask = ImageChops.subtract(dilated, alpha)
    outline_layer = Image.new("RGBA", img.size, color + (255,))
    canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
    canvas.paste(outline_layer, mask=outline_mask)
    canvas.alpha_composite(img)
    return canvas


# ---------------------------------------------------------------------------
# zebra：白色身体 + 黑条纹，面朝右
# ---------------------------------------------------------------------------
def draw_zebra() -> Image.Image:
    W, H = 360, 280
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 4 条腿（圆角矩形）
    for lx, ly in [(70, 175), (150, 178), (235, 178), (300, 180)]:
        d.rounded_rectangle(_b(lx, ly, lx + 36, ly + 78), radius=12, fill=ZEBRA_WHITE)
    # 身体（大椭圆）
    d.ellipse(_b(48, 96, 312, 196), fill=ZEBRA_WHITE)
    # 脖颈
    d.ellipse(_b(250, 76, 330, 150), fill=ZEBRA_WHITE)
    # 头（右侧，面朝右）
    d.ellipse(_b(292, 66, 360, 132), fill=ZEBRA_WHITE)
    # 鬃毛（顶部锯齿三角）
    d.polygon([(150, 104), (172, 58), (194, 104), (216, 58), (238, 104), (260, 58), (282, 104)],
              fill=ZEBRA_WHITE)
    # 尾巴
    d.line([(50, 130), (24, 176)], fill=ZEBRA_WHITE, width=14)
    d.ellipse(_b(14, 166, 40, 192), fill=ZEBRA_WHITE)
    # 耳朵
    d.polygon([(316, 70), (330, 44), (340, 78)], fill=ZEBRA_WHITE)
    # 黑色条纹（画在身体上，只影响颜色不影响外轮廓 alpha）
    for cx in (96, 134, 172, 210, 248):
        d.ellipse(_b(cx - 8, 104, cx + 8, 188), fill=ZEBRA_BLACK)
    for cx in (110, 150, 190, 230):
        d.ellipse(_b(cx - 5, 110, cx + 5, 180), fill=ZEBRA_BLACK)
    # 头部条纹 + 眼睛
    d.ellipse(_b(316, 84, 326, 96), fill=ZEBRA_BLACK)
    d.ellipse(_b(336, 84, 346, 96), fill=ZEBRA_BLACK)
    d.ellipse(_b(300, 92, 312, 104), fill=(255, 255, 255))  # 眼白留空（在白头上不可见，省略）
    # 鼻孔
    d.ellipse(_b(344, 112, 354, 122), fill=ZEBRA_BLACK)
    return finish_outline(img)


# ---------------------------------------------------------------------------
# bench：木长椅（椅背 + 椅面 + 腿）
# ---------------------------------------------------------------------------
def draw_bench() -> Image.Image:
    W, H = 520, 230
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 椅背
    d.rounded_rectangle(_b(36, 20, 484, 96), radius=20, fill=WOOD)
    # 椅背竖向板缝（细线，低饱和暗色）
    for x in range(96, 484, 64):
        d.line([(x, 30), (x, 86)], fill=WOOD_DARK, width=4)
    # 椅面
    d.rounded_rectangle(_b(20, 96, 500, 150), radius=22, fill=WOOD)
    # 椅面板缝
    for x in range(80, 500, 70):
        d.line([(x, 104), (x, 142)], fill=WOOD_DARK, width=4)
    # 4 条腿
    for lx in (44, 238, 432):
        d.rounded_rectangle(_b(lx, 150, lx + 44, 214), radius=14, fill=WOOD)
    # 扶手端帽
    d.rounded_rectangle(_b(16, 86, 56, 150), radius=14, fill=WOOD)
    d.rounded_rectangle(_b(464, 86, 504, 150), radius=14, fill=WOOD)
    return finish_outline(img)


# ---------------------------------------------------------------------------
# stopSign：红色八角形 + STOP 白字 + 灰杆
# ---------------------------------------------------------------------------
def draw_stop_sign() -> Image.Image:
    W, H = 220, 300
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = 110, 110
    r = 96
    # 正八角形：顶点从 22.5° 起每 45°
    pts = [(cx + r * math.cos(math.radians(22.5 + i * 45)),
            cy + r * math.sin(math.radians(22.5 + i * 45))) for i in range(8)]
    d.polygon(pts, fill=STOP_RED)
    # 内描白边（装饰，低对比）
    inner = [(cx + (r - 12) * math.cos(math.radians(22.5 + i * 45)),
              cy + (r - 12) * math.sin(math.radians(22.5 + i * 45))) for i in range(8)]
    d.line(inner + [inner[0]], fill=(255, 255, 255), width=4)
    # STOP 文字
    font = _font(44)
    d.text((cx, cy), "STOP", font=font, fill=STOP_TEXT, anchor="mm")
    # 杆
    d.rounded_rectangle(_b(100, 200, 120, 296), radius=8, fill=POLE)
    return finish_outline(img)


# ---------------------------------------------------------------------------
# clock：罗马数字表盘 + 指针（10:10 造型）。渲染器会按朝向整体旋转。
# ---------------------------------------------------------------------------
def draw_clock() -> Image.Image:
    W = H = 300
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = 150
    r = 132
    # 表盘
    d.ellipse(_b(cx - r, cy - r, cx + r, cy + r), fill=CLOCK_FACE)
    # 罗马数字 I..XII
    numerals = ["XII", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
    font = _font(26)
    for i, label in enumerate(numerals):
        theta = math.radians(i * 30)           # 12 点起顺时针
        nr = r - 30
        x = cx + nr * math.sin(theta)
        y = cy - nr * math.cos(theta)
        d.text((x, y), label, font=font, fill=CLOCK_INK, anchor="mm")
    # 60 分钟刻度
    for i in range(60):
        theta = math.radians(i * 6)
        outer = (cx + (r - 8) * math.sin(theta), cy - (r - 8) * math.cos(theta))
        inner = (cx + (r - 16) * math.sin(theta), cy - (r - 16) * math.cos(theta))
        d.line([outer, inner], fill=CLOCK_INK, width=2 if i % 5 else 4)
    # 指针：10:10（时针指 10，分针指 2）
    hour_ang = math.radians(10 * 30)           # 时针角度（含分针修正略去，造型即可）
    min_ang = math.radians(2 * 30)
    d.line([(cx, cy), (cx + 62 * math.sin(hour_ang), cy - 62 * math.cos(hour_ang))],
           fill=CLOCK_INK, width=12)
    d.line([(cx, cy), (cx + 98 * math.sin(min_ang), cy - 98 * math.cos(min_ang))],
           fill=CLOCK_INK, width=9)
    # 中心轴
    d.ellipse(_b(cx - 12, cy - 12, cx + 12, cy + 12), fill=CLOCK_INK)
    return finish_outline(img)


# ---------------------------------------------------------------------------
# person：圆头 + 上衣 + 裤子，面朝前
# ---------------------------------------------------------------------------
def draw_person() -> Image.Image:
    W, H = 180, 420
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 头（放大，更接近 Kenney 卡通比例）
    d.ellipse(_b(46, 8, 134, 96), fill=SKIN)
    # 头发
    d.chord(_b(46, 8, 134, 96), 200, 340, fill=(74, 50, 38))
    # 眼睛
    d.ellipse(_b(66, 44, 78, 56), fill=(40, 30, 28))
    d.ellipse(_b(102, 44, 114, 56), fill=(40, 30, 28))
    # 上衣（躯干 + 两臂连成一体，避免分离描边）
    d.rounded_rectangle(_b(40, 90, 140, 252), radius=30, fill=SHIRT)
    # 裤子
    d.rounded_rectangle(_b(48, 238, 132, 392), radius=18, fill=PANTS)
    # 裤腿分隔
    d.line([(90, 250), (90, 388)], fill=(30, 38, 70), width=5)
    # 鞋
    d.rounded_rectangle(_b(46, 382, 92, 410), radius=12, fill=(50, 50, 55))
    d.rounded_rectangle(_b(88, 382, 134, 410), radius=12, fill=(50, 50, 55))
    return finish_outline(img)


# ---------------------------------------------------------------------------
# background：浅棕纸纹（4032×3024，JPEG）
# ---------------------------------------------------------------------------
def make_background() -> Image.Image:
    NATIVE = (4032, 3024)
    # 先在小尺寸上生成噪点纹理，再放大，控制开销
    tile = Image.new("RGB", (504, 378), (236, 224, 200))
    px = tile.load()
    rng = random.Random(2024)
    for y in range(tile.height):
        for x in range(tile.width):
            n = rng.randint(-12, 12)
            px[x, y] = (max(0, min(255, 236 + n)),
                        max(0, min(255, 224 + n)),
                        max(0, min(255, 200 + n)))
    tex = tile.resize(NATIVE, Image.Resampling.LANCZOS)
    # 叠加柔和的径向暗角，营造纸张感
    overlay = Image.new("RGBA", NATIVE, (0, 0, 0, 0))
    grad = ImageDraw.Draw(overlay)
    cx, cy = NATIVE[0] / 2, NATIVE[1] / 2
    max_r = math.hypot(cx, cy)
    steps = 40
    for i in range(steps):
        f = i / steps
        rad = int(max_r * (1.0 - f))
        alpha = int(70 * f * f)
        grad.ellipse(_b(cx - rad, cy - rad, cx + rad, cy + rad), fill=(120, 90, 60, alpha))
    out = Image.alpha_composite(tex.convert("RGBA"), overlay)
    return out.convert("RGB")


SPRITE_JOBS = {
    "zebra.png": draw_zebra,
    "bench.png": draw_bench,
    "stopSign.png": draw_stop_sign,
    "clock.png": draw_clock,
    "person.png": draw_person,
}


def main() -> None:
    assets = Path(__file__).resolve().parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for filename, fn in SPRITE_JOBS.items():
        dest = assets / filename
        img = fn()
        img.save(dest)
        print("wrote %-16s %s  %dx%d" % (dest.name, dest, img.width, img.height))
    bg = make_background()
    bg_path = assets / "background.jpeg"
    bg.save(bg_path, format="JPEG", quality=92)
    print("wrote %-16s %s  %dx%d" % (bg_path.name, bg_path, bg.width, bg.height))


if __name__ == "__main__":
    main()
