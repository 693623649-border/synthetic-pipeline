"""生成接近 zoo-bus-vqa 原数据集风格的参考图（PIL 几何合成）。

数据流：背景纹理 -> 动物层 -> 停止标志(绿标签) -> 长椅(黄标签) -> 时钟表盘+朝向红点
对齐原图配置：5 长椅、3 停止标志、4 大象+3 长颈鹿+1 斑马、罗马数字时钟、浅棕纸纹背景。
"""
import math, random
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 960
BG = (226, 214, 188)          # 浅棕哑光底
PAPER_DARK = (210, 197, 168)  # 纸纹暗调


def font(size, bold=True):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _b(xa, ya, xb, yb):
    """flip 安全的绘制框：返回 [min,min,max,max]，避免 x0>x1。"""
    return [min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)]


def draw_background(img):
    """浅棕底 + 随机纸纹噪点，模拟哑光纸质质感。"""
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, H], fill=BG)
    rnd = random.Random(7)
    for _ in range(9000):
        x, y = rnd.randint(0, W - 1), rnd.randint(0, H - 1)
        d.point((x, y), fill=PAPER_DARK if rnd.random() < 0.5 else (236, 226, 202))


def tag(d, rx, ry, text, fill, edge, text_color=(20, 20, 20)):
    """通用编号标签：彩色矩形 + 数字，rx/ry 为物体右下角坐标。"""
    tw, th = 34, 26
    d.rounded_rectangle([rx, ry, rx + tw, ry + th], radius=5, fill=fill, outline=edge, width=2)
    f = font(20)
    bb = d.textbbox((0, 0), text, font=f)
    d.text((rx + (tw - (bb[2] - bb[0])) / 2, ry + (th - (bb[3] - bb[1])) / 2 - 2),
           text, font=f, fill=text_color)


def draw_bench(d, cx, cy, number):
    """棕色长椅（椅背+椅面+支架）+ 右下角黄色编号标签。"""
    w, h = 120, 46
    x0, y0 = cx - w / 2, cy - h / 2
    d.rectangle([x0 + 12, y0 + h, x0 + 24, y0 + h + 16], fill=(60, 40, 22))     # 左支架
    d.rectangle([x0 + w - 24, y0 + h, x0 + w - 12, y0 + h + 16], fill=(60, 40, 22))  # 右支架
    d.rounded_rectangle([x0, y0, x0 + w, y0 + 14], radius=5, fill=(120, 72, 36), outline=(70, 42, 20), width=2)  # 椅背
    d.rounded_rectangle([x0, y0 + 14, x0 + w, y0 + h], radius=6, fill=(146, 92, 48), outline=(70, 42, 20), width=2)  # 椅面
    d.line([x0 + 12, y0 + h * 0.72, x0 + w - 12, y0 + h * 0.72], fill=(190, 130, 80), width=3)
    tag(d, x0 + w - 30, y0 + h - 6, str(number), (243, 200, 45), (120, 90, 10))  # 黄标签


def draw_stop_sign(d, cx, cy, number):
    """红色八角形 STOP + 杆 + 右下角绿色编号标签。"""
    r = 46
    pts = [(cx + r * math.cos(math.radians(22.5 + i * 45)),
            cy + r * math.sin(math.radians(22.5 + i * 45))) for i in range(8)]
    d.polygon(pts, fill=(205, 38, 36), outline=(110, 12, 12))
    f = font(26)
    bb = d.textbbox((0, 0), "STOP", font=f)
    d.text((cx - (bb[2] - bb[0]) / 2, cy - (bb[3] - bb[1]) / 2 - 2), "STOP", font=f, fill=(250, 250, 250))
    d.rectangle([cx - 5, cy + r, cx + 5, cy + r + 34], fill=(90, 90, 90))  # 杆
    tag(d, cx + r - 12, cy + r - 6, str(number), (60, 160, 70), (22, 90, 30), text_color=(255, 255, 255))  # 绿标签


def draw_elephant(d, cx, cy, flip=False):
    """简化卡通大象：灰身 + 圆头 + 扇耳 + 长鼻（flip=True 朝左）。"""
    body, body_edge = (150, 155, 165), (80, 85, 95)
    s = -1 if flip else 1
    d.ellipse(_b(cx - 46, cy - 26, cx + 34, cy + 30), fill=body, outline=body_edge, width=3)        # 身体
    d.ellipse(_b(cx + 20 * s, cy - 30, cx + 58 * s, cy + 8), fill=body, outline=body_edge, width=3)  # 头
    d.polygon([(cx + 24 * s, cy - 26), (cx + 44 * s, cy - 44), (cx + 40 * s, cy - 10)], fill=body, outline=body_edge)  # 耳
    d.arc(_b(cx + 40 * s, cy - 4, cx + 72 * s, cy + 40), 200, 340, fill=body_edge, width=8)          # 鼻
    for lx in (-0.32, -0.05, 0.22):
        d.rectangle([cx + lx * 90 - 7, cy + 22, cx + lx * 90 + 7, cy + 44], fill=(95, 100, 110))    # 腿


def draw_giraffe(d, cx, cy, flip=False):
    """简化卡通长颈鹿：长脖 + 小头 + 黄身 + 棕斑（flip=True 朝左）。"""
    body, edge, spot = (228, 185, 70), (150, 110, 30), (165, 120, 40)
    s = -1 if flip else 1
    d.ellipse(_b(cx - 34, cy - 6, cx + 34, cy + 38), fill=body, outline=edge, width=3)              # 身体
    d.polygon([(cx + 8 * s, cy + 4), (cx + 30 * s, cy - 54), (cx + 40 * s, cy - 54), (cx + 22 * s, cy + 8)], fill=body, outline=edge)  # 脖
    d.ellipse(_b(cx + 24 * s, cy - 66, cx + 52 * s, cy - 38), fill=body, outline=edge, width=3)     # 头
    for lx in (-0.18, 0.02, 0.22):
        d.rectangle([cx + lx * 90 - 5, cy + 34, cx + lx * 90 + 5, cy + 58], fill=(190, 145, 55))    # 腿
    for (sx, sy) in [(-14, 14), (4, 22), (16, 10), (-4, 28)]:
        d.ellipse(_b(cx + sx - 5, cy + sy - 5, cx + sx + 5, cy + sy + 5), fill=spot)                # 斑点


def draw_zebra(d, cx, cy, flip=False):
    """简化卡通斑马：白身 + 黑条纹（flip=True 朝左）。"""
    body, edge = (245, 245, 245), (30, 30, 30)
    s = -1 if flip else 1
    d.ellipse(_b(cx - 40, cy - 20, cx + 38, cy + 26), fill=body, outline=edge, width=3)             # 身体
    d.ellipse(_b(cx + 22 * s, cy - 26, cx + 54 * s, cy + 6), fill=body, outline=edge, width=3)      # 头
    for tx in range(-28, 28, 11):
        d.line([cx + tx, cy - 14, cx + tx, cy + 20], fill=edge, width=3)                            # 条纹
    for lx in (-0.25, 0.0, 0.22):
        d.rectangle([cx + lx * 80 - 5, cy + 20, cx + lx * 80 + 5, cy + 44], fill=(60, 60, 60))      # 腿


def draw_clock(d, cx, cy, r):
    """中央时钟：表壳 + 白盘 + 罗马数字 + 刻度 + 指针 + 底部红色朝向点。"""
    d.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], fill=(70, 75, 80))                 # 表壳
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(243, 240, 232), outline=(45, 50, 55), width=4)  # 白盘
    romans = ["XII", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]
    f = font(max(14, int(r * 0.22)))
    for i, label in enumerate(romans):  # 12 个罗马数字绕环排布
        ang = math.radians(-90 + i * 30)
        tx, ty = cx + (r * 0.76) * math.cos(ang), cy + (r * 0.76) * math.sin(ang)
        bb = d.textbbox((0, 0), label, font=f)
        d.text((tx - (bb[2] - bb[0]) / 2, ty - (bb[3] - bb[1]) / 2 - 1), label, font=f, fill=(30, 30, 30))
    for i in range(60):  # 60 格刻度
        ang = math.radians(-90 + i * 6)
        ir = r - 6 if i % 5 == 0 else r - 3
        d.line([cx + ir * math.cos(ang), cy + ir * math.sin(ang),
                cx + (r - 2) * math.cos(ang), cy + (r - 2) * math.sin(ang)],
               fill=(60, 60, 60), width=2 if i % 5 == 0 else 1)
    d.line([cx, cy, cx + math.cos(math.radians(-60)) * r * 0.5, cy + math.sin(math.radians(-60)) * r * 0.5], fill=(30, 30, 30), width=6)  # 时针
    d.line([cx, cy, cx + math.cos(math.radians(60)) * r * 0.72, cy + math.sin(math.radians(60)) * r * 0.72], fill=(30, 30, 30), width=4)   # 分针
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(30, 30, 30))                                 # 轴心
    d.ellipse([cx - 13, cy + r - 6, cx + 13, cy + r + 20], fill=(220, 30, 30), outline=(120, 10, 10))  # 底部红色朝向点


def main():
    img = Image.new("RGB", (W, H), BG)
    draw_background(img)
    d = ImageDraw.Draw(img, "RGBA")
    # 1) 动物层（4 大象 + 3 长颈鹿 + 1 斑马）
    draw_elephant(d, 340, 300, flip=False)
    draw_elephant(d, 430, 390, flip=False)
    draw_elephant(d, 940, 300, flip=True)
    draw_elephant(d, 850, 390, flip=True)
    draw_giraffe(d, 490, 190, flip=False)
    draw_giraffe(d, 790, 190, flip=True)
    draw_giraffe(d, 470, 760, flip=False)
    draw_zebra(d, 820, 760, flip=True)
    # 2) 停止标志（3 个，绿标签 1-3）
    draw_stop_sign(d, 175, 210, 1)
    draw_stop_sign(d, 640, 875, 2)
    draw_stop_sign(d, 1105, 210, 3)
    # 3) 长椅（5 个，黄标签 1-5）
    draw_bench(d, 175, 710, 1)
    draw_bench(d, 175, 540, 2)
    draw_bench(d, 640, 470, 3)
    draw_bench(d, 640, 640, 4)
    draw_bench(d, 1105, 540, 5)
    # 4) 中央时钟（罗马数字 + 朝向红点）
    draw_clock(d, 640, 250, 78)
    out = "artifacts/reference_demo.png"
    img.save(out, "PNG")
    print("saved:", out, img.size)


if __name__ == "__main__":
    main()
