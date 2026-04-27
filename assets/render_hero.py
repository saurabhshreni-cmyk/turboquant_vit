"""Render a polished hero banner used at the top of the README."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


W, H = 1600, 480


def _gradient(width: int, height: int) -> Image.Image:
    """Diagonal purple gradient from deep navy → violet."""
    base = Image.new("RGB", (width, height), (15, 23, 42))
    px = base.load()
    for y in range(height):
        for x in range(width):
            t = ((x / max(width - 1, 1)) * 0.6
                 + (y / max(height - 1, 1)) * 0.4)
            r = int(15 + (76 - 15) * t)        # → 4c1d95
            g = int(23 + (29 - 23) * t)
            b = int(42 + (149 - 42) * t)
            px[x, y] = (r, g, b)
    return base


def _glow(width: int, height: int, cx: int, cy: int, radius: int,
          color: tuple[int, int, int]) -> Image.Image:
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 fill=color + (220,))
    return layer.filter(ImageFilter.GaussianBlur(radius // 2))


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    img = _gradient(W, H).convert("RGBA")
    img.alpha_composite(_glow(W, H, 280, 200, 380, (139, 92, 246)))
    img.alpha_composite(_glow(W, H, 1300, 320, 320, (124, 58, 237)))
    img.alpha_composite(_glow(W, H, 800, 80, 180, (167, 139, 250)))

    draw = ImageDraw.Draw(img)
    title_font = _font(96)
    sub_font = _font(34)
    pill_font = _font(22)

    title = "TurboQuant × ViT"
    sub = "KV-Cache Compression for Vision Transformers"
    pill_left = "ICLR 2026"
    pill_mid = "9.14× compression · 0.00% top-1 loss"
    pill_right = "Streamlit live demo"

    # Title
    tw = draw.textlength(title, font=title_font)
    draw.text(((W - tw) / 2, 130), title, font=title_font, fill=(245, 243, 255))

    # Subtitle
    sw = draw.textlength(sub, font=sub_font)
    draw.text(((W - sw) / 2, 250), sub, font=sub_font, fill=(196, 181, 253))

    # Pills row
    pills = [pill_left, pill_mid, pill_right]
    pill_widths = [draw.textlength(p, font=pill_font) for p in pills]
    pad_x, pad_y, gap = 22, 12, 28
    total = sum(pill_widths) + 2 * pad_x * len(pills) + gap * (len(pills) - 1)
    cursor = (W - total) // 2
    for text, w in zip(pills, pill_widths):
        x0, y0 = cursor, 340
        x1, y1 = cursor + w + 2 * pad_x, y0 + 50
        draw.rounded_rectangle([x0, y0, x1, y1], radius=25,
                               fill=(76, 29, 149, 200),
                               outline=(196, 181, 253), width=2)
        draw.text((x0 + pad_x, y0 + pad_y - 2), text, font=pill_font,
                  fill=(237, 233, 254))
        cursor = x1 + gap

    out = Path(__file__).parent / "hero.png"
    img.convert("RGB").save(out, optimize=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
