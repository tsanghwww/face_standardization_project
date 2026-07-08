"""Visualize one-image hard-zero vs Phase2 parameter standardization."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "original": (90, 90, 90),
    "hard_zero": (120, 120, 120),
    "stage2_xgb_weighted": (38, 166, 154),
    "standardize": (51, 160, 110),
    "weak_standardize": (245, 166, 35),
    "reject": (220, 80, 80),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--hard-zero-manifest", required=True, type=Path)
    parser.add_argument("--phase2-manifest", required=True, type=Path)
    parser.add_argument("--out-png", required=True, type=Path)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_first_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return rows[0]


def f(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except ValueError:
        return 0.0
    return value if math.isfinite(value) else 0.0


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    canvas = Image.new("RGB", size, (246, 246, 246))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def draw_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, label: str, original: float, value: float, color: tuple[int, int, int]) -> None:
    small = font(18)
    draw.text((x, y), label, fill=(40, 40, 40), font=small)
    bar_x = x + 170
    bar_y = y + 4
    draw.rectangle((bar_x, bar_y, bar_x + w, bar_y + 18), outline=(190, 190, 190), width=1)
    max_value = max(original, value, 1e-6)
    orig_w = int(original / max_value * w)
    val_w = int(value / max_value * w)
    draw.rectangle((bar_x, bar_y, bar_x + orig_w, bar_y + 8), fill=(190, 190, 190))
    draw.rectangle((bar_x, bar_y + 10, bar_x + val_w, bar_y + 18), fill=color)
    ratio = value / max(original, 1e-8)
    draw.text((bar_x + w + 18, y - 2), f"{value:.4f} / {original:.4f}  r={ratio:.3f}", fill=(45, 45, 45), font=small)


def draw_method_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    row: dict[str, str],
    color: tuple[int, int, int],
    decision: str,
) -> dict[str, float]:
    title_font = font(26)
    body = font(18)
    draw.rounded_rectangle((x, y, x + 620, y + 300), radius=10, outline=(210, 210, 210), width=2, fill=(255, 255, 255))
    draw.rectangle((x, y, x + 620, y + 46), fill=color)
    draw.text((x + 18, y + 9), title, fill="white", font=title_font)
    if decision:
        dcolor = COLORS.get(decision, (100, 100, 100))
        draw.rounded_rectangle((x + 410, y + 9, x + 600, y + 37), radius=6, fill=dcolor)
        draw.text((x + 424, y + 12), decision, fill="white", font=body)

    original_exp = f(row, "original_exp_norm")
    original_head = f(row, "original_head_pose_norm")
    original_jaw = f(row, "original_jaw_pose_norm")
    std_exp = f(row, "standardized_exp_norm")
    std_head = f(row, "standardized_head_pose_norm")
    std_jaw = f(row, "standardized_jaw_pose_norm")
    draw_bar(draw, x + 28, y + 78, 250, "expression", original_exp, std_exp, color)
    draw_bar(draw, x + 28, y + 126, 250, "head pose", original_head, std_head, color)
    draw_bar(draw, x + 28, y + 174, 250, "jaw pose", original_jaw, std_jaw, color)
    if "confidence" in row:
        draw.text((x + 28, y + 228), f"confidence={f(row, 'confidence'):.4f}   reject_score={f(row, 'reject_score'):.4f}", fill=(45, 45, 45), font=body)
        draw.text((x + 28, y + 255), f"alpha_exp={f(row, 'alpha_expression'):.4f}   alpha_head={f(row, 'alpha_head_pose'):.4f}   alpha_jaw={f(row, 'alpha_jaw_pose'):.4f}", fill=(45, 45, 45), font=body)
    else:
        draw.text((x + 28, y + 232), "hard-zero sets expression and pose exactly to 0", fill=(45, 45, 45), font=body)
    return {
        "original_exp_norm": original_exp,
        "original_head_pose_norm": original_head,
        "original_jaw_pose_norm": original_jaw,
        "standardized_exp_norm": std_exp,
        "standardized_head_pose_norm": std_head,
        "standardized_jaw_pose_norm": std_jaw,
    }


def main() -> None:
    args = parse_args()
    hard = read_first_row(args.hard_zero_manifest)
    phase2 = read_first_row(args.phase2_manifest)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1440, 960
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((40, 30), "Single Image Parameter Comparison", fill=(35, 35, 35), font=font(34))
    draw.text((42, 78), f"image_id={phase2.get('image_id', hard.get('image_id', 'unknown'))}", fill=(85, 85, 85), font=font(18))

    img = fit_image(args.image, (440, 680))
    canvas.paste(img, (50, 140))
    draw.text((50, 830), "Original input image", fill=(45, 45, 45), font=font(22))
    draw.text((50, 860), "Gray bar = original norm; colored bar = standardized norm", fill=(85, 85, 85), font=font(17))

    hard_metrics = draw_method_card(draw, 560, 150, "Hard-zero baseline", hard, COLORS["hard_zero"], "hard_zero")
    phase2_metrics = draw_method_card(
        draw,
        560,
        510,
        "Stage2 XGB-weighted Phase2",
        phase2,
        COLORS["stage2_xgb_weighted"],
        phase2.get("decision", ""),
    )
    canvas.save(args.out_png)

    payload = {
        "image": str(args.image),
        "hard_zero": hard_metrics,
        "stage2_xgb_weighted": {
            **phase2_metrics,
            "decision": phase2.get("decision", ""),
            "confidence": f(phase2, "confidence"),
            "reject_score": f(phase2, "reject_score"),
            "alpha_expression": f(phase2, "alpha_expression"),
            "alpha_head_pose": f(phase2, "alpha_head_pose"),
            "alpha_jaw_pose": f(phase2, "alpha_jaw_pose"),
        },
    }
    out_json = args.out_json or args.out_png.with_suffix(".json")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out_png": str(args.out_png), "out_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
