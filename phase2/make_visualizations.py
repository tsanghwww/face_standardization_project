"""Create report-ready visualizations for Phase2 validation results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "hard_zero": (120, 120, 120),
    "stage1": (74, 144, 226),
    "stage2": (80, 190, 140),
    "stage3": (245, 166, 35),
    "stage1_xgb_weighted": (126, 87, 194),
    "stage2_xgb_weighted": (38, 166, 154),
    "stage3_xgb_weighted": (239, 83, 80),
    "standardize": (51, 160, 110),
    "weak_standardize": (245, 166, 35),
    "reject": (220, 80, 80),
    "high": (51, 160, 110),
    "medium": (245, 166, 35),
    "low": (220, 80, 80),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-table", required=True, type=Path)
    parser.add_argument("--comparison-summary", required=True, type=Path)
    parser.add_argument("--xgb-manifest", required=True, type=Path)
    parser.add_argument("--stage-manifest", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def font(size: int = 18) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def draw_title(draw: ImageDraw.ImageDraw, title: str, width: int) -> None:
    f = font(28)
    bbox = draw.textbbox((0, 0), title, font=f)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, 24), title, fill=(35, 35, 35), font=f)


def save_grouped_bar_chart(
    path: Path,
    title: str,
    groups: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    y_label: str,
    value_format: str = "{:.0f}",
) -> None:
    width, height = 1500, 840
    margin_l, margin_r, margin_t, margin_b = 110, 50, 95, 150
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, title, width)
    small = font(16)
    label_font = font(18)
    axis_font = font(20)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    max_value = max(max(values) for _, values, _ in series)
    max_value = max(max_value, 1.0)
    max_axis = max_value * 1.15

    for i in range(6):
        y = margin_t + plot_h - i * plot_h / 5
        value = max_axis * i / 5
        draw.line((margin_l, y, width - margin_r, y), fill=(230, 230, 230), width=1)
        draw.text((20, y - 10), value_format.format(value), fill=(80, 80, 80), font=small)
    draw.line((margin_l, margin_t, margin_l, margin_t + plot_h), fill=(80, 80, 80), width=2)
    draw.line((margin_l, margin_t + plot_h, width - margin_r, margin_t + plot_h), fill=(80, 80, 80), width=2)
    draw.text((18, margin_t - 40), y_label, fill=(60, 60, 60), font=axis_font)

    group_w = plot_w / len(groups)
    bar_gap = 5
    bar_w = min(34, (group_w - 28) / max(1, len(series)) - bar_gap)
    for g_idx, group in enumerate(groups):
        center_x = margin_l + group_w * (g_idx + 0.5)
        start_x = center_x - (len(series) * (bar_w + bar_gap) - bar_gap) / 2
        for s_idx, (name, values, color) in enumerate(series):
            value = values[g_idx]
            bar_h = value / max_axis * plot_h
            x0 = start_x + s_idx * (bar_w + bar_gap)
            y0 = margin_t + plot_h - bar_h
            draw.rectangle((x0, y0, x0 + bar_w, margin_t + plot_h), fill=color)
            if bar_h > 30:
                draw.text((x0 - 8, y0 - 22), value_format.format(value), fill=(50, 50, 50), font=small)
        draw.text((center_x - 65, margin_t + plot_h + 18), group, fill=(45, 45, 45), font=label_font)

    legend_x = margin_l
    legend_y = height - 82
    for name, _, color in series:
        draw.rectangle((legend_x, legend_y, legend_x + 22, legend_y + 22), fill=color)
        draw.text((legend_x + 30, legend_y - 2), name, fill=(45, 45, 45), font=small)
        legend_x += 210
    img.save(path)


def make_decision_chart(table_rows: list[dict[str, str]], out_dir: Path) -> None:
    runs = [row["run"] for row in table_rows if row["run"] != "hard_zero"]
    decisions = ["standardize", "weak_standardize", "reject"]
    series = []
    for decision in decisions:
        values = []
        for row in table_rows:
            if row["run"] == "hard_zero":
                continue
            counts = json.loads(row["decision_counts"])
            values.append(float(counts.get(decision, 0)))
        series.append((decision, values, COLORS[decision]))
    save_grouped_bar_chart(out_dir / "decision_counts.png", "Phase2 Decision Counts", runs, series, "samples")


def make_residual_ratio_chart(table_rows: list[dict[str, str]], out_dir: Path) -> None:
    rows = [row for row in table_rows if row["run"] != "hard_zero"]
    runs = [row["run"] for row in rows]
    series = [
        ("expression", [as_float(row["exp_ratio_mean"]) for row in rows], (74, 144, 226)),
        ("head pose", [as_float(row["head_pose_ratio_mean"]) for row in rows], (80, 190, 140)),
        ("jaw pose", [as_float(row["jaw_pose_ratio_mean"]) for row in rows], (245, 166, 35)),
    ]
    save_grouped_bar_chart(
        out_dir / "residual_ratios.png",
        "Mean Residual Ratio After Standardization",
        runs,
        series,
        "standardized / original",
        value_format="{:.2f}",
    )


def make_xgb_charts(xgb_rows: list[dict[str, str]], out_dir: Path) -> None:
    counts = Counter(row.get("xgb_quality_label", "unknown") for row in xgb_rows)
    labels = ["high", "medium", "low"]
    series = [("count", [float(counts[label]) for label in labels], (74, 144, 226))]
    save_grouped_bar_chart(out_dir / "xgb_quality_counts.png", "XGBoost Quality Labels", labels, series, "samples")

    scores = np.asarray([as_float(row["xgb_quality_score"]) for row in xgb_rows], dtype=np.float32)
    bins = np.linspace(0.0, 1.0, 21)
    hist, edges = np.histogram(scores, bins=bins)
    groups = [f"{edges[i]:.2f}" for i in range(len(hist))]
    series = [("score", [float(v) for v in hist], (126, 87, 194))]
    save_grouped_bar_chart(out_dir / "xgb_score_histogram.png", "XGBoost Quality Score Histogram", groups, series, "samples")


def choose_contact_rows(stage_rows: list[dict[str, str]], xgb_rows: dict[str, dict[str, str]], per_group: int = 2) -> list[dict[str, str]]:
    groups = [
        ("high", "standardize"),
        ("high", "reject"),
        ("medium", "standardize"),
        ("medium", "reject"),
        ("low", "standardize"),
        ("low", "reject"),
    ]
    chosen: list[dict[str, str]] = []
    for label, decision in groups:
        matches = [
            row
            for row in stage_rows
            if row.get("decision") == decision and xgb_rows.get(row["image_id"], {}).get("xgb_quality_label") == label
        ]
        matches.sort(key=lambda r: abs(as_float(r.get("quality_score", "0")) - 0.5))
        for row in matches[:per_group]:
            enriched = dict(row)
            enriched["xgb_quality_label"] = label
            enriched["xgb_quality_score"] = xgb_rows[row["image_id"]].get("xgb_quality_score", "")
            chosen.append(enriched)
    return chosen


def load_face(path: Path, size: int = 160) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_contact_sheet(stage_rows: list[dict[str, str]], xgb_rows: dict[str, dict[str, str]], images_dir: Path, out_dir: Path) -> None:
    rows = choose_contact_rows(stage_rows, xgb_rows)
    tile_w, tile_h = 220, 255
    cols = 4
    rows_n = math.ceil(len(rows) / cols)
    width = cols * tile_w + 50
    height = rows_n * tile_h + 90
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_title(draw, "Representative Samples: stage2_xgb_weighted", width)
    text_font = font(14)
    for idx, row in enumerate(rows):
        col = idx % cols
        r = idx // cols
        x = 25 + col * tile_w
        y = 82 + r * tile_h
        image_path = images_dir / f"{row['image_id']}.png"
        try:
            face = load_face(image_path)
        except Exception:
            face = Image.new("RGB", (160, 160), (235, 235, 235))
        img.paste(face, (x + 30, y))
        decision = row.get("decision", "")
        label = row.get("xgb_quality_label", "")
        color = COLORS.get(decision, (80, 80, 80))
        draw.rectangle((x + 30, y + 166, x + 190, y + 190), fill=color)
        draw.text((x + 36, y + 169), f"{label} / {decision}", fill="white", font=text_font)
        draw.text((x + 30, y + 198), f"id={row['image_id']} q={as_float(row.get('xgb_quality_score', '0')):.3f}", fill=(40, 40, 40), font=text_font)
        draw.text((x + 30, y + 218), f"conf={as_float(row.get('confidence', '0')):.3f} rej={as_float(row.get('reject_score', '0')):.3f}", fill=(40, 40, 40), font=text_font)
    img.save(out_dir / "representative_samples_stage2_xgb_weighted.png")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_rows = read_csv(args.comparison_table)
    xgb_list = read_csv(args.xgb_manifest)
    stage_rows = read_csv(args.stage_manifest)
    xgb_rows = {row["image_id"]: row for row in xgb_list}

    make_decision_chart(table_rows, args.out_dir)
    make_residual_ratio_chart(table_rows, args.out_dir)
    make_xgb_charts(xgb_list, args.out_dir)
    make_contact_sheet(stage_rows, xgb_rows, args.images_dir, args.out_dir)

    manifest = {
        "decision_counts": str(args.out_dir / "decision_counts.png"),
        "residual_ratios": str(args.out_dir / "residual_ratios.png"),
        "xgb_quality_counts": str(args.out_dir / "xgb_quality_counts.png"),
        "xgb_score_histogram": str(args.out_dir / "xgb_score_histogram.png"),
        "representative_samples": str(args.out_dir / "representative_samples_stage2_xgb_weighted.png"),
    }
    (args.out_dir / "visualization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
