"""Create the data-rich figure plates used by the academic report.

The source PDFs are vector drawings produced with ReportLab.  They are then
rasterised at print resolution for reliable embedding in Word.  No chart is a
screen capture: every mark is regenerated from the competition data or the
versioned numerical evidence recorded in this repository.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


INCH = 72.0
FONT_DIR = Path(r"C:\Windows\Fonts")

INK = "17242D"
MUTED = "59666F"
GRID = "D8DEE3"
PAPER = "FFFFFF"
PANEL = "F5F7F8"
NAVY = "173B57"
BLUE = "2F6B8A"
TEAL = "2B7A78"
GOLD = "B58528"
BURGUNDY = "8B3A4A"
GREEN = "26734D"
RED = "A24A42"
LAVENDER = "6F6A8A"

CLASS_ORDER = ["functional", "functional needs repair", "non functional"]
CLASS_LABELS = ["Functional", "Repair", "Non-functional"]
CLASS_COLOURS = [TEAL, GOLD, BURGUNDY]


def _colour(value: str, alpha: float | None = None):
    from reportlab.lib.colors import Color, HexColor

    colour = HexColor(f"#{value}")
    if alpha is None:
        return colour
    return Color(colour.red, colour.green, colour.blue, alpha=alpha)


def register_fonts() -> None:
    font_specs = {
        "FigureSans": FONT_DIR / "cambria.ttc",
        "FigureSans-Bold": FONT_DIR / "cambriab.ttf",
        "FigureSans-Italic": FONT_DIR / "cambriai.ttf",
        "FigureSerif": FONT_DIR / "cambria.ttc",
        "FigureSerif-Bold": FONT_DIR / "cambriab.ttf",
        "FigureSerif-Italic": FONT_DIR / "cambriai.ttf",
    }
    for name, path in font_specs.items():
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(path)))


@dataclass
class FigureSurface:
    path: Path
    width_in: float
    height_in: float

    def __enter__(self) -> canvas.Canvas:
        register_fonts()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._canvas = canvas.Canvas(
            str(self.path), pagesize=(self.width_in * INCH, self.height_in * INCH)
        )
        self._canvas.setTitle(self.path.stem)
        self._canvas.setFillColor(_colour(PAPER))
        self._canvas.rect(0, 0, self.width_in * INCH, self.height_in * INCH, fill=1, stroke=0)
        return self._canvas

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._canvas.showPage()
            self._canvas.save()


def text_width(text: str, font: str = "FigureSans", size: float = 8) -> float:
    return pdfmetrics.stringWidth(str(text), font, size)


def wrap_text(text: str, max_width: float, font: str = "FigureSans", size: float = 8) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and text_width(candidate, font, size) > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def draw_text(
    c: canvas.Canvas,
    x: float,
    y: float,
    text: str,
    *,
    font: str = "FigureSans",
    size: float = 8,
    colour: str = INK,
    align: str = "left",
) -> None:
    c.setFont(font, size)
    c.setFillColor(_colour(colour))
    if align == "center":
        c.drawCentredString(x, y, str(text))
    elif align == "right":
        c.drawRightString(x, y, str(text))
    else:
        c.drawString(x, y, str(text))


def draw_wrapped(
    c: canvas.Canvas,
    x: float,
    y: float,
    text: str,
    max_width: float,
    *,
    font: str = "FigureSans",
    size: float = 8,
    colour: str = INK,
    leading: float | None = None,
    align: str = "left",
) -> float:
    leading = leading or size * 1.18
    lines = wrap_text(text, max_width, font, size)
    for index, line in enumerate(lines):
        draw_text(c, x, y - index * leading, line, font=font, size=size, colour=colour, align=align)
    return y - len(lines) * leading


def panel(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    title: str,
    *,
    fill: str = PANEL,
) -> None:
    c.setFillColor(_colour(fill))
    c.setStrokeColor(_colour(GRID))
    c.setLineWidth(0.55)
    c.roundRect(x, y, width, height, 5, fill=1, stroke=1)
    c.setFillColor(_colour(NAVY))
    c.circle(x + 13, y + height - 13, 7.5, fill=1, stroke=0)
    draw_text(c, x + 13, y + height - 16, label, font="FigureSans-Bold", size=7.5, colour=PAPER, align="center")
    draw_text(c, x + 26, y + height - 17, title, font="FigureSans-Bold", size=9.2, colour=INK)


def arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, colour: str = BLUE, width: float = 1.2) -> None:
    c.setStrokeColor(_colour(colour))
    c.setFillColor(_colour(colour))
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 5.5
    spread = 0.55
    points = [
        (x2, y2),
        (x2 - length * math.cos(angle - spread), y2 - length * math.sin(angle - spread)),
        (x2 - length * math.cos(angle + spread), y2 - length * math.sin(angle + spread)),
    ]
    path = c.beginPath()
    path.moveTo(*points[0])
    path.lineTo(*points[1])
    path.lineTo(*points[2])
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def draw_badge(c: canvas.Canvas, x: float, y: float, value: str, label: str, colour: str = NAVY) -> None:
    c.setFillColor(_colour(colour))
    c.roundRect(x, y, 67, 34, 4, fill=1, stroke=0)
    draw_text(c, x + 33.5, y + 18, value, font="FigureSans-Bold", size=12, colour=PAPER, align="center")
    draw_text(c, x + 33.5, y + 7, label.upper(), font="FigureSans-Bold", size=5.8, colour=PAPER, align="center")


def create_study_design(path: Path) -> None:
    with FigureSurface(path, 7.0, 3.45) as c:
        width, height = 7.0 * INCH, 3.45 * INCH
        draw_text(c, 14, height - 18, "From restricted competition data to a frozen public submission", font="FigureSans-Bold", size=11, colour=NAVY)
        draw_text(c, width - 14, height - 18, "STUDY DESIGN", font="FigureSans-Bold", size=6.5, colour=MUTED, align="right")

        stages = [
            (16, "DATA", "59,400 labelled\n14,850 unlabelled", BLUE),
            (114, "AUDIT", "schema, sentinels,\nhierarchies, leakage", TEAL),
            (212, "EVIDENCE", "5 fixed folds\n+ reserved local set", GOLD),
            (310, "SYNTHESIS", "6 representations\nweighted probabilities", LAVENDER),
            (408, "FREEZE", "hash, row order,\nclass validation", BURGUNDY),
        ]
        y, box_w, box_h = 116, 82, 76
        for index, (x, title, body, colour) in enumerate(stages):
            c.setFillColor(_colour(PAPER))
            c.setStrokeColor(_colour(colour))
            c.setLineWidth(1.2)
            c.roundRect(x, y, box_w, box_h, 6, fill=1, stroke=1)
            c.setFillColor(_colour(colour))
            c.rect(x, y + box_h - 18, box_w, 18, fill=1, stroke=0)
            draw_text(c, x + box_w / 2, y + box_h - 13, title, font="FigureSans-Bold", size=7, colour=PAPER, align="center")
            for line_no, line in enumerate(body.split("\n")):
                draw_text(c, x + box_w / 2, y + 35 - line_no * 12, line, font="FigureSans", size=7.2, colour=INK, align="center")
            if index < len(stages) - 1:
                arrow(c, x + box_w + 4, y + box_h / 2, stages[index + 1][0] - 5, y + box_h / 2, MUTED, 0.9)

        draw_text(c, 20, 96, "Every learned transformation was fitted inside the relevant training partition.", font="FigureSans-Italic", size=7.3, colour=MUTED)
        c.setStrokeColor(_colour(GRID))
        c.line(16, 86, width - 16, 86)

        draw_badge(c, 31, 33, "81.898%", "development")
        draw_badge(c, 113, 33, "81.397%", "used local", TEAL)
        draw_badge(c, 195, 33, "0.8298", "public score", BURGUNDY)
        draw_badge(c, 277, 33, "#2", "observed rank", GOLD)
        draw_badge(c, 359, 33, "14,850", "validated rows", LAVENDER)


def create_data_anatomy(path: Path, values: pd.DataFrame, labels: pd.DataFrame) -> None:
    df = values.merge(labels, on="id", validate="one_to_one")
    class_counts = labels["status_group"].value_counts().reindex(CLASS_ORDER)
    numeric_fields = ["amount_tsh", "gps_height", "population", "construction_year", "num_private", "longitude"]
    missing_rates = values[numeric_fields].isna().mean() * 100
    zero_rates = values[numeric_fields].eq(0).mean() * 100
    cardinality_fields = ["wpt_name", "subvillage", "scheme_name", "installer", "ward", "funder", "lga", "region"]
    cardinalities = values[cardinality_fields].nunique(dropna=False)
    known_year = values.loc[values["construction_year"] > 0, "construction_year"]
    decades = (known_year // 10 * 10).astype(int).value_counts().sort_index()

    with FigureSurface(path, 7.0, 5.25) as c:
        w, h = 7 * INCH, 5.25 * INCH
        panel(c, 10, 195, 238, 170, "A", "Class composition")
        total = int(class_counts.sum())
        x0, y0, bar_w, bar_h = 25, 290, 207, 23
        offset = x0
        for count, colour in zip(class_counts, CLASS_COLOURS):
            segment = bar_w * count / total
            c.setFillColor(_colour(colour))
            c.rect(offset, y0, segment, bar_h, fill=1, stroke=0)
            offset += segment
        for idx, (label, count, colour) in enumerate(zip(CLASS_LABELS, class_counts, CLASS_COLOURS)):
            yy = 266 - idx * 24
            c.setFillColor(_colour(colour))
            c.circle(29, yy + 3, 4, fill=1, stroke=0)
            draw_text(c, 39, yy, label, font="FigureSans-Bold", size=7.4)
            draw_text(c, 154, yy, f"{int(count):,}", font="FigureSans", size=7.4, align="right")
            draw_text(c, 229, yy, f"{count / total * 100:5.2f}%", font="FigureSans-Bold", size=7.4, colour=colour, align="right")
        draw_text(c, 25, 208, "Repair is 7.47x smaller than the functional class.", font="FigureSans-Italic", size=6.8, colour=MUTED)

        panel(c, 256, 195, 238, 170, "B", "Zeros are often semantic states")
        x_left, plot_right, value_x, y_top = 330, 450, 486, 322
        max_rate = 100
        for idx, field in enumerate(numeric_fields):
            yy = y_top - idx * 21
            draw_text(c, 268, yy - 2, field.replace("_", " "), font="FigureSans", size=6.8)
            c.setStrokeColor(_colour(GRID))
            c.setLineWidth(3.2)
            c.line(x_left, yy, plot_right, yy)
            zx = x_left + (plot_right - x_left) * zero_rates[field] / max_rate
            c.setStrokeColor(_colour(BURGUNDY))
            c.line(x_left, yy, zx, yy)
            c.setFillColor(_colour(BURGUNDY))
            c.circle(zx, yy, 3.2, fill=1, stroke=0)
            if missing_rates[field] > 0:
                mx = x_left + (plot_right - x_left) * missing_rates[field] / max_rate
                c.setFillColor(_colour(BLUE))
                c.circle(mx, yy, 2.2, fill=1, stroke=0)
            draw_text(c, value_x, yy - 2, f"{zero_rates[field]:.1f}%", font="FigureSans-Bold", size=6.5, colour=BURGUNDY, align="right")
        draw_text(c, 331, 205, "0", font="FigureSans", size=6, colour=MUTED)
        draw_text(c, plot_right, 205, "100%", font="FigureSans", size=6, colour=MUTED, align="right")

        panel(c, 10, 12, 238, 170, "C", "Categorical cardinality spans four orders")
        x_base, plot_end, value_x, y_top = 82, 195, 232, 141
        max_log = math.log10(max(cardinalities))
        for idx, field in enumerate(cardinality_fields):
            yy = y_top - idx * 16.2
            draw_text(c, 23, yy - 2, field.replace("_", " "), font="FigureSans", size=6.5)
            c.setStrokeColor(_colour(GRID))
            c.setLineWidth(1)
            c.line(x_base, yy, plot_end, yy)
            xx = x_base + (plot_end - x_base) * math.log10(cardinalities[field]) / max_log
            c.setFillColor(_colour(TEAL if cardinalities[field] < 3000 else GOLD))
            c.circle(xx, yy, 3, fill=1, stroke=0)
            draw_text(c, value_x, yy - 2, f"{int(cardinalities[field]):,}", font="FigureSans-Bold", size=6.2, align="right")
        draw_text(c, 23, 16, "Log-scaled axis; counts include explicit missing states.", font="FigureSans-Italic", size=5.8, colour=MUTED)

        panel(c, 256, 12, 238, 170, "D", "Construction history is partly unrecorded")
        chart_x, chart_y, chart_w, chart_h = 272, 42, 206, 99
        decade_keys = [d for d in decades.index if 1960 <= d <= 2010]
        values_y = [int(decades.get(d, 0)) for d in decade_keys]
        unknown = int((values["construction_year"] == 0).sum())
        labels_x = ["unknown"] + [str(d)[2:] + "s" for d in decade_keys]
        plot_values = [unknown] + values_y
        max_value = max(plot_values)
        gap = 3
        bw = (chart_w - gap * (len(plot_values) - 1)) / len(plot_values)
        for idx, (lab, value) in enumerate(zip(labels_x, plot_values)):
            bh = chart_h * value / max_value
            x = chart_x + idx * (bw + gap)
            c.setFillColor(_colour(GOLD if idx == 0 else BLUE))
            c.rect(x, chart_y, bw, bh, fill=1, stroke=0)
            draw_text(c, x + bw / 2, chart_y - 10, lab, font="FigureSans", size=5.7, colour=MUTED, align="center")
        draw_text(c, chart_x, 151, f"{unknown:,} rows (34.9%) use year 0", font="FigureSans-Bold", size=7.1, colour=GOLD)
        draw_text(c, chart_x, 23, "Known construction years are summarised by decade.", font="FigureSans-Italic", size=6.2, colour=MUTED)


def create_spatial_heterogeneity(path: Path, values: pd.DataFrame, labels: pd.DataFrame) -> None:
    df = values.merge(labels, on="id", validate="one_to_one")
    valid = df[df["longitude"].between(28, 42) & df["latitude"].between(-13, 0)].copy()
    region_mix = pd.crosstab(df["region"], df["status_group"], normalize="index").reindex(columns=CLASS_ORDER).fillna(0)
    # Reproduce the geographic audit's ordering: highest non-functional share first.
    region_mix = region_mix.sort_values("non functional", ascending=False)

    with FigureSurface(path, 7.0, 5.35) as c:
        panel(c, 10, 12, 286, 360, "A", "Mapped label distribution")
        x0, y0, plot_w, plot_h = 27, 45, 251, 288
        lon_min, lon_max = 28.5, 41.5
        lat_min, lat_max = -12.2, -0.8
        longitude_scale = math.cos(math.radians((lat_min + lat_max) / 2))
        projected_width = (lon_max - lon_min) * longitude_scale
        projected_height = lat_max - lat_min
        scale = min(plot_w / projected_width, plot_h / projected_height)
        map_w = projected_width * scale
        map_h = projected_height * scale
        map_x = x0 + (plot_w - map_w) / 2
        map_y = y0 + (plot_h - map_h) / 2
        c.saveState()
        c.rect(map_x, map_y, map_w, map_h, fill=0, stroke=0)
        try:
            c.setFillAlpha(0.28)
        except AttributeError:
            pass
        for status, colour in zip(CLASS_ORDER, CLASS_COLOURS):
            subset = valid[valid["status_group"] == status]
            c.setFillColor(_colour(colour))
            for lon, lat in subset[["longitude", "latitude"]].itertuples(index=False, name=None):
                x = map_x + (lon - lon_min) * longitude_scale * scale
                y = map_y + (lat - lat_min) * scale
                if map_x <= x <= map_x + map_w and map_y <= y <= map_y + map_h:
                    c.circle(x, y, 0.36, fill=1, stroke=0)
        try:
            c.setFillAlpha(1)
        except AttributeError:
            pass
        c.restoreState()
        c.setStrokeColor(_colour(GRID))
        c.rect(map_x, map_y, map_w, map_h, fill=0, stroke=1)
        for idx, (lab, colour) in enumerate(zip(CLASS_LABELS, CLASS_COLOURS)):
            lx = 31 + idx * 82
            c.setFillColor(_colour(colour))
            c.circle(lx, 29, 3.2, fill=1, stroke=0)
            draw_text(c, lx + 7, 27, lab, font="FigureSans", size=6.4)
        draw_text(c, 282, 27, f"n={len(valid):,}", font="FigureSans-Bold", size=6.4, colour=MUTED, align="right")

        panel(c, 304, 12, 190, 360, "B", "Regional class composition")
        label_x, bar_x, bar_w, y_top = 315, 380, 102, 335
        row_h = 13.5
        for idx, (region, row) in enumerate(region_mix.iterrows()):
            yy = y_top - idx * row_h
            draw_text(c, label_x, yy - 1.5, region[:15], font="FigureSans", size=5.65)
            offset = bar_x
            for status, colour in zip(CLASS_ORDER, CLASS_COLOURS):
                segment = bar_w * row[status]
                c.setFillColor(_colour(colour))
                c.rect(offset, yy - 2, segment, 7, fill=1, stroke=0)
                offset += segment
        draw_text(c, bar_x, 28, "0", font="FigureSans", size=5.5, colour=MUTED)
        draw_text(c, bar_x + bar_w, 28, "100%", font="FigureSans", size=5.5, colour=MUTED, align="right")
        draw_text(c, 315, 17, "Ordered by non-functional share", font="FigureSans-Italic", size=5.3, colour=MUTED)


def create_validation_design(path: Path) -> None:
    with FigureSurface(path, 7.0, 4.25) as c:
        w, h = 7 * INCH, 4.25 * INCH
        draw_text(c, 14, h - 19, "Evidence hierarchy and leakage boundary", font="FigureSans-Bold", size=11, colour=NAVY)

        c.setFillColor(_colour(PANEL))
        c.roundRect(17, 176, 470, 93, 7, fill=1, stroke=0)
        draw_text(c, 29, 250, "LABELLED DATA  |  59,400 rows", font="FigureSans-Bold", size=8.3, colour=NAVY)
        c.setFillColor(_colour(BLUE))
        c.roundRect(30, 196, 335, 38, 4, fill=1, stroke=0)
        draw_text(c, 197.5, 216, "DEVELOPMENT 47,520", font="FigureSans-Bold", size=9, colour=PAPER, align="center")
        draw_text(c, 197.5, 204, "candidate selection on five fixed stratified folds", font="FigureSans", size=6.4, colour=PAPER, align="center")
        c.setFillColor(_colour(GOLD))
        c.roundRect(375, 196, 96, 38, 4, fill=1, stroke=0)
        draw_text(c, 423, 216, "LOCAL 11,880", font="FigureSans-Bold", size=8.5, colour=PAPER, align="center")
        draw_text(c, 423, 204, "bounded confirmation", font="FigureSans", size=6.1, colour=PAPER, align="center")

        fold_y = 129
        draw_text(c, 29, 158, "OUTER-FOLD PROTOCOL", font="FigureSans-Bold", size=7, colour=MUTED)
        for fold in range(5):
            x = 30 + fold * 68
            for cell in range(5):
                c.setFillColor(_colour(GOLD if cell == fold else TEAL))
                c.rect(x + cell * 11.5, fold_y, 10, 20, fill=1, stroke=0)
            draw_text(c, x + 28, fold_y - 11, f"fold {fold + 1}", font="FigureSans", size=5.8, colour=MUTED, align="center")
        draw_text(c, 373, 141, "Inner cross-fit", font="FigureSans-Bold", size=7, colour=TEAL)
        draw_text(c, 373, 129, "target-aware transforms", font="FigureSans", size=6.2, colour=MUTED)
        arrow(c, 350, 139, 367, 139, TEAL, 0.9)
        draw_text(c, 373, 108, "Outer validation", font="FigureSans-Bold", size=7, colour=GOLD)
        draw_text(c, 373, 96, "never informs fitting", font="FigureSans", size=6.2, colour=MUTED)

        c.setFillColor(_colour("F8F2E7"))
        c.setStrokeColor(_colour(GOLD))
        c.roundRect(17, 26, 470, 49, 6, fill=1, stroke=1)
        draw_text(c, 29, 58, "UNLABELLED COMPETITION SET  |  14,850 rows", font="FigureSans-Bold", size=7.8, colour=GOLD)
        draw_text(c, 29, 43, "Excluded from supervised selection; used only for frozen full-data prediction and structural validation.", font="FigureSans", size=6.7, colour=INK)

        draw_text(c, 252, 105, "promotion gate", font="FigureSans-Bold", size=6.4, colour=NAVY, align="center")
        draw_text(c, 252, 92, ">=0.10 pp, >=3 folds, worst-fold and recall limits", font="FigureSans", size=5.6, colour=MUTED, align="center")
        arrow(c, 252, 86, 252, 77, NAVY, 1.0)


MODEL_POINTS = {
    "Decision tree": (74.98, 15.17),
    "Logistic": (75.06, 13.96),
    "KNN": (77.89, 31.99),
    "Extra Trees": (78.84, 39.40),
    "Hist. boost": (80.21, 28.69),
    "Random Forest": (80.59, 36.86),
    "50:50 soft vote": (80.83, 36.39),
}

FOLD_POINTS = {
    "Decision tree": [74.68, 75.12, 75.33, 74.76, 75.02],
    "Extra Trees": [78.67, 78.86, 78.11, 79.63, 78.92],
    "Hist. boost": [80.05, 80.12, 79.95, 80.77, 80.16],
    "Random Forest": [80.88, 80.39, 80.17, 81.05, 80.47],
    "50:50 soft vote": [81.00, 80.78, 80.17, 81.51, 80.70],
}


def create_model_family_evidence(path: Path) -> None:
    with FigureSurface(path, 7.0, 4.75) as c:
        panel(c, 10, 12, 286, 316, "A", "Accuracy and repair-recall trade-off")
        x0, y0, pw, ph = 48, 55, 225, 225
        xmin, xmax, ymin, ymax = 74, 82, 10, 45
        for tick in [74, 76, 78, 80, 82]:
            xx = x0 + (tick - xmin) / (xmax - xmin) * pw
            c.setStrokeColor(_colour(GRID))
            c.line(xx, y0, xx, y0 + ph)
            draw_text(c, xx, y0 - 13, tick, font="FigureSans", size=6.2, colour=MUTED, align="center")
        for tick in [10, 20, 30, 40]:
            yy = y0 + (tick - ymin) / (ymax - ymin) * ph
            c.setStrokeColor(_colour(GRID))
            c.line(x0, yy, x0 + pw, yy)
            draw_text(c, x0 - 6, yy - 2, tick, font="FigureSans", size=6.2, colour=MUTED, align="right")
        for idx, (name, (acc, recall)) in enumerate(MODEL_POINTS.items()):
            xx = x0 + (acc - xmin) / (xmax - xmin) * pw
            yy = y0 + (recall - ymin) / (ymax - ymin) * ph
            colour = GREEN if name == "50:50 soft vote" else (GOLD if name == "Extra Trees" else BLUE)
            c.setFillColor(_colour(colour))
            c.circle(xx, yy, 4.1, fill=1, stroke=0)
            dx = -5 if name in ("Extra Trees", "Random Forest") else 5
            align = "right" if dx < 0 else "left"
            draw_text(c, xx + dx, yy + (6 if idx % 2 == 0 else -10), name, font="FigureSans-Bold", size=5.9, colour=colour, align=align)
        draw_text(c, x0 + pw / 2, 30, "mean five-fold accuracy (%)", font="FigureSans-Bold", size=6.8, colour=MUTED, align="center")
        c.saveState()
        c.translate(18, y0 + ph / 2)
        c.rotate(90)
        draw_text(c, 0, 0, "repair recall (%)", font="FigureSans-Bold", size=6.8, colour=MUTED, align="center")
        c.restoreState()
        draw_text(c, 24, 23, "Diagnostic outliers omitted from axes: majority 54.31/0.00; Gaussian NB 28.33/94.67.", font="FigureSans-Italic", size=5.6, colour=MUTED)

        panel(c, 304, 12, 190, 316, "B", "Variation across the same five folds")
        x0, x1, y_top = 376, 478, 273
        xmin, xmax = 74, 82
        for tick in [74, 76, 78, 80, 82]:
            xx = x0 + (tick - xmin) / (xmax - xmin) * (x1 - x0)
            c.setStrokeColor(_colour(GRID))
            c.line(xx, 69, xx, 283)
            draw_text(c, xx, 56, tick, font="FigureSans", size=6, colour=MUTED, align="center")
        for idx, (name, values) in enumerate(FOLD_POINTS.items()):
            yy = y_top - idx * 43
            draw_text(c, 316, yy - 2, name, font="FigureSans", size=6.2)
            xs = [x0 + (value - xmin) / (xmax - xmin) * (x1 - x0) for value in values]
            c.setStrokeColor(_colour(MUTED))
            c.setLineWidth(1.4)
            c.line(min(xs), yy, max(xs), yy)
            for xx in xs:
                c.setFillColor(_colour(BLUE))
                c.circle(xx, yy, 2.5, fill=1, stroke=0)
            mean_x = x0 + (np.mean(values) - xmin) / (xmax - xmin) * (x1 - x0)
            c.setFillColor(_colour(GREEN))
            c.circle(mean_x, yy, 4.1, fill=1, stroke=0)
        draw_text(c, 397, 30, "accuracy (%)", font="FigureSans-Bold", size=6.8, colour=MUTED, align="center")


def create_search_ledger(path: Path) -> None:
    early = [
        ("Regional partial pool", -0.1557),
        ("Geographic centroids", -0.0590),
        ("Population median", -0.0340),
        ("MLP third voter", -0.0231),
        ("ANN 2% contribution", -0.0080),
        ("Spatial outcome voter", -0.0020),
        ("Source + class", 0.0110),
        ("Management only", 0.0130),
        ("Funder frequency", 0.0320),
        ("Target encoding 20%", 0.0480),
        ("Identity CatBoost 20%", 0.1160),
    ]
    late = [
        ("Frequency identity vote", -0.1557),
        ("Label filtering", -0.0568),
        ("Name-text vote", -0.0358),
        ("Calendar categories", -0.0295),
        ("Pseudo-labelling", -0.0105),
        ("Source-class cross", 0.0063),
        ("Physical back-offs", 0.0063),
    ]
    with FigureSurface(path, 7.0, 5.0) as c:
        panel(c, 10, 12, 238, 330, "A", "Against accepted 81.6246% vote")
        panel(c, 256, 12, 238, 330, "B", "Against promoted 81.7403% identity vote")

        def ledger(items: Sequence[tuple[str, float]], x: float, y: float, width: float, height: float) -> None:
            label_w = 105
            plot_x0 = x + label_w
            value_x = x + width - 7
            plot_x1 = value_x - 30
            xmin, xmax = -0.18, 0.13
            zero_x = plot_x0 + (0 - xmin) / (xmax - xmin) * (plot_x1 - plot_x0)
            gate_x = plot_x0 + (0.10 - xmin) / (xmax - xmin) * (plot_x1 - plot_x0)
            c.setStrokeColor(_colour(MUTED))
            c.setLineWidth(0.8)
            c.line(zero_x, y + 30, zero_x, y + height - 35)
            c.setStrokeColor(_colour(GREEN))
            c.setDash(2, 2)
            c.line(gate_x, y + 30, gate_x, y + height - 35)
            c.setDash()
            draw_text(c, gate_x, y + height - 27, "+0.10 gate", font="FigureSans-Bold", size=5.6, colour=GREEN, align="center")
            row_h = (height - 66) / len(items)
            for idx, (name, delta) in enumerate(items):
                yy = y + height - 49 - idx * row_h
                draw_text(c, x + 7, yy - 2, name, font="FigureSans", size=5.8)
                c.setStrokeColor(_colour(GRID))
                c.setLineWidth(0.65)
                xx = plot_x0 + (delta - xmin) / (xmax - xmin) * (plot_x1 - plot_x0)
                c.line(zero_x, yy, xx, yy)
                colour = GREEN if delta >= 0.10 else (GOLD if delta > 0 else MUTED)
                c.setFillColor(_colour(colour))
                c.circle(xx, yy, 3.1, fill=1, stroke=0)
                draw_text(c, value_x, yy - 2, f"{delta:+.3f}", font="FigureSans-Bold", size=5.5, colour=colour, align="right")
            draw_text(c, plot_x0, y + 16, "accuracy change (percentage points)", font="FigureSans-Bold", size=5.7, colour=MUTED)

        ledger(early, 15, 20, 228, 310)
        ledger(late, 261, 20, 228, 310)


def create_ensemble_architecture(path: Path) -> None:
    components = [
        ("Deep top-50 identity XGBoost", 33, 600, "23.3 s", BLUE),
        ("Spatial-height XGBoost", 11, 1082, "9.2 s", TEAL),
        ("Accepted Random Forest", 18, None, "31.7 s", NAVY),
        ("Categorical-frequency Random Forest", 9, None, "4.9 s", LAVENDER),
        ("Spatial-height Random Forest", 9, None, "31.2 s", GOLD),
        ("Complete-identity CatBoost", 20, 2173, "22.0 s", BURGUNDY),
    ]
    with FigureSurface(path, 7.0, 4.8) as c:
        w, h = 7 * INCH, 4.8 * INCH
        draw_text(c, 14, h - 19, "Representation-level probability synthesis", font="FigureSans-Bold", size=11, colour=NAVY)
        left_x, box_w, box_h, gap, top_y = 16, 192, 34, 8, 287
        fusion_x, fusion_y, fusion_w, fusion_h = 294, 112, 92, 172
        c.setFillColor(_colour(PANEL))
        c.setStrokeColor(_colour(GRID))
        c.roundRect(fusion_x, fusion_y, fusion_w, fusion_h, 8, fill=1, stroke=1)
        draw_text(c, fusion_x + fusion_w / 2, fusion_y + 144, "WEIGHTED", font="FigureSans-Bold", size=7.5, colour=NAVY, align="center")
        draw_text(c, fusion_x + fusion_w / 2, fusion_y + 130, "PROBABILITY", font="FigureSans-Bold", size=7.5, colour=NAVY, align="center")
        draw_text(c, fusion_x + fusion_w / 2, fusion_y + 116, "FUSION", font="FigureSans-Bold", size=7.5, colour=NAVY, align="center")
        draw_text(c, fusion_x + fusion_w / 2, fusion_y + 83, "sum w = 1.00", font="FigureSans", size=6.8, colour=MUTED, align="center")
        draw_text(c, fusion_x + fusion_w / 2, fusion_y + 61, "arg max", font="FigureSerif-Italic", size=11, colour=INK, align="center")
        draw_text(c, fusion_x + fusion_w / 2, fusion_y + 43, "over 3 classes", font="FigureSans", size=6.5, colour=MUTED, align="center")

        for idx, (name, weight, trees, timing, colour) in enumerate(components):
            y = top_y - idx * (box_h + gap)
            c.setFillColor(_colour(PAPER))
            c.setStrokeColor(_colour(colour))
            c.roundRect(left_x, y, box_w, box_h, 4, fill=1, stroke=1)
            c.setFillColor(_colour(colour))
            c.rect(left_x, y, 7, box_h, fill=1, stroke=0)
            draw_text(c, left_x + 14, y + 20, name, font="FigureSans-Bold", size=6.9, colour=INK)
            detail = f"{trees:,} trees" if trees is not None else "bagged forest"
            draw_text(c, left_x + 14, y + 8, f"{detail} | full-data fit + predict {timing}", font="FigureSans", size=5.7, colour=MUTED)
            draw_text(c, left_x + box_w - 8, y + 13, f"{weight}%", font="FigureSans-Bold", size=9.5, colour=colour, align="right")
            start_y = y + box_h / 2
            end_y = fusion_y + fusion_h * (0.13 + idx * 0.145)
            c.setStrokeColor(_colour(colour))
            c.setLineWidth(0.7 + weight / 8)
            c.line(left_x + box_w + 3, start_y, fusion_x - 5, end_y)

        output_x = 410
        outputs = [("functional", 60.721, TEAL), ("repair", 3.643, GOLD), ("non-functional", 35.636, BURGUNDY)]
        for idx, (name, share, colour) in enumerate(outputs):
            y = 228 - idx * 62
            c.setFillColor(_colour(colour))
            c.roundRect(output_x, y, 76, 42, 5, fill=1, stroke=0)
            draw_text(c, output_x + 38, y + 24, f"{share:.3f}%", font="FigureSans-Bold", size=9.3, colour=PAPER, align="center")
            draw_text(c, output_x + 38, y + 10, name.upper(), font="FigureSans-Bold", size=5.6, colour=PAPER, align="center")
            arrow(c, fusion_x + fusion_w + 4, fusion_y + fusion_h / 2, output_x - 5, y + 21, colour, 0.9)
        draw_text(c, output_x + 38, 291, "OUTPUT CLASS SHARES", font="FigureSans-Bold", size=6.2, colour=MUTED, align="center")
        c.setStrokeColor(_colour(GRID))
        c.line(16, 44, w - 16, 44)
        draw_text(c, w / 2, 24, "P(class | x) is the weighted mean of component probabilities; prediction is the highest-probability class.", font="FigureSerif-Italic", size=8.3, colour=INK, align="center")


def create_evidence_convergence(path: Path) -> None:
    candidates = ["Accepted vote", "Identity vote", "Archive", "Deep seed 22", "Deep seed 24"]
    dev = [81.6246, 81.7403, 81.7929, 81.8771, 81.8981]
    local = [80.82, 81.0859, 81.1869, 81.3552, 81.3973]
    public = [0.8241, 0.8259, 0.8288, np.nan, 0.8298]
    with FigureSurface(path, 7.0, 5.0) as c:
        panels = [(10, "A", "Development accuracy", dev, 81.58, 81.94, "%"), (174, "B", "Used local-test accuracy", local, 80.75, 81.45, "%"), (338, "C", "Public classification rate", public, 0.8235, 0.8302, "")]
        colours = [MUTED, BLUE, TEAL, LAVENDER, BURGUNDY]
        for x, label, title, vals, vmin, vmax, suffix in panels:
            panel(c, x, 154, 156, 188, label, title)
            y_top = 300
            for idx, (name, value, colour) in enumerate(zip(candidates, vals, colours)):
                yy = y_top - idx * 28
                draw_text(c, x + 9, yy - 2, name, font="FigureSans", size=5.8)
                if np.isnan(value):
                    draw_text(c, x + 9, yy - 9, "not submitted", font="FigureSans-Italic", size=5.2, colour=MUTED)
                    continue
                xx = x + 79 + (value - vmin) / (vmax - vmin) * 60
                c.setStrokeColor(_colour(GRID))
                c.line(x + 79, yy, x + 139, yy)
                c.setFillColor(_colour(colour))
                c.circle(xx, yy, 3.6, fill=1, stroke=0)
                format_value = f"{value:.4f}{suffix}"
                draw_text(c, x + 9, yy - 9, format_value, font="FigureSans-Bold", size=5.2, colour=colour)

        panel(c, 10, 12, 484, 130, "D", "Paired local uncertainty: effect estimates and intervals")
        comparisons = [
            ("Archive -> deep seed 20260822", 0.1684, -0.0421, 0.3788, 20, 0.135),
            ("Seed 20260822 -> seed 20260824", 0.0421, -0.0673, 0.1515, 5, 0.542),
        ]
        x0, x1, xmin, xmax = 246, 450, -0.10, 0.40
        zero_x = x0 + (0 - xmin) / (xmax - xmin) * (x1 - x0)
        c.setStrokeColor(_colour(MUTED))
        c.line(zero_x, 36, zero_x, 102)
        for idx, (name, estimate, lower, upper, rows, pvalue) in enumerate(comparisons):
            yy = 91 - idx * 42
            draw_text(c, 25, yy + 3, name, font="FigureSans-Bold", size=6.8)
            draw_text(c, 25, yy - 10, f"{rows:+d} net rows | exact McNemar p={pvalue:.3f}", font="FigureSans", size=6.1, colour=MUTED)
            lx = x0 + (lower - xmin) / (xmax - xmin) * (x1 - x0)
            ux = x0 + (upper - xmin) / (xmax - xmin) * (x1 - x0)
            ex = x0 + (estimate - xmin) / (xmax - xmin) * (x1 - x0)
            c.setStrokeColor(_colour(BLUE))
            c.setLineWidth(2)
            c.line(lx, yy, ux, yy)
            c.setFillColor(_colour(BURGUNDY))
            c.circle(ex, yy, 4, fill=1, stroke=0)
            draw_text(c, 486, yy - 2, f"{estimate:+.4f} pp", font="FigureSans-Bold", size=6.4, colour=BURGUNDY, align="right")
        draw_text(c, zero_x, 23, "0", font="FigureSans", size=5.8, colour=MUTED, align="center")


def create_leaderboard_evidence(path: Path, screenshot: Path) -> None:
    image = Image.open(screenshot).convert("RGB")
    bbox = Image.eval(image, lambda p: 255 - p).getbbox()
    if bbox:
        left, top, right, bottom = bbox
        pad = 18
        image = image.crop((max(0, left - pad), max(0, top - pad), min(image.width, right + pad), min(image.height, bottom + pad)))
    target_w = 2800
    target_h = int(image.height * target_w / image.width)
    image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    border = 18
    framed = Image.new("RGB", (target_w + border * 2, target_h + border * 2 + 110), "white")
    framed.paste(image, (border, border))
    draw = ImageDraw.Draw(framed)
    try:
        bold = ImageFont.truetype(str(FONT_DIR / "cambriab.ttf"), 40)
        regular = ImageFont.truetype(str(FONT_DIR / "cambria.ttc"), 30)
    except OSError:
        bold = ImageFont.load_default()
        regular = ImageFont.load_default()
    draw.rectangle((border, target_h + border, target_w + border, target_h + border + 110), fill="#F5F7F8")
    draw.text((40, target_h + 42), "PUBLIC EVIDENCE SNAPSHOT", font=bold, fill=f"#{NAVY}")
    draw.text((720, target_h + 48), "0.8298 | observed rank 2 | 23 August 2026", font=regular, fill=f"#{MUTED}")
    framed.save(path, dpi=(360, 360), quality=96)


def rasterise_pdf(pdf_path: Path, png_path: Path, dpi: int = 360) -> None:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise FileNotFoundError("pdftoppm is required to rasterise vector figure plates")
    prefix = png_path.with_suffix("")
    result = subprocess.run(
        [pdftoppm, "-png", "-singlefile", "-r", str(dpi), str(pdf_path), str(prefix)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not png_path.exists():
        raise RuntimeError(f"Figure rasterisation failed for {pdf_path}:\n{result.stdout}\n{result.stderr}")


def generate_all(data_dir: Path, asset_dir: Path, leaderboard_screenshot: Path) -> dict[str, Path]:
    values = pd.read_csv(data_dir / "TrainingSetValues.csv")
    labels = pd.read_csv(data_dir / "TrainingSetLabels.csv")
    asset_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "study_design": (create_study_design, ()),
        "data_anatomy": (create_data_anatomy, (values, labels)),
        "spatial": (create_spatial_heterogeneity, (values, labels)),
        "validation": (create_validation_design, ()),
        "model_family": (create_model_family_evidence, ()),
        "search_ledger": (create_search_ledger, ()),
        "ensemble": (create_ensemble_architecture, ()),
        "convergence": (create_evidence_convergence, ()),
    }
    outputs: dict[str, Path] = {}
    for index, (key, (creator, args)) in enumerate(specs.items(), start=1):
        pdf_path = asset_dir / f"figure-{index:02d}-{key.replace('_', '-')}.pdf"
        png_path = pdf_path.with_suffix(".png")
        creator(pdf_path, *args)
        rasterise_pdf(pdf_path, png_path)
        pdf_path.unlink(missing_ok=True)
        outputs[key] = png_path
    leaderboard = asset_dir / "figure-09-leaderboard-evidence.png"
    create_leaderboard_evidence(leaderboard, leaderboard_screenshot)
    outputs["leaderboard"] = leaderboard
    return outputs
