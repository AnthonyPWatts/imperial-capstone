"""Generate the publication-quality Pump It Up model-development report.

The generator deliberately uses only the repository's existing Python runtime,
the standard library, Matplotlib and Pillow.  It writes WordprocessingML
directly so the output remains reproducible without adding a production or
development dependency to the project.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
import textwrap
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


PUBLICATION_DIR = Path(__file__).resolve().parent
ASSET_DIR = PUBLICATION_DIR / "assets"
OUTPUT_PATH = PUBLICATION_DIR / "pump-it-up-model-development-report.docx"
DEFAULT_SCREENSHOT = Path(
    r"C:\Users\antho\AppData\Local\Temp\codex-clipboard-f843b5e0-268b-4e30-abed-cfb58acf84ba.png"
)

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}

NAVY = "0B2545"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
TEAL = "2F7F76"
GOLD = "C9962D"
INK = "202B33"
MUTED = "5E6B75"
LIGHT_BLUE = "E8EEF5"
LIGHT_GREY = "F2F4F7"
CALLOUT = "F4F6F9"
POSITIVE = "1F3A5F"
RISK = "9B1C1C"
WHITE = "FFFFFF"


def xml_text(value: object) -> str:
    return escape(str(value), quote=False)


def set_chart_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#9AA8B2",
            "axes.linewidth": 0.8,
            "xtick.color": "#42515C",
            "ytick.color": "#42515C",
            "text.color": "#202B33",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_lifecycle_chart(path: Path) -> None:
    steps = [
        "1  Define goal\nand scope",
        "2  Gather\nthe data",
        "3  Explore\nthe data",
        "4  Clean and\npreprocess",
        "5  Select and\nengineer features",
        "6  Define the\nML task",
        "7  Partition\nthe data",
        "8  Select and\ntrain candidates",
        "9  Evaluate and\ninterpret",
        "10  Deploy\nand iterate",
    ]
    positions = [
        (0.09, 0.72),
        (0.29, 0.72),
        (0.49, 0.72),
        (0.69, 0.72),
        (0.89, 0.72),
        (0.89, 0.28),
        (0.69, 0.28),
        (0.49, 0.28),
        (0.29, 0.28),
        (0.09, 0.28),
    ]
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("The course lifecycle was a loop, not a one-way pipeline", color=f"#{NAVY}", pad=18)

    for index, ((x, y), label) in enumerate(zip(positions, steps), start=1):
        colour = f"#{BLUE}" if index not in (7, 9, 10) else f"#{TEAL}"
        box = FancyBboxPatch(
            (x - 0.078, y - 0.075),
            0.156,
            0.15,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.4,
            edgecolor=colour,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9.3, weight="bold", color=f"#{INK}")

    for start, end in zip(positions[:-1], positions[1:]):
        sx, sy = start
        ex, ey = end
        if abs(sy - ey) < 0.01:
            dx = 0.09 if ex > sx else -0.09
            start_point = (sx + dx, sy)
            end_point = (ex - dx, ey)
        else:
            start_point = (sx, sy - 0.09)
            end_point = (ex, ey + 0.09)
        ax.add_patch(
            FancyArrowPatch(
                start_point,
                end_point,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.5,
                color="#78909C",
            )
        )

    feedback_specs = [
        ((0.09, 0.19), (0.09, 0.81), -0.35, "reset the goal"),
        ((0.12, 0.25), (0.49, 0.63), -0.22, "new evidence"),
        ((0.13, 0.29), (0.87, 0.66), 0.18, "feature and method loops"),
    ]
    for start, end, rad, label in feedback_specs:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            linestyle="--",
            connectionstyle=f"arc3,rad={rad}",
            color=f"#{GOLD}",
        )
        ax.add_patch(arrow)
        midpoint_x = (start[0] + end[0]) / 2
        midpoint_y = (start[1] + end[1]) / 2 + (0.13 if rad > 0 else -0.02)
        ax.text(midpoint_x, midpoint_y, label, fontsize=8.5, color=f"#{GOLD}", ha="center")

    ax.text(
        0.5,
        0.055,
        "Public leaderboard results closed a loop only after the model recipe and submission file had been frozen.",
        ha="center",
        fontsize=9,
        color=f"#{MUTED}",
    )
    save_figure(fig, path)


def create_public_timeline(path: Path) -> None:
    labels = [
        "Early\nensemble",
        "RF + HGB",
        "XGB + RF",
        "Source +\nclass",
        "Archive\nsynthesis",
        "Deep archive\nseed 24",
    ]
    scores = [0.8170, 0.8223, 0.8241, 0.8246, 0.8288, 0.8298]
    dates = ["15 Aug", "21 Aug", "21 Aug", "22 Aug", "23 Aug", "23 Aug"]
    x = list(range(len(scores)))
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    ax.plot(x, scores, color=f"#{BLUE}", linewidth=2.8, marker="o", markersize=8)
    ax.fill_between(x, scores, 0.814, color=f"#{BLUE}", alpha=0.07)
    ax.axhline(0.826, color=f"#{GOLD}", linestyle="--", linewidth=1.7, label="Project target 0.826")
    for index, (score, date) in enumerate(zip(scores, dates)):
        ax.annotate(
            f"{score:.4f}\n{date}",
            (index, score),
            xytext=(0, 12 if index % 2 == 0 else -36),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            weight="bold" if index == len(scores) - 1 else "normal",
            color=f"#{NAVY}",
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.814, 0.832)
    ax.set_ylabel("Public classification rate")
    ax.set_title("Public evidence improved in discrete, pre-recorded steps", color=f"#{NAVY}", pad=12)
    ax.grid(axis="y", color="#DDE4E8", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    ax.text(
        0.01,
        0.03,
        "Reference outside plotted scale: constant functional baseline = 0.5461",
        transform=ax.transAxes,
        fontsize=8.5,
        color=f"#{MUTED}",
    )
    save_figure(fig, path)


def create_ensemble_chart(path: Path) -> None:
    names = [
        "Deep top-50 identity XGBoost",
        "Complete-identity CatBoost",
        "Accepted Random Forest",
        "Spatial-height XGBoost",
        "Categorical-frequency Random Forest",
        "Spatial-height Random Forest",
    ]
    weights = [33, 20, 18, 11, 9, 9]
    colours = [f"#{BLUE}", f"#{TEAL}", f"#{DARK_BLUE}", "#6B8EAD", "#86A79F", "#A7BBCD"]
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    y = list(range(len(names)))
    bars = ax.barh(y, weights, color=colours, height=0.66)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 36)
    ax.set_xlabel("Share of final probability vote (%)")
    ax.set_title("The final prediction was a six-model probability synthesis", color=f"#{NAVY}", pad=12)
    ax.grid(axis="x", color="#E3E8EB", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, weights):
        ax.text(value + 0.7, bar.get_y() + bar.get_height() / 2, f"{value}%", va="center", weight="bold")
    save_figure(fig, path)


def create_evidence_chart(path: Path) -> None:
    candidates = ["Accepted\n55:45 vote", "Source +\nclass", "Complete\nidentity", "Archive\nsynthesis", "Deep seed\n20260822", "Final seed\n20260824"]
    development = [81.6246, 81.6350, 81.7403, 81.7929, 81.8771, 81.8981]
    local = [80.8200, 80.8333, 81.0859, 81.1869, 81.3552, 81.3973]
    x = list(range(len(candidates)))
    width = 0.37
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    left = [v - width / 2 for v in x]
    right = [v + width / 2 for v in x]
    b1 = ax.bar(left, development, width, color=f"#{BLUE}", label="Five-fold development")
    b2 = ax.bar(right, local, width, color=f"#{TEAL}", label="Used local test")
    ax.set_ylim(80.4, 82.1)
    ax.set_ylabel("Accuracy (%) - truncated axis")
    ax.set_xticks(x, candidates)
    ax.set_title("Development and local evidence moved in the same direction", color=f"#{NAVY}", pad=12)
    ax.grid(axis="y", color="#E3E8EB", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.035,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7.7,
                rotation=90,
            )
    ax.text(
        0.99,
        0.02,
        "The local test was later reused for bounded confirmation, so it is supporting evidence rather than a fresh final holdout.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.2,
        color=f"#{MUTED}",
    )
    save_figure(fig, path)


def create_search_decision_chart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Model search narrowed through evidence gates", color=f"#{NAVY}", pad=16)

    boxes = [
        (0.04, 0.42, 0.18, 0.20, "AUDIT + BASELINES\n36 predictors\nfixed splits\naccuracy + recall", NAVY, "white"),
        (0.29, 0.69, 0.21, 0.18, "PROMOTED SIGNAL\nXGBoost + RF\ncomplete identities\nfrequency + height", BLUE, "white"),
        (0.29, 0.15, 0.21, 0.30, "DIAGNOSTIC / REJECTED\nANN and broad resampling\ntarget/spatial outcome rates\nfuzzy and ordinal targets\npseudo-label/transductive loops", MUTED, "white"),
        (0.58, 0.62, 0.18, 0.22, "ARCHIVE SYNTHESIS\nsix fixed voters\n81.7929% dev\n81.1869% local\n0.8288 public", TEAL, "white"),
        (0.58, 0.17, 0.18, 0.22, "HELD NEAR-MISS\ndepth-17 XGBoost\ntop-50 identities\n600 trees\npredeclared seeds", GOLD, "white"),
        (0.82, 0.39, 0.15, 0.26, "FINAL FREEZE\nseed 20260824\n81.8981% dev\n81.3973% local\n0.8298 public\nrank #2", POSITIVE, "white"),
    ]
    for x, y, width, height, text, colour, text_colour in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.015", facecolor=f"#{colour}", edgecolor="none"
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9.1, color=text_colour, linespacing=1.35)

    arrows = [
        ((0.22, 0.57), (0.29, 0.76), BLUE, "promotion gate"),
        ((0.22, 0.47), (0.29, 0.31), MUTED, "record learning"),
        ((0.50, 0.78), (0.58, 0.74), TEAL, "fixed synthesis"),
        ((0.50, 0.72), (0.58, 0.31), GOLD, "archive reconstruction"),
        ((0.76, 0.72), (0.82, 0.58), POSITIVE, "retain 67%"),
        ((0.76, 0.28), (0.82, 0.46), POSITIVE, "replace 33%"),
    ]
    for start, end, colour, label in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color=f"#{colour}"))
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.03, label, ha="center", fontsize=7.9, color=f"#{colour}")
    ax.text(
        0.5,
        0.055,
        "A branch could be useful without being promoted: negative results protected the final recipe from unbounded tuning.",
        ha="center",
        fontsize=9,
        color=f"#{MUTED}",
    )
    save_figure(fig, path)


def create_class_distribution_chart(path: Path) -> None:
    names = ["Functional", "Functional needs repair", "Non-functional"]
    counts = [32259, 4317, 22824]
    shares = [54.31, 7.27, 38.42]
    colours = [f"#{TEAL}", f"#{GOLD}", f"#{BLUE}"]
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    bars = ax.barh(names, counts, color=colours, height=0.58)
    ax.invert_yaxis()
    ax.set_xlim(0, 35000)
    ax.set_xlabel("Labelled training rows")
    ax.set_title("The repair class was the central imbalance challenge", color=f"#{NAVY}", pad=12)
    ax.grid(axis="x", color="#E3E8EB", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, count, share in zip(bars, counts, shares):
        ax.text(count + 650, bar.get_y() + bar.get_height() / 2, f"{count:,}  |  {share:.2f}%", va="center", fontsize=10, weight="bold")
    save_figure(fig, path)


def generate_visuals(screenshot: Path) -> dict[str, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    set_chart_style()
    paths = {
        "lifecycle": ASSET_DIR / "iterative-lifecycle.png",
        "timeline": ASSET_DIR / "public-performance-timeline.png",
        "ensemble": ASSET_DIR / "final-ensemble-weights.png",
        "evidence": ASSET_DIR / "development-local-evidence.png",
        "search": ASSET_DIR / "model-search-decision-map.png",
        "classes": ASSET_DIR / "training-class-distribution.png",
        "leaderboard": ASSET_DIR / "leaderboard-rank-2.png",
    }
    create_lifecycle_chart(paths["lifecycle"])
    create_public_timeline(paths["timeline"])
    create_ensemble_chart(paths["ensemble"])
    create_evidence_chart(paths["evidence"])
    create_search_decision_chart(paths["search"])
    create_class_distribution_chart(paths["classes"])
    if not screenshot.exists():
        raise FileNotFoundError(f"Leaderboard screenshot not found: {screenshot}")
    shutil.copyfile(screenshot, paths["leaderboard"])
    return paths


@dataclass
class ImageRef:
    rel_id: str
    media_name: str
    path: Path


class DocxBuilder:
    def __init__(self) -> None:
        self.blocks: list[str] = []
        self.images: list[ImageRef] = []
        self.next_image_id = 1

    @staticmethod
    def _run(
        text: str,
        *,
        bold: bool = False,
        italic: bool = False,
        colour: str | None = None,
        size: float | None = None,
        font: str = "Calibri",
        break_before: bool = False,
    ) -> str:
        properties = [
            f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:eastAsia="{font}"/>',
        ]
        if bold:
            properties.append("<w:b/><w:bCs/>")
        if italic:
            properties.append("<w:i/><w:iCs/>")
        if colour:
            properties.append(f'<w:color w:val="{colour}"/>')
        if size is not None:
            half_points = int(round(size * 2))
            properties.append(f'<w:sz w:val="{half_points}"/><w:szCs w:val="{half_points}"/>')
        break_xml = "<w:br/>" if break_before else ""
        preserve = ' xml:space="preserve"' if text.startswith(" ") or text.endswith(" ") else ""
        return f"<w:r><w:rPr>{''.join(properties)}</w:rPr>{break_xml}<w:t{preserve}>{xml_text(text)}</w:t></w:r>"

    def paragraph(
        self,
        text: str = "",
        *,
        style: str = "Normal",
        align: str | None = None,
        keep_next: bool = False,
        keep_lines: bool = False,
        page_break_before: bool = False,
        num_id: int | None = None,
        bold: bool = False,
        italic: bool = False,
        colour: str | None = None,
        size: float | None = None,
        font: str = "Calibri",
        before: int | None = None,
        after: int | None = None,
        shading: str | None = None,
        border: str | None = None,
    ) -> None:
        ppr = [f'<w:pStyle w:val="{style}"/>']
        if align:
            ppr.append(f'<w:jc w:val="{align}"/>')
        if keep_next:
            ppr.append("<w:keepNext/>")
        if keep_lines:
            ppr.append("<w:keepLines/>")
        if page_break_before:
            ppr.append("<w:pageBreakBefore/>")
        if num_id is not None:
            ppr.append(f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{num_id}"/></w:numPr>')
        if before is not None or after is not None:
            values = []
            if before is not None:
                values.append(f'w:before="{before}"')
            if after is not None:
                values.append(f'w:after="{after}"')
            ppr.append(f"<w:spacing {' '.join(values)}/>")
        if shading:
            ppr.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{shading}"/>')
        if border:
            ppr.append(
                f'<w:pBdr><w:left w:val="single" w:sz="18" w:space="10" w:color="{border}"/></w:pBdr>'
            )
        content = self._run(text, bold=bold, italic=italic, colour=colour, size=size, font=font)
        self.blocks.append(f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{content}</w:p>")

    def rich_paragraph(
        self,
        runs: list[dict[str, object]],
        *,
        style: str = "Normal",
        align: str | None = None,
        keep_next: bool = False,
        keep_lines: bool = False,
        shading: str | None = None,
        border: str | None = None,
        before: int | None = None,
        after: int | None = None,
    ) -> None:
        ppr = [f'<w:pStyle w:val="{style}"/>']
        if align:
            ppr.append(f'<w:jc w:val="{align}"/>')
        if keep_next:
            ppr.append("<w:keepNext/>")
        if keep_lines:
            ppr.append("<w:keepLines/>")
        if shading:
            ppr.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{shading}"/>')
        if border:
            ppr.append(f'<w:pBdr><w:left w:val="single" w:sz="18" w:space="10" w:color="{border}"/></w:pBdr>')
        if before is not None or after is not None:
            values = []
            if before is not None:
                values.append(f'w:before="{before}"')
            if after is not None:
                values.append(f'w:after="{after}"')
            ppr.append(f"<w:spacing {' '.join(values)}/>")
        run_xml = "".join(self._run(**run) for run in runs)
        self.blocks.append(f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{run_xml}</w:p>")

    def heading(self, level: int, text: str, *, page_break_before: bool = False) -> None:
        self.paragraph(text, style=f"Heading{level}", keep_next=True, page_break_before=page_break_before)

    def bullet(self, text: str) -> None:
        self.paragraph(text, style="ListParagraph", num_id=1)

    def numbered(self, text: str) -> None:
        self.paragraph(text, style="ListParagraph", num_id=2)

    def page_break(self) -> None:
        self.blocks.append("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")

    def spacer(self, points: float) -> None:
        self.blocks.append(
            f'<w:p><w:pPr><w:spacing w:before="0" w:after="{int(points * 20)}"/></w:pPr><w:r><w:t></w:t></w:r></w:p>'
        )

    def table(
        self,
        headers: list[str],
        rows: list[list[object]],
        widths: list[int],
        *,
        aligns: list[str] | None = None,
        caption: str | None = None,
        source: str | None = None,
    ) -> None:
        if sum(widths) != 9360:
            raise ValueError(f"Table widths must total 9360 DXA, got {sum(widths)}")
        if caption:
            self.paragraph(caption, style="TableCaption", keep_next=True)
        aligns = aligns or ["left"] * len(headers)
        grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in widths)
        table_rows = [self._table_row(headers, widths, aligns, header=True)]
        table_rows.extend(self._table_row(row, widths, aligns, header=False) for row in rows)
        tbl_pr = (
            '<w:tblW w:w="9360" w:type="dxa"/>'
            '<w:tblInd w:w="120" w:type="dxa"/>'
            '<w:tblLayout w:type="fixed"/>'
            '<w:tblBorders>'
            '<w:top w:val="single" w:sz="4" w:space="0" w:color="B9C4CC"/>'
            '<w:left w:val="single" w:sz="4" w:space="0" w:color="B9C4CC"/>'
            '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="B9C4CC"/>'
            '<w:right w:val="single" w:sz="4" w:space="0" w:color="B9C4CC"/>'
            '<w:insideH w:val="single" w:sz="3" w:space="0" w:color="D7DEE3"/>'
            '<w:insideV w:val="single" w:sz="3" w:space="0" w:color="D7DEE3"/>'
            '</w:tblBorders>'
            '<w:tblCellMar>'
            '<w:top w:w="80" w:type="dxa"/><w:start w:w="120" w:type="dxa"/>'
            '<w:bottom w:w="80" w:type="dxa"/><w:end w:w="120" w:type="dxa"/>'
            '</w:tblCellMar>'
        )
        self.blocks.append(f"<w:tbl><w:tblPr>{tbl_pr}</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{''.join(table_rows)}</w:tbl>")
        if source:
            self.paragraph(source, style="TableSource")

    def _table_row(self, values: list[object], widths: list[int], aligns: list[str], *, header: bool) -> str:
        cells = []
        for value, width, align in zip(values, widths, aligns):
            fill = f'<w:shd w:val="clear" w:color="auto" w:fill="{LIGHT_GREY}"/>' if header else ""
            tc_pr = f'<w:tcW w:w="{width}" w:type="dxa"/><w:vAlign w:val="center"/>{fill}'
            run = self._run(str(value), bold=header, colour=NAVY if header else INK, size=9.2 if header else 9.4)
            p = (
                f'<w:p><w:pPr><w:pStyle w:val="TableText"/><w:jc w:val="{align}"/>'
                '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
                f'{run}</w:p>'
            )
            cells.append(f"<w:tc><w:tcPr>{tc_pr}</w:tcPr>{p}</w:tc>")
        tr_pr = '<w:tblHeader/><w:cantSplit/>' if header else '<w:cantSplit/>'
        return f"<w:tr><w:trPr>{tr_pr}</w:trPr>{''.join(cells)}</w:tr>"

    def figure(self, path: Path, width_inches: float, caption: str, alt_text: str) -> None:
        with Image.open(path) as image:
            pixel_width, pixel_height = image.size
        height_inches = width_inches * pixel_height / pixel_width
        cx = int(width_inches * 914400)
        cy = int(height_inches * 914400)
        rel_id = f"rId{10 + len(self.images)}"
        media_name = f"image{self.next_image_id}{path.suffix.lower()}"
        doc_pr_id = self.next_image_id
        self.next_image_id += 1
        self.images.append(ImageRef(rel_id, media_name, path))
        drawing = f"""
        <w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">
          <wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>
          <wp:docPr id="{doc_pr_id}" name="Figure {doc_pr_id}" descr="{escape(alt_text, quote=True)}"/>
          <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
          <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="{media_name}"/><pic:cNvPicPr/></pic:nvPicPr>
            <pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
            <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
            <a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>
          </a:graphicData></a:graphic>
        </wp:inline></w:drawing></w:r>
        """
        self.blocks.append(
            '<w:p><w:pPr><w:pStyle w:val="Figure"/><w:jc w:val="center"/><w:keepNext/>'
            '<w:spacing w:before="120" w:after="60"/></w:pPr>' + drawing + '</w:p>'
        )
        self.paragraph(caption, style="Caption", align="center", keep_lines=True)

    def document_xml(self) -> str:
        body = "".join(self.blocks)
        section = (
            '<w:sectPr>'
            '<w:headerReference w:type="default" r:id="rId3"/>'
            '<w:footerReference w:type="default" r:id="rId4"/>'
            '<w:pgSz w:w="12240" w:h="15840"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/>'
            '<w:cols w:space="708"/><w:docGrid w:linePitch="360"/>'
            '</w:sectPr>'
        )
        ns_attrs = " ".join(f'xmlns:{prefix}="{uri}"' for prefix, uri in NS.items())
        return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {ns_attrs}><w:body>{body}{section}</w:body></w:document>'


def styles_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS['w']}">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Calibri"/><w:sz w:val="22"/><w:szCs w:val="22"/><w:lang w:val="en-GB"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:before="0" w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/><w:qFormat/><w:pPr><w:widowControl/><w:spacing w:before="0" w:after="120" w:line="264" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:color w:val="{INK}"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="CoverKicker"><w:name w:val="Cover Kicker"/><w:basedOn w:val="Normal"/><w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="360"/></w:pPr><w:rPr><w:b/><w:caps/><w:color w:val="{GOLD}"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="CoverTitle"><w:name w:val="Cover Title"/><w:basedOn w:val="Normal"/><w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="160"/><w:keepNext/></w:pPr><w:rPr><w:b/><w:color w:val="{NAVY}"/><w:sz w:val="60"/><w:szCs w:val="60"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="CoverSubtitle"><w:name w:val="Cover Subtitle"/><w:basedOn w:val="Normal"/><w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="40"/></w:pPr><w:rPr><w:color w:val="{DARK_BLUE}"/><w:sz w:val="30"/><w:szCs w:val="30"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/>
    <w:pPr><w:spacing w:before="0" w:after="120"/><w:keepNext/></w:pPr><w:rPr><w:b/><w:color w:val="{NAVY}"/><w:sz w:val="48"/><w:szCs w:val="48"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:qFormat/>
    <w:pPr><w:spacing w:before="0" w:after="280"/></w:pPr><w:rPr><w:color w:val="{MUTED}"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:uiPriority w:val="9"/>
    <w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="320" w:after="160"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="{BLUE}"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:uiPriority w:val="9"/>
    <w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:color w:val="{BLUE}"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:uiPriority w:val="9"/>
    <w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="160" w:after="80"/><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:b/><w:color w:val="{DARK_BLUE}"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="0" w:after="160" w:line="280" w:lineRule="auto"/><w:contextualSpacing/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Figure"><w:name w:val="Figure"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="60"/><w:keepNext/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="160"/><w:keepLines/></w:pPr><w:rPr><w:i/><w:color w:val="{MUTED}"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="TableCaption"><w:name w:val="Table Caption"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="80" w:after="80"/><w:keepNext/></w:pPr><w:rPr><w:b/><w:color w:val="{DARK_BLUE}"/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="TableSource"><w:name w:val="Table Source"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="80" w:after="80"/></w:pPr><w:rPr><w:i/><w:color w:val="{MUTED}"/><w:sz w:val="17"/><w:szCs w:val="17"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr></w:style>
  <w:style w:type="character" w:styleId="CodeChar"><w:name w:val="Code Char"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:color w:val="{NAVY}"/><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr></w:style>
</w:styles>'''


def numbering_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="{NS['w']}">
  <w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0">
    <w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="&#x2022;"/><w:lvlJc w:val="left"/>
    <w:pPr><w:tabs><w:tab w:val="num" w:pos="720"/></w:tabs><w:ind w:left="720" w:hanging="360"/><w:spacing w:after="160" w:line="280" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/></w:rPr></w:lvl></w:abstractNum>
  <w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0">
    <w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/>
    <w:pPr><w:tabs><w:tab w:val="num" w:pos="720"/></w:tabs><w:ind w:left="720" w:hanging="360"/><w:spacing w:after="160" w:line="280" w:lineRule="auto"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/></w:rPr></w:lvl></w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
  <w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>
</w:numbering>'''


def header_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr xmlns:w="{NS['w']}" xmlns:r="{NS['r']}">
  <w:p><w:pPr><w:tabs><w:tab w:val="right" w:pos="9360"/></w:tabs><w:spacing w:before="0" w:after="0"/></w:pPr>
    <w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:b/><w:color w:val="{NAVY}"/><w:sz w:val="17"/></w:rPr><w:t>PUMP IT UP | MODEL DEVELOPMENT REPORT</w:t></w:r>
    <w:r><w:tab/></w:r><w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:color w:val="{MUTED}"/><w:sz w:val="17"/></w:rPr><w:t>23 AUGUST 2026</w:t></w:r>
  </w:p>
</w:hdr>'''


def footer_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="{NS['w']}" xmlns:r="{NS['r']}">
  <w:p><w:pPr><w:tabs><w:tab w:val="right" w:pos="9360"/></w:tabs><w:spacing w:before="0" w:after="0"/><w:pBdr><w:top w:val="single" w:sz="4" w:space="6" w:color="D7DEE3"/></w:pBdr></w:pPr>
    <w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:color w:val="{MUTED}"/><w:sz w:val="17"/></w:rPr><w:t>Public result observed 23 August 2026; live rank may change</w:t></w:r>
    <w:r><w:tab/></w:r><w:r><w:rPr><w:color w:val="{MUTED}"/><w:sz w:val="17"/></w:rPr><w:t>Page </w:t></w:r>
    <w:fldSimple w:instr="PAGE"><w:r><w:rPr><w:color w:val="{MUTED}"/><w:sz w:val="17"/></w:rPr><w:t>1</w:t></w:r></w:fldSimple>
  </w:p>
</w:ftr>'''


def build_document(visuals: dict[str, Path]) -> DocxBuilder:
    doc = DocxBuilder()

    # Editorial cover pattern: generous whitespace, centred title stack, quiet metadata.
    doc.spacer(86)
    doc.paragraph("MODEL DEVELOPMENT REPORT", style="CoverKicker")
    doc.paragraph("Pump It Up", style="CoverTitle")
    doc.paragraph("From audited data to a 0.8298 public score", style="CoverSubtitle")
    doc.paragraph("A course-aligned, evidence-led account of the Tanzania waterpoint classification project", style="CoverSubtitle", size=12, colour=MUTED)
    doc.spacer(44)
    doc.paragraph("Imperial College London ML/AI programme | Stage 1 practice capstone", align="center", bold=True, colour=NAVY, size=10.5)
    doc.paragraph("Prepared by Anthony P. Watts | 23 August 2026", align="center", colour=MUTED, size=10)
    doc.spacer(40)
    doc.rich_paragraph(
        [
            {"text": "0.8298", "bold": True, "size": 20, "colour": POSITIVE},
            {"text": "  PUBLIC SCORE     ", "bold": True, "size": 9, "colour": MUTED},
            {"text": "#2", "bold": True, "size": 20, "colour": POSITIVE},
            {"text": "  OBSERVED RANK     ", "bold": True, "size": 9, "colour": MUTED},
            {"text": "15", "bold": True, "size": 20, "colour": POSITIVE},
            {"text": "  SUBMISSIONS", "bold": True, "size": 9, "colour": MUTED},
        ],
        align="center",
        shading=LIGHT_GREY,
        border=GOLD,
        before=180,
        after=180,
    )
    doc.paragraph("Observed rank is a dated public-leaderboard snapshot, not a permanent position or a private-leaderboard result.", align="center", italic=True, colour=MUTED, size=9)
    doc.page_break()

    doc.heading(1, "Executive summary")
    doc.rich_paragraph(
        [
            {"text": "Outcome. ", "bold": True, "colour": POSITIVE},
            {"text": "The final, unchanged competition file scored 0.8298 publicly and was observed at rank 2 on 23 August 2026, 0.0001 behind the leader. It exceeded the 0.826 project target by 0.0038."},
        ],
        shading=CALLOUT,
        border=TEAL,
        before=120,
        after=160,
    )
    doc.paragraph(
        "The work began as a practice competition and matured into a disciplined modelling study. The task was to classify each Tanzanian waterpoint as functional, functional but needing repair, or non-functional. The supplied data contained 59,400 labelled rows and 14,850 competition rows. Accuracy was the competition measure; per-class recall and probability-quality diagnostics were retained so the small repair class was not hidden by the headline score."
    )
    doc.paragraph(
        "The selected model was not a single learner. It was a fixed six-component probability vote combining a deep top-50-identity XGBoost model, two spatial-height models, two Random Forest representations, and a complete-identity CatBoost model. Only the seed of the 33% deep component changed in the final bounded comparison; depth 17, 600 trees, feature policy, and all six weights stayed fixed."
    )
    doc.table(
        ["Measure", "Final evidence", "Interpretation"],
        [
            ["Development accuracy", "81.8981%", "Five fixed stratified folds"],
            ["Used local-test accuracy", "81.3973%", "11,880 rows; bounded confirmation"],
            ["Public score", "0.8298", "+0.0010 over archive synthesis"],
            ["Public rank", "#2 observed", "23 August 2026; live rank may change"],
            ["Distance to #1", "0.0001", "One ten-thousandth of public classification rate"],
            ["Submission integrity", "SHA-256 fe5de9ea...08fee2", "14,850 unique IDs in template order"],
        ],
        [2200, 2200, 4960],
        aligns=["left", "center", "left"],
        caption="Table 1. Final result at a glance",
        source="Repository evidence: reports/deep-follow-up-search.md and submissions/2026-08-23-deep-archive-seed-20260824/README.md.",
    )
    doc.heading(2, "What made the result credible")
    for item in [
        "Immutable source data and guarded structural removals were separated from learned preprocessing.",
        "A stratified 20% local test and five fixed development folds were established before model selection.",
        "Target-aware and spatial features were cross-fitted so a row could not encode or neighbour itself.",
        "Promotion gates bounded most feature, model, and blend searches; useful negative results were recorded rather than quietly discarded.",
        "The final CSV was validated and submitted unchanged; the public result was treated as outcome evidence, not as a new tuning signal.",
    ]:
        doc.bullet(item)
    doc.page_break()

    doc.heading(1, "Course-aligned report map")
    doc.paragraph(
        "The report follows the programme's exact ten-step machine-learning lifecycle. The numbering is stable, but the work itself was iterative: findings at evaluation and deployment repeatedly reopened earlier questions under bounded experiment plans."
    )
    lifecycle_names = [
        "Define the goal and scope",
        "Gather the data",
        "Explore the data",
        "Clean and preprocess the data",
        "Select and engineer features",
        "Define the machine-learning task",
        "Partition the data",
        "Select and train candidate methods",
        "Evaluate and interpret the results",
        "Deploy and iterate",
    ]
    for name in lifecycle_names:
        doc.numbered(name)
    doc.figure(
        visuals["lifecycle"],
        6.35,
        "Figure 1. The ten-step course lifecycle with the feedback loops used in the project.",
        "Flowchart of the ten course lifecycle steps in a snake-shaped sequence, with dashed feedback arrows from deploy and iterate back to goal setting, exploration, and feature engineering.",
    )
    doc.page_break()

    doc.heading(1, "1. Define the goal and scope")
    doc.paragraph(
        "The immediate goal was to predict survey-time waterpoint condition for the DrivenData Pump It Up competition. The practical motivation was access to potable water: a useful classifier can help prioritise investigation and maintenance, although the competition label alone cannot prescribe an intervention. Stage 1 was treated as optional practice and portfolio evidence rather than as the assessed black-box optimisation project that begins later in the course."
    )
    doc.heading(2, "Success criteria")
    doc.bullet("Primary competition objective: maximise multiclass accuracy on the supplied public scoring system.")
    doc.bullet("Project benchmark: reach at least 0.826 public classification rate; the final score exceeded it by 0.0038.")
    doc.bullet("Diagnostic objective: retain classwise recall, especially for the 7.27% repair class, plus macro F1, balanced accuracy, log loss, and multiclass Brier score.")
    doc.bullet("Process objective: keep every material result reproducible, reviewable, and connected to a predeclared or tightly bounded experiment.")
    doc.heading(2, "Scope boundaries")
    doc.paragraph(
        "The project used only the supplied feature and label tables for final training. External-data proposals were considered but not promoted. Competition deployment meant producing a validated CSV, not putting a field decision system into operational use. No claim is made that the model predicts future pump failure, diagnoses engineering causes, or estimates the causal effect of repairs."
    )
    doc.rich_paragraph(
        [
            {"text": "Interpretation boundary. ", "bold": True, "colour": RISK},
            {"text": "status_group is a survey-time nominal label. It should not be read as a timeless property, a failure forecast, or a rehabilitation-priority score."},
        ],
        shading="FFF7F5",
        border=RISK,
        before=120,
        after=160,
    )
    doc.paragraph("Repository evidence: reports/training-label-provenance-and-semantics.md and reports/model-search-conclusion.md.", style="TableSource")

    doc.heading(1, "2. Gather the data", page_break_before=True)
    doc.paragraph(
        "DrivenData supplied two feature tables with the same 40 raw columns and a separate label table keyed by id. The modelling set contained 59,400 unique labelled IDs; the competition set contained 14,850 IDs. The target table had one complete status_group value for every training ID and exactly the three documented classes."
    )
    doc.table(
        ["Asset", "Rows", "Raw columns", "Role"],
        [
            ["Training values", "59,400", "40", "Predictors and identifiers"],
            ["Training labels", "59,400", "2", "id plus status_group"],
            ["Competition values", "14,850", "40", "Unlabelled prediction target"],
        ],
        [2500, 1300, 1500, 4060],
        aligns=["left", "center", "center", "left"],
        caption="Table 2. Supplied competition data",
        source="Repository evidence: reports/data-preparation-next-steps.md and the maintained data-audit notebooks.",
    )
    doc.heading(2, "Provenance contract")
    doc.paragraph(
        "Canonical data was recreated from untouched competition CSVs on every run. The code did not maintain a second cleaned-data copy. ID and schema validation preceded every modelling handoff and every competition export. This kept the source record separate from decisions learned inside a training fold."
    )
    doc.bullet("Join labels by id and assert one-to-one coverage.")
    doc.bullet("Apply the same structural checks independently to training and competition features.")
    doc.bullet("Retain id as submission metadata and exclude it from the predictor matrix.")
    doc.bullet("Do not infer means, modes, category rates, or target relationships before partitioning.")

    doc.heading(1, "3. Explore the data", page_break_before=True)
    doc.paragraph(
        "Exploration covered schema, duplicates, class balance, numeric sentinel states, missingness, geography, categorical hierarchies, sparse identities, and probability ambiguity. The analysis found a conventional majority/minority problem rather than a large training-to-competition prior shift. The rare repair class was 7.47 times smaller than the functional class and was consistently the hardest state to separate."
    )
    doc.figure(
        visuals["classes"],
        6.1,
        "Figure 2. Training-label counts and shares.",
        "Horizontal bars showing 32,259 functional rows, 4,317 functional-needs-repair rows, and 22,824 non-functional rows, with the repair class much smaller than the others.",
    )
    doc.heading(2, "Structural findings that changed the modelling frame")
    doc.table(
        ["Finding", "Evidence", "Decision"],
        [
            ["quantity_group duplicates quantity", "Row-for-row in both feature files", "Keep quantity; drop quantity_group"],
            ["payment relabels payment_type", "One-to-one mapping in both files", "Keep payment_type; drop payment"],
            ["recorded_by is constant", "100% GeoData Consultants Ltd", "Drop recorded_by"],
            ["Physical fields form hierarchies", "Extraction, source, quality, waterpoint", "Retain granular levels until tested"],
            ["Coordinate validity is incomplete", "57,588 mapped; 1,812 excluded", "Use explicit validity/fallback rules"],
        ],
        [3000, 3000, 3360],
        caption="Table 3. High-impact audit findings",
        source="Repository evidence: reports/data-preparation-next-steps.md and project-status.json.",
    )
    doc.heading(2, "What the public baselines established")
    doc.paragraph(
        "Three constant-class submissions scored 0.5461, 0.0719, and 0.3820. Their sum of 1.0000 and their proximity to the training shares suggested no material aggregate class-prior shift. The functional baseline of 0.5461 became the public floor, not a model-selection target."
    )

    doc.heading(1, "4. Clean and preprocess the data", page_break_before=True)
    doc.paragraph(
        "The clean/preprocess boundary deliberately separated fixed structural facts from learned statistics. Three guarded removals reduced 40 raw columns to 37 columns including id, leaving 36 candidate predictors. The removal function stopped if source schema, duplicate equality, payment mapping, or the recorded_by constant differed from the audited contract."
    )
    doc.heading(2, "Fold-fitted transformations")
    doc.bullet("Median imputation and missingness indicators for numeric features.")
    doc.bullet("Explicit missing and rare states for categorical features, followed by one-hot encoding for conventional tree pipelines.")
    doc.bullet("days_since_recorded in place of the raw date, measured from the fixed 2015-02-02 competition-era reference.")
    doc.bullet("Fold-fitted category occurrence counts, identity indicators, target encodings, and spatial transforms whenever they were evaluated.")
    doc.bullet("Native categorical values for CatBoost, preserving its ordered categorical treatment rather than forcing every identity through one-hot encoding.")
    doc.heading(2, "Leakage controls")
    doc.paragraph(
        "No validation-row target could contribute to a target encoding or a spatial outcome feature used to predict that row. Training-side supervised features were produced by an inner cross-fit; outer validation used mappings or neighbour indexes fitted only on outer training. Unknown identities fell back to the outer-training class prior. This was more restrictive than ordinary preprocessing because it applied to every supervised representation, not only to the final estimator."
    )
    doc.rich_paragraph(
        [
            {"text": "Why it mattered. ", "bold": True, "colour": POSITIVE},
            {"text": "High-cardinality names and nearby labelled pumps can look extraordinarily predictive if a row is allowed to leak its own status into the feature. Cross-fitting converted that apparent shortcut into honest out-of-fold evidence."},
        ],
        shading=CALLOUT,
        border=TEAL,
        before=120,
        after=160,
    )
    doc.paragraph("Repository evidence: src/data_preparation.py, src/raw_feature_column_policy.json, reports/cross-fitted-target-encoding-screen.md and reports/cross-fitted-spatial-outcome-screen.md.", style="TableSource")

    doc.heading(1, "5. Select and engineer features", page_break_before=True)
    doc.paragraph(
        "Feature work progressed from broad audit findings to bounded family screens. The project kept a representation only when it improved fixed-fold evidence or supplied useful diversity in a later fixed synthesis. Negative results were retained because they defined where further tuning was unlikely to repay the validation risk."
    )
    doc.table(
        ["Feature family", "Result", "Final role"],
        [
            ["Physical and administrative hierarchies", "Granular levels generally outperformed compact removals", "Retained in accepted frame"],
            ["Complete sparse identities", "CatBoost contribution improved development and local evidence", "20% final weight"],
            ["All-categorical occurrence counts", "Complementary Random Forest representation", "9% final weight"],
            ["Spatial-height reconstruction", "Useful XGBoost and Random Forest diversity", "11% and 9% weights"],
            ["Top-50 exact identity indicators", "Strengthened deep XGBoost and archive reconstruction", "33% final weight"],
            ["Target-encoded and spatial outcome rates", "Informative alone; weaker in the complete vote", "Rejected after leakage-safe screen"],
        ],
        [2700, 3500, 3160],
        caption="Table 4. Feature-family decisions that shaped the final model",
        source="Repository evidence: the feature-family reports indexed by reports/README.md.",
    )
    doc.heading(2, "Identity features: compact signal without an unbounded vocabulary")
    doc.paragraph(
        "The official solution archive motivated 50 common-value indicators for each of funder, installer, wpt_name, subvillage, ward, and scheme_name. The lists were learned inside each model-fitting partition, with lexical tie-breaking and all-zero fallback for unseen values. This created 300 sparse indicators and avoided treating the first 50 values as a global fact learned from validation data."
    )
    doc.heading(2, "Spatial features: useful physical context, dangerous target context")
    doc.paragraph(
        "Coordinate-based height reconstruction supplied complementary model representations. By contrast, cross-fitted local outcome rates were 0.629 percentage points below the accepted ensemble as a complete feature policy; even a 15% spatial voter remained 0.002 points below baseline and lost 2.635 points of repair recall. Nearby outcomes were real signal, but largely duplicated geography already available to the model."
    )

    doc.heading(1, "6. Define the machine-learning task", page_break_before=True)
    doc.paragraph(
        "The final task remained flat, nominal, three-class probabilistic classification. The ordered wording of the labels made binary and ordinal alternatives worth testing, but it did not make the target inherently ordinal. The competition decision rule was the maximum predicted class probability and the primary selection metric was ordinary accuracy."
    )
    doc.heading(2, "Metrics and why they were paired")
    doc.table(
        ["Metric", "Use", "Caution"],
        [
            ["Accuracy", "Competition objective and promotion gate", "Can hide repair-class failure"],
            ["Per-class recall", "Functional, repair, and non-functional diagnosis", "Not the competition objective"],
            ["Balanced accuracy / macro F1", "Equal-class supporting view", "Can favour different error trade-offs"],
            ["Log loss / multiclass Brier", "Probability quality", "Sensitive to confidence, not only hard labels"],
            ["McNemar / paired bootstrap", "Uncertainty of row-paired changes", "Intervals remain conditional on reused evidence"],
        ],
        [2300, 3500, 3560],
        caption="Table 5. Evaluation measures",
        source="Repository evidence: reports/probability-reliability-and-edge-cases.md and the confirmation reports.",
    )
    doc.heading(2, "Alternatives tested, not assumed")
    doc.paragraph(
        "A binary operational/non-operational decomposition improved one conditional boundary but forced zero repair recall and reduced full development accuracy. Fuzzy memberships increased repair mass but every complete fuzzy vote lost accuracy. Cumulative ordinal XGBoost required coherence projection on roughly one quarter of rows and improved the archive by only five development rows at best. Each experiment clarified the label structure without replacing the nominal task."
    )

    doc.heading(1, "7. Partition the data", page_break_before=True)
    doc.paragraph(
        "A reproducible stratified 20% local test reserved 11,880 rows. The remaining 47,520 development rows were assigned to five fixed stratified folds. The same folds supported candidate comparisons, probability alignment, and fixed synthesis work. The 14,850 competition rows remained unlabelled and were used only for final full-data prediction and structural validation."
    )
    doc.table(
        ["Partition", "Rows", "Purpose", "Selection status"],
        [
            ["Development", "47,520", "Five-fold model and feature comparisons", "Repeatedly reused under bounded gates"],
            ["Local test", "11,880", "Confirmation after a candidate was fixed", "Later reused for narrow confirmation"],
            ["Competition", "14,850", "Unlabelled public/private scoring", "Outcome evidence only"],
        ],
        [1900, 1300, 3500, 2660],
        aligns=["left", "center", "left", "left"],
        caption="Table 6. Partition and evidence roles",
        source="Repository evidence: reports/data-preparation-next-steps.md and reports/deep-archive-confirmation.md.",
    )
    doc.heading(2, "The validation caveat")
    doc.paragraph(
        "Fixed folds prevent accidental movement of the goalposts, but they do not make repeated consultation free. By the final stage, many bounded hypotheses had been screened on the same development folds and the local test had been opened more than once for confirmation. The final report therefore treats very small gains as uncertain and the public 0.8298 as final outcome evidence, not permission for another round of seed or weight selection."
    )

    doc.heading(1, "8. Select and train candidate methods", page_break_before=True)
    doc.paragraph(
        "Method selection moved from inexpensive baselines to stronger learners and then to representation diversity. Decision trees, Extra Trees, histogram gradient boosting, Random Forest, XGBoost, LightGBM, CatBoost, multilayer perceptrons, binary decompositions, ordinal models, and several probability combinations were evaluated. Promotion required more than a single mean: fold wins, worst-fold movement, class recall, and later local confirmation constrained the search."
    )
    doc.figure(
        visuals["search"],
        6.4,
        "Figure 3. Decision map from broad screening to the frozen final model.",
        "Decision map showing audit and baselines splitting into promoted and rejected branches, archive synthesis and a held deep-model near-miss, then converging on the final seed-20260824 model.",
    )
    doc.heading(2, "A compressed method progression")
    doc.bullet("Extra Trees plus histogram boosting established a 0.8170 public proof of concept before formal partitioning.")
    doc.bullet("Random Forest plus histogram boosting reached 81.37% development and 80.82% local accuracy, then scored 0.8223.")
    doc.bullet("Depth-8 XGBoost plus Random Forest raised development accuracy to 81.625% and the public score to 0.8241.")
    doc.bullet("Complete-identity CatBoost, frequency, and spatial-height representations provided complementary probability errors.")
    doc.bullet("Archive synthesis combined six fixed components and scored 0.8288 publicly.")
    doc.bullet("A depth-17, 600-tree XGBoost reconstruction with fold-fitted top-50 identity indicators replaced only the 33% primary component.")

    doc.heading(2, "The final six-component vote")
    doc.figure(
        visuals["ensemble"],
        6.25,
        "Figure 4. Fixed component weights in the submitted seed-20260824 model.",
        "Horizontal bar chart of final model weights: 33% deep top-50 identity XGBoost, 20% complete-identity CatBoost, 18% accepted Random Forest, 11% spatial-height XGBoost, and two 9% Random Forest representations.",
    )
    doc.paragraph(
        "The final seed comparison did not reopen architecture, feature, or weight selection. Seed 20260824 was already one of three values declared in the archived deep-XGBoost screen. It replaced seed 20260822 because it improved both reused evidence sets while also moving balanced accuracy, macro F1, log loss, and Brier score in the same direction."
    )
    doc.paragraph("Repository evidence: reports/archived-deep-xgboost-screen.md, reports/deep-archive-confirmation.md and reports/deep-follow-up-search.md.", style="TableSource")

    doc.heading(1, "9. Evaluate and interpret the results", page_break_before=True)
    doc.paragraph(
        "The strongest story is consistency, not the size of any one validation gain. Development, used local test, and public scoring all ordered the complete-identity, archive, and deep-archive candidates in the same direction. The margins on reused local evidence were small, but the final public change from 0.8288 to 0.8298 was larger and reached the observed second position."
    )
    doc.figure(
        visuals["evidence"],
        6.35,
        "Figure 5. Accuracy progression on the five development folds and the used local test.",
        "Grouped bars comparing development and local accuracy for six candidates, increasing from the accepted 55:45 vote through source, identity, archive, and two deep-archive seeds.",
    )
    doc.table(
        ["Candidate", "Development", "Local test", "Public"],
        [
            ["Accepted 55:45 XGBoost / RF", "81.6246%", "80.82%", "0.8241"],
            ["Complete-identity vote", "81.7403%", "81.0859%", "0.8259"],
            ["Archive synthesis", "81.7929%", "81.1869%", "0.8288"],
            ["Deep archive, seed 20260822", "81.8771%", "81.3552%", "Not submitted"],
            ["Deep archive, seed 20260824", "81.8981%", "81.3973%", "0.8298"],
        ],
        [3300, 1900, 1900, 2260],
        aligns=["left", "center", "center", "center"],
        caption="Table 7. Comparable headline evidence",
        source="Values are reported at the precision retained in the repository; the accepted local value was published as 80.82%.",
    )
    doc.page_break()
    doc.heading(2, "Public performance timeline")
    doc.figure(
        visuals["timeline"],
        6.35,
        "Figure 6. Public model scores from the early ensemble to the final frozen submission.",
        "Line chart of public classification scores rising from 0.8170 to 0.8298, with the 0.826 project target crossed by archive synthesis and the final deep-archive model.",
    )
    doc.heading(2, "Uncertainty and paired evidence")
    doc.table(
        ["Comparison", "Net local rows", "Exact McNemar p", "Paired bootstrap 95% CI"],
        [
            ["Seed 20260822 vs archive", "+20", "0.135", "-0.0421 to +0.3788 pp"],
            ["Seed 20260824 vs seed 20260822", "+5", "0.542", "-0.0673 to +0.1515 pp"],
        ],
        [3000, 1700, 1800, 2860],
        aligns=["left", "center", "center", "center"],
        caption="Table 8. Row-paired local confirmation",
        source="Repository evidence: reports/deep-archive-confirmation.md and reports/deep-follow-up-search.md.",
    )
    doc.paragraph(
        "Neither p-value nor interval proves a repeatable local gain. Their role was to prevent five or twenty changed decisions from being described with false certainty. The candidate was selected because multiple measures moved coherently and because the model change was pre-bounded, not because the uncertainty analysis crossed a conventional threshold."
    )

    doc.heading(2, "Leaderboard outcome")
    doc.figure(
        visuals["leaderboard"],
        6.4,
        "Figure 7. DrivenData public leaderboard immediately after the final submission.",
        "Screenshot of the DrivenData Pump It Up public leaderboard showing anthonypwatts at rank 2 with a best public classification rate of 0.8298, 0.0001 behind rank 1.",
    )
    doc.paragraph(
        "The screenshot records the public state at the moment of observation: 0.8298, rank 2, fifteen submissions, and a 0.0001 gap to first place. Rank is live and may change. The competition's private score was not available when this report was produced."
    )

    doc.heading(1, "10. Deploy and iterate", page_break_before=True)
    doc.paragraph(
        "Deployment for this project meant a deterministic full-data refit, a structurally validated prediction file, and an unchanged upload to DrivenData. The final file contained 14,850 unique IDs in template order and only documented class values. Its hard predictions were 60.721% functional, 3.643% functional needs repair, and 35.636% non-functional."
    )
    doc.table(
        ["Deployment item", "Frozen value"],
        [
            ["Candidate", "Deep-archive six-component synthesis, seed 20260824"],
            ["Primary XGBoost", "Depth 17, learning rate 0.02, exactly 600 trees"],
            ["Competition file", "01-deep-archive-seed-20260824.csv"],
            ["SHA-256", "fe5de9ea46bad2b35226bc97ebdfb743807fb758df1d259e8f8a51609808fee2"],
            ["Submission status", "Submitted unchanged in the third daily slot on 23 August 2026"],
            ["Verification", "Full project unit suite: 178 tests passed"],
        ],
        [2600, 6760],
        caption="Table 9. Frozen deployment record",
        source="Repository evidence: the seed-20260824 submission README and manifest; test result recorded during final candidate verification.",
    )
    doc.heading(2, "Reproduction commands")
    doc.paragraph("Run from stage-1-pump-it-up with the existing repository virtual environment:")
    doc.rich_paragraph(
        [
            {"text": r"..\.venv\Scripts\python.exe .\scripts\run_deep_seed_20260824_confirmation.py", "font": "Consolas", "size": 9.2, "colour": NAVY},
            {"text": r"..\.venv\Scripts\python.exe .\scripts\generate_deep_archive_submission.py --seed 20260824", "font": "Consolas", "size": 9.2, "colour": NAVY, "break_before": True},
        ],
        shading=LIGHT_GREY,
        border=BLUE,
        before=100,
        after=140,
    )
    doc.heading(2, "The final iteration decision")
    doc.paragraph(
        "The correct next loop is not another public-score-guided seed or blend search. The model, weights, and submission hash are frozen. Future work should record the private leaderboard result when available, preserve the public screenshot as dated evidence, and use genuinely new data or a new competition phase before reopening selection."
    )

    doc.heading(1, "Methods beyond the core course treatment", page_break_before=True)
    doc.paragraph(
        "The course lifecycle supplied the organising discipline. Several technical methods used to complete the competition study went beyond the material needed to understand the ten steps. They are explained here so the report does not hide sophistication behind a model name."
    )
    doc.heading(2, "Leakage-safe nested and cross-fitted target features")
    doc.paragraph(
        "A target encoding estimates a class distribution for an identity such as funder or ward. If an observation contributes its own label to that estimate, validation becomes optimistic. The implementation therefore created training encodings from separate inner folds and transformed outer validation from mappings fitted on outer training only. Spatial outcome rates used the same idea with geographic neighbours and explicitly prevented self-neighbours. These methods were tested honestly and rejected when they failed the promotion gate."
    )
    doc.heading(2, "Native categorical CatBoost")
    doc.paragraph(
        "CatBoost can process categorical identities with ordered, training-sequence-aware statistics rather than expanding every value into a conventional one-hot column. A depth-8 complete-identity CatBoost contribution captured information from funder, installer, waterpoint name, subvillage, ward, and scheme name. It did not replace the whole model; its complementary probability errors justified a fixed 20% role."
    )
    doc.heading(2, "Probability soft voting and representation synthesis")
    doc.paragraph(
        "Soft voting averages class-probability vectors before taking the largest class. Unlike hard majority voting, it retains how strongly each component supports every class. The archive and deep-archive models used different feature representations and learner families so their errors were not identical. Fixed weighted probability synthesis converted that diversity into additional correct rows without inventing a second-stage learner that could overfit the reused folds."
    )
    doc.heading(2, "Archival reconstruction and top-50 identity indicators")
    doc.paragraph(
        "The official solution archive contained a historical 0.8264 approach using common-string indicators and a deep XGBoost configuration. The project reconstructed the idea inside its own fold-safe pipeline rather than copying an old competition file. Six lists of 50 common identities were learned within every fold; the depth-17, 600-tree model was evaluated with explicit modern XGBoost defaults and predeclared seeds. This is reproducible method transfer, not external-label leakage."
    )
    doc.heading(2, "Uncertainty for paired classifier changes")
    doc.paragraph(
        "McNemar's exact test compares the off-diagonal counts: rows only candidate A gets right versus rows only candidate B gets right. The paired bootstrap resamples rows and recalculates the accuracy difference, preserving the pairing. These tools did not turn small gains into proof; they quantified how easily five or twenty net rows could arise under the same finite local sample."
    )
    doc.page_break()
    doc.heading(2, "Transductive, pseudo-label, fuzzy, and ordinal experiments")
    doc.paragraph(
        "Several advanced branches were deliberately treated as experiments rather than automatic improvements. Transductive frequency features allowed unlabelled competition values to influence unsupervised occurrence support; pseudo-labelling added high-confidence predicted competition rows back into training. Uniform fuzzy memberships gave the repair class partial overlap with adjacent states. Cumulative ordinal XGBoost fitted two ordered binary boundaries and projected inconsistent probabilities back to a coherent three-class distribution."
    )
    doc.table(
        ["Branch", "Why it was plausible", "Why it stopped"],
        [
            ["Transductive frequency", "Competition identities reveal support without labels", "No robust promotion beyond frozen baseline"],
            ["Pseudo-labelling", "Confident predictions could enlarge training support", "Risked reinforcing model errors; fixed screens did not justify deployment"],
            ["Fuzzy targets", "Repair may share structure with both outer classes", "Every complete fuzzy vote lost accuracy and log-loss quality"],
            ["Ordinal boundaries", "Repair wording suggests an intermediate state", "Independent boundaries crossed on about one quarter of rows; best archive gain was five rows"],
            ["Oversampling", "More repair examples raised repair recall", "Accuracy fell; 2.5x replay scored 0.8174 publicly"],
        ],
        [2200, 3400, 3760],
        caption="Table 10. Advanced branches that informed but did not define the final model",
        source="Repository evidence: transductive, fuzzy, ordinal, binary-reduction and oversampling reports.",
    )
    doc.rich_paragraph(
        [
            {"text": "General lesson. ", "bold": True, "colour": POSITIVE},
            {"text": "An advanced technique earns a place only when its evidence is better than its validation and operational risk. Sophistication was diagnostic unless it improved the frozen comparison."},
        ],
        shading=CALLOUT,
        border=GOLD,
        before=120,
        after=160,
    )

    doc.heading(1, "Limitations and responsible interpretation", page_break_before=True)
    doc.heading(2, "Leaderboard selection risk")
    doc.paragraph(
        "Public leaderboard scores are rounded aggregates over a competition-defined subset. Fifteen submissions provide useful outcome evidence but also create a temptation to select on public noise. The final score was not used to change the model again. The observed #2 position and 0.0001 gap are snapshots, and the private leaderboard may order models differently."
    )
    doc.heading(2, "Validation reuse")
    doc.paragraph(
        "The five development folds were deliberately fixed, but a long sequence of bounded experiments still adapts to them. The local test was initially one-time evidence and later became a used confirmation set. Reported paired intervals are honest about finite-sample uncertainty but cannot remove adaptive-selection bias. A genuinely fresh labelled sample would be the strongest next evaluation."
    )
    doc.heading(2, "Class and domain limits")
    doc.bullet("Repair recall remained far below functional recall; high overall accuracy does not imply equal service across states.")
    doc.bullet("The model describes recorded condition at survey time and may drift as pumps, reporting practices, or maintenance systems change.")
    doc.bullet("Administrative names and geography can encode operational context, but they can also proxy for uneven data coverage or service provision.")
    doc.bullet("Missing or invalid coordinates required explicit fallbacks; 1,812 labelled locations were excluded from the mapped-location count.")
    doc.bullet("A field deployment would require current labels, cost-sensitive decision design, monitoring, and human review beyond this competition artefact.")
    doc.heading(2, "What remains unknown")
    doc.paragraph(
        "Private-leaderboard performance, future temporal stability, causal maintenance impact, and operational cost trade-offs were not available. The report therefore claims a reproducible competition result and a strong public ranking - not a production-ready water-service allocation system."
    )

    doc.heading(1, "Conclusion", page_break_before=True)
    doc.paragraph(
        "The project exceeded its 0.826 target and reached an observed public rank of 2 with a 0.8298 score. More importantly, it produced a defensible chain from immutable source data through fold-safe preprocessing, bounded model search, paired confirmation, deterministic full-data refit, and an unchanged submission hash."
    )
    doc.paragraph(
        "The final margin came from combining representations rather than from a single dramatic algorithmic discovery. Native identity handling, occurrence support, reconstructed height, Random Forest diversity, and a deep top-50-identity XGBoost component each contributed probability information. The seed-20260824 candidate was a narrow final refinement within a fixed recipe, not a reopened hyperparameter sweep."
    )
    doc.rich_paragraph(
        [
            {"text": "Final position. ", "bold": True, "colour": POSITIVE},
            {"text": "Freeze the 0.8298 model and its fe5de9ea...08fee2 hash. Record private performance when it becomes available. Reopen modelling only with genuinely new evidence, not because a live leaderboard moves."},
        ],
        shading=CALLOUT,
        border=TEAL,
        before=160,
        after=180,
    )

    doc.heading(1, "Appendix: evidence register", page_break_before=True)
    doc.paragraph(
        "The report is a synthesis of repository evidence. The following artefacts are the most direct route to the exact claims, recipes, and decisions; the reports index contains the full experiment history. Paths are relative to stage-1-pump-it-up."
    )
    doc.table(
        ["Repository artefact", "Evidence supplied"],
        [
            ["reports/data-preparation-next-steps.md", "Rows, class balance, structural removals, partition and early model progression"],
            ["reports/training-label-provenance-and-semantics.md", "Target meaning and interpretation boundaries"],
            ["reports/cross-fitted-target-encoding-screen.md", "Nested leakage-safe identity rates and rejection evidence"],
            ["reports/cross-fitted-spatial-outcome-screen.md", "Neighbour leakage controls and spatial-rate rejection"],
            ["reports/archive-synthesis-confirmation.md", "Six-component recipe and 0.8288 candidate confirmation"],
            ["reports/deep-archive-confirmation.md", "Depth-17 substitution, local metrics, uncertainty and seed-20260822 file"],
            ["reports/deep-follow-up-search.md", "Seed-20260824 comparison, final public result and bounded rejected follow-ups"],
            ["submissions/2026-08-23-deep-archive-seed-20260824/", "Final manifest, hash, class shares and reproduction commands"],
            ["project-status.json", "Dated score, rank, submission count and dashboard summary"],
        ],
        [4100, 5260],
        caption="Table 11. Primary evidence paths",
        source="The experiment-specific Markdown reports provide additional method, fold, runtime and negative-result detail.",
    )
    doc.heading(2, "Report production note")
    doc.paragraph(
        "This document was generated reproducibly from generate_report.py using the repository's existing Python runtime. Charts were created from the numerical evidence cited above; the leaderboard image is the supplied screenshot. No competition identifiers or raw prediction rows are embedded in the report."
    )
    return doc


def content_types_xml(images: list[ImageRef]) -> str:
    image_extensions = sorted({image.path.suffix.lower().lstrip(".") for image in images})
    defaults = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
    ]
    for extension in image_extensions:
        content_type = "image/png" if extension == "png" else f"image/{extension}"
        defaults.append(f'<Default Extension="{extension}" ContentType="{content_type}"/>')
    overrides = [
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>',
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>',
        '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>',
        '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>',
        '<Override PartName="/word/fontTable.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml"/>',
        '<Override PartName="/word/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
        '<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>',
        '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' + "".join(defaults + overrides) + "</Types>"


def document_relationships_xml(images: list[ImageRef]) -> str:
    relationships = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>',
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>',
        '<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>',
        '<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
        '<Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable" Target="fontTable.xml"/>',
        '<Relationship Id="rId7" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>',
    ]
    relationships.extend(
        f'<Relationship Id="{image.rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{image.media_name}"/>'
        for image in images
    )
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + "".join(relationships) + "</Relationships>"


def write_docx(doc: DocxBuilder, output: Path) -> None:
    package_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    core = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Pump It Up - Model Development Report</dc:title><dc:subject>Course-aligned machine-learning report</dc:subject>
  <dc:creator>Anthony P. Watts</dc:creator><cp:lastModifiedBy>Anthony P. Watts</cp:lastModifiedBy>
  <dc:description>Evidence-led account of the Pump It Up model-development lifecycle and final 0.8298 public result.</dc:description>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-08-23T00:00:00Z</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">2026-08-23T00:00:00Z</dcterms:modified>
</cp:coreProperties>'''
    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex document workflow</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop><Company></Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion></Properties>'''
    settings = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings xmlns:w="{NS['w']}"><w:zoom w:percent="100"/><w:defaultTabStop w:val="720"/><w:updateFields w:val="true"/><w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat></w:settings>'''
    font_table = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:fonts xmlns:w="{NS['w']}"><w:font w:name="Calibri"><w:family w:val="swiss"/><w:pitch w:val="variable"/></w:font><w:font w:name="Consolas"><w:family w:val="modern"/><w:pitch w:val="fixed"/></w:font></w:fonts>'''
    theme = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office"><a:themeElements><a:clrScheme name="Office"><a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1><a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="0B2545"/></a:dk2><a:lt2><a:srgbClr val="F2F4F7"/></a:lt2><a:accent1><a:srgbClr val="2E74B5"/></a:accent1><a:accent2><a:srgbClr val="2F7F76"/></a:accent2><a:accent3><a:srgbClr val="C9962D"/></a:accent3><a:accent4><a:srgbClr val="6B8EAD"/></a:accent4><a:accent5><a:srgbClr val="86A79F"/></a:accent5><a:accent6><a:srgbClr val="A7BBCD"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme><a:fontScheme name="Office"><a:majorFont><a:latin typeface="Calibri Light"/></a:majorFont><a:minorFont><a:latin typeface="Calibri"/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme></a:themeElements></a:theme>'''

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(doc.images))
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)
        archive.writestr("word/document.xml", doc.document_xml())
        archive.writestr("word/styles.xml", styles_xml())
        archive.writestr("word/numbering.xml", numbering_xml())
        archive.writestr("word/settings.xml", settings)
        archive.writestr("word/fontTable.xml", font_table)
        archive.writestr("word/theme/theme1.xml", theme)
        archive.writestr("word/header1.xml", header_xml())
        archive.writestr("word/footer1.xml", footer_xml())
        archive.writestr("word/_rels/document.xml.rels", document_relationships_xml(doc.images))
        for image in doc.images:
            archive.write(image.path, f"word/media/{image.media_name}")


def normalise_with_word(source: Path, output: Path) -> None:
    """Open-and-repair the generated package, then save a canonical DOCX.

    Word's object model normalises ordering and compatibility details that are
    deliberately omitted from the compact direct-OOXML writer.  This keeps the
    checked-in generator dependency-free while ensuring the deliverable opens
    cleanly in Word and preserves accessibility tags, styles and table widths.
    """

    def ps_quote(path: Path) -> str:
        return "'" + str(path.resolve()).replace("'", "''") + "'"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    command = (
        f"$source={ps_quote(source)}; $dest={ps_quote(output)}; "
        "$word=New-Object -ComObject Word.Application; "
        "$word.Visible=$false; $word.DisplayAlerts=0; "
        "try { "
        "$doc=$word.Documents.Open($source,$false,$false,$false,'','',$false,'','',0,0,$false,$true,0,$true,''); "
        "$doc.Repaginate(); $doc.Fields.Update(); $doc.SaveAs2($dest,16); $doc.Close($false); "
        "} finally { $word.Quit() }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not output.exists():
        raise RuntimeError(
            "Word normalisation failed.\n"
            + result.stdout
            + ("\n" if result.stdout and result.stderr else "")
            + result.stderr
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path, default=DEFAULT_SCREENSHOT, help="Rank-2 leaderboard screenshot")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="DOCX output path")
    args = parser.parse_args()
    visuals = generate_visuals(args.screenshot)
    document = build_document(visuals)
    with tempfile.TemporaryDirectory(prefix="pump_it_up_report_") as temp_dir:
        raw_path = Path(temp_dir) / "pump-it-up-model-development-report.raw.docx"
        write_docx(document, raw_path)
        normalise_with_word(raw_path, args.output)
    print(f"Created {args.output}")
    print(f"Embedded {len(document.images)} figures")


if __name__ == "__main__":
    main()
