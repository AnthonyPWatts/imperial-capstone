"""Dependency-free structural, image and accessibility audit for the report DOCX."""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "ep": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
}


def q(prefix: str, local: str) -> str:
    return f"{{{NS[prefix]}}}{local}"


def attr(element: ET.Element | None, prefix: str, name: str) -> str | None:
    return None if element is None else element.get(q(prefix, name))


def text_of(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.iter(q("w", "t"))).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "docx",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("pump-it-up-model-development-report.docx"),
    )
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    with zipfile.ZipFile(args.docx) as archive:
        names = set(archive.namelist())

        def xml(name: str) -> ET.Element:
            return ET.fromstring(archive.read(name))

        document = xml("word/document.xml")
        styles = xml("word/styles.xml")
        core = xml("docProps/core.xml")
        app = xml("docProps/app.xml")
        authored_part_names = [
            name
            for name in names
            if name == "word/document.xml"
            or re.fullmatch(r"word/(?:header|footer)\d+\.xml", name)
        ]
        authored_parts = {name: xml(name) for name in authored_part_names}
        authored_text = "\n".join(text_of(part) for part in authored_parts.values())

        dash_characters = {
            "\u2010": "hyphen",
            "\u2011": "non-breaking hyphen",
            "\u2012": "figure dash",
            "\u2013": "en dash",
            "\u2014": "em dash",
            "\u2015": "horizontal bar",
        }
        found_dashes = sorted(
            {name for character, name in dash_characters.items() if character in authored_text}
        )
        if found_dashes:
            failures.append(f"Non-ASCII dash characters found: {found_dashes}")

        forbidden_fragments = [
            "RESEARCH ARTICLE  •  MULTICLASS CLASSIFICATION  •  REPRODUCIBLE MODEL DEVELOPMENT",
            "Imperial College London",
            "scientifically credible",
            "restrained conclusion",
            "equally informative",
            "At this point",
            "central methodological result",
            "breakthrough effects",
            "presenting the final model as inevitable",
            "deliberately separates",
            "management, extraction, payment, quantity, quality and construction",
            "linear, tree, nearest-neighbour, boosting and bagging",
            "development, local-test and public",
            "technology, climate and recording standards",
        ]
        found_fragments = [fragment for fragment in forbidden_fragments if fragment in authored_text]
        if found_fragments:
            failures.append(f"Editorially prohibited text found: {found_fragments}")

        style_names: dict[str, str] = {}
        for style in styles.findall("w:style", NS):
            style_id = attr(style, "w", "styleId")
            name = attr(style.find("w:name", NS), "w", "val")
            if style_id and name:
                style_names[style_id] = name

        audited_style_ids = {
            "Normal",
            "Title",
            "Subtitle",
            "Heading1",
            "Heading2",
            "Heading3",
            "Caption",
            "Quote",
            "ListBullet",
            "ListNumber",
        }
        font_values: set[str] = set()
        for part in authored_parts.values():
            for fonts in part.findall(".//w:rFonts", NS):
                for key in ("ascii", "hAnsi", "eastAsia", "cs"):
                    value = attr(fonts, "w", key)
                    if value:
                        font_values.add(value)
        for style in styles.findall("w:style", NS):
            style_id = attr(style, "w", "styleId") or ""
            if style_id not in audited_style_ids:
                continue
            for fonts in style.findall(".//w:rFonts", NS):
                for key in ("ascii", "hAnsi", "eastAsia", "cs"):
                    value = attr(fonts, "w", key)
                    if value:
                        font_values.add(value)
        if font_values != {"Cambria"}:
            failures.append(f"Authored text does not use the single-font specification: {sorted(font_values)}")

        title = (core.findtext("dc:title", default="", namespaces=NS) or "").strip()
        language_nodes = styles.findall(".//w:lang", NS)
        languages = {attr(node, "w", "val") for node in language_nodes if attr(node, "w", "val")}
        language = "en-GB" if "en-GB" in languages else next(iter(languages), None)
        if not title:
            failures.append("Document title metadata is missing")
        if not language:
            failures.append("Document language metadata is missing")

        headings: list[dict[str, object]] = []
        previous_level: int | None = None
        fake_bullets: list[str] = []
        placeholder_pattern = re.compile(r"\b(?:TODO|TBD|FIXME|XXX)\b|\{\{.+?\}\}", re.I)
        placeholders: list[str] = []
        numbered_paragraphs = 0
        for paragraph in document.findall(".//w:p", NS):
            paragraph_text = text_of(paragraph)
            style_id = attr(paragraph.find("w:pPr/w:pStyle", NS), "w", "val") or ""
            style_name = style_names.get(style_id, style_id)
            match = re.fullmatch(r"heading\s*([1-9])", style_name, re.I)
            if match:
                level = int(match.group(1))
                headings.append({"level": level, "text": paragraph_text})
                if previous_level is not None and level > previous_level + 1:
                    failures.append(
                        f"Heading hierarchy skips from level {previous_level} to {level}: {paragraph_text}"
                    )
                previous_level = level
            if paragraph.find("w:pPr/w:numPr", NS) is not None or style_name in {"List Bullet", "List Number"}:
                numbered_paragraphs += 1
            if not match and re.match(r"^\s*(?:[-*•‣▪]|\d+[.)])\s+", paragraph_text):
                fake_bullets.append(paragraph_text[:80])
            if placeholder_pattern.search(paragraph_text):
                placeholders.append(paragraph_text[:120])

        if not headings:
            failures.append("No semantic headings were found")
        if fake_bullets:
            failures.append(f"Fake bullet/list prefixes found: {fake_bullets}")
        if placeholders:
            failures.append(f"Placeholder text found: {placeholders}")
        if numbered_paragraphs == 0:
            failures.append("No paragraphs use real Word numbering")

        tables = document.findall(".//w:tbl", NS)
        table_findings: list[dict[str, object]] = []
        for number, table in enumerate(tables, start=1):
            table_width = int(attr(table.find("w:tblPr/w:tblW", NS), "w", "w") or -1)
            table_indent = int(attr(table.find("w:tblPr/w:tblInd", NS), "w", "w") or -1)
            table_alignment = attr(table.find("w:tblPr/w:jc", NS), "w", "val") or "left"
            grid_widths = [int(attr(node, "w", "w") or 0) for node in table.findall("w:tblGrid/w:gridCol", NS)]
            rows = table.findall("w:tr", NS)
            repeating_header = bool(rows and rows[0].find("w:trPr/w:tblHeader", NS) is not None)
            row_widths = []
            for row in rows:
                cell_widths = [
                    int(attr(node, "w", "w") or 0)
                    for node in row.findall("w:tc/w:tcPr/w:tcW", NS)
                ]
                row_widths.append(sum(cell_widths))
            finding = {
                "number": number,
                "width_dxa": table_width,
                "indent_dxa": table_indent,
                "alignment": table_alignment,
                "grid_width_dxa": sum(grid_widths),
                "row_widths_dxa": row_widths,
                "repeating_header": repeating_header,
            }
            table_findings.append(finding)
            if (
                table_width != 4693
                or sum(grid_widths) != table_width
                or table_indent not in {-1, 0}
                or table_alignment != "left"
            ):
                failures.append(f"Table {number} has inconsistent fixed geometry: {finding}")
            if any(width != table_width for width in row_widths):
                failures.append(f"Table {number} has a row whose cell widths do not match the declared width")
            if len(rows) > 1 and not repeating_header:
                failures.append(f"Table {number} does not mark its first row as a repeating header")

        doc_properties = document.findall(".//wp:docPr", NS)
        alt_texts = [(node.get("descr") or "").strip() for node in doc_properties]
        inline_images = document.findall(".//wp:inline", NS)
        floating_images = document.findall(".//wp:anchor", NS)
        blips = document.findall(".//a:blip", NS)
        media = sorted(name for name in names if name.startswith("word/media/") and not name.endswith("/"))
        if not doc_properties:
            failures.append("No image drawing properties were found")
        if any(not description for description in alt_texts):
            failures.append("At least one image has empty alternative text")
        if floating_images:
            failures.append(f"Found {len(floating_images)} floating image(s); figures must be inline")
        if len(inline_images) != len(media) or len(blips) != len(media):
            failures.append(
                f"Image counts disagree: {len(media)} media, {len(inline_images)} inline drawings, {len(blips)} references"
            )
        duplicate_alts = sorted({alt for alt in alt_texts if alt_texts.count(alt) > 1})
        if duplicate_alts:
            warnings.append(f"Duplicate image alternative text: {duplicate_alts}")

        section = document.find(".//w:sectPr", NS)
        page_size = section.find("w:pgSz", NS) if section is not None else None
        page_margins = section.find("w:pgMar", NS) if section is not None else None
        section_geometry = {
            "page_width_dxa": int(attr(page_size, "w", "w") or -1),
            "page_height_dxa": int(attr(page_size, "w", "h") or -1),
            "margin_top_dxa": int(attr(page_margins, "w", "top") or -1),
            "margin_right_dxa": int(attr(page_margins, "w", "right") or -1),
            "margin_bottom_dxa": int(attr(page_margins, "w", "bottom") or -1),
            "margin_left_dxa": int(attr(page_margins, "w", "left") or -1),
            "header_dxa": int(attr(page_margins, "w", "header") or -1),
            "footer_dxa": int(attr(page_margins, "w", "footer") or -1),
        }
        expected_geometry = {
            "page_width_dxa": 11906,
            "page_height_dxa": 16838,
            "margin_top_dxa": 1008,
            "margin_right_dxa": 1080,
            "margin_bottom_dxa": 936,
            "margin_left_dxa": 1080,
            "header_dxa": 432,
            "footer_dxa": 432,
        }
        if section_geometry != expected_geometry:
            failures.append(f"Page geometry differs from the selected preset: {section_geometry}")

        report = {
            "document": str(args.docx.resolve()),
            "metadata": {
                "title": title,
                "language": language,
                "cached_package_pages": int(app.findtext("ep:Pages", default="0", namespaces=NS) or 0),
                "cached_package_words": int(app.findtext("ep:Words", default="0", namespaces=NS) or 0),
            },
            "structure": {
                "headings": headings,
                "semantic_heading_count": len(headings),
                "real_numbered_paragraph_count": numbered_paragraphs,
                "table_count": len(tables),
                "tables": table_findings,
                "section_geometry": section_geometry,
            },
            "editorial": {
                "authored_font_values": sorted(font_values),
                "non_ascii_dash_types": found_dashes,
                "prohibited_fragments": found_fragments,
                "heading_4_1_present": "4.1  Representation-specific candidate models" in authored_text,
                "heading_4_2_present": "4.2  Evaluation of candidate modifications" in authored_text,
            },
            "images": {
                "media_count": len(media),
                "inline_count": len(inline_images),
                "floating_count": len(floating_images),
                "alt_text_count": len(alt_texts),
                "alt_texts": alt_texts,
            },
            "warnings": warnings,
            "failures": failures,
            "passed": not failures,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
