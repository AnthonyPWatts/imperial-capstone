# Pump It Up publication report

`pump-it-up-model-development-report.docx` is the publication-quality account
of the Stage 1 Pump It Up model-development lifecycle and final public result.

## Rebuild

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe .\stage-1-pump-it-up\reports\publication\generate_report.py
```

The generator uses the existing repository virtual environment, Matplotlib,
Pillow and the Python standard library. It writes WordprocessingML directly;
no new package is required. Installed Microsoft Word then performs a final
open-and-repair/save pass to normalise package ordering and compatibility
metadata. The supplied leaderboard screenshot is copied into `assets/` during
generation and embedded in the DOCX.

Run the dependency-free OOXML image, accessibility and geometry audit with:

```powershell
.\.venv\Scripts\python.exe .\stage-1-pump-it-up\reports\publication\audit_report.py
```

The audit checks document language and title metadata, semantic heading order,
real list numbering, fixed table geometry and repeating header rows, inline
figure placement, image-reference counts and meaningful alternative text.

## Design contract

The report uses the documents skill's `standard_business_brief` preset and the
`editorial_cover` opening pattern. The template picker was unavailable, so the
specified preset was applied as the documented fallback.

- US Letter portrait with 1 inch margins and 6.5 inch / 9360 DXA content width.
- Calibri 11 pt body, 6 pt after, 1.10 line spacing.
- Heading 1/2/3 at 16/13/12 pt with the preset blue hierarchy and exact
  16/8, 12/6 and 8/4 pt spacing.
- Real Word numbering definitions with 0.25 inch markers, 0.5 inch text and
  0.25 inch hanging indents.
- Fixed-DXA tables with 9360 DXA total width, 120 DXA indent, matching grid and
  cell widths, 80/120 DXA cell margins and repeating header rows.
- Inline images with meaningful alt text and separate caption paragraphs.

Generated render PNGs, PDFs and audit JSON are QA intermediates and are kept
outside this publication directory.
