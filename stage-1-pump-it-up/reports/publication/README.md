# Pump It Up publication report

`pump-it-up-model-development-report.docx` is the publication-format account
of the Stage 1 Pump It Up study through the **23 August 2026** public result
(`0.8298`, observed rank 2), revised on **28 August 2026**. It is structured as
a research article. It predates the September generalisation investigation,
the `0.8304` result and the native-category follow-up; see the
[Stage 1 README](../../README.md) for the latest recorded position.

## Rebuild

From the repository root, use a Python 3.11+ environment with the packages in
`requirements-publication.txt` and Poppler's `pdftoppm` on `PATH`:

```powershell
python .\stage-1-pump-it-up\reports\publication\generate_report.py
```

The generator reconstructs the nine figure plates from the versioned CSV
files or recorded model-development evidence, rasterises their vector masters
at 360 dpi and embeds them in an editable DOCX. Microsoft Word performs a
final open, repaginate and save pass. On a machine without Word, pass
`--no-word-normalise` and inspect the result in the target word processor.

Run the dependency-free OOXML audit with:

```powershell
python .\stage-1-pump-it-up\reports\publication\audit_report.py
```

The audit checks UK-English metadata, heading hierarchy, genuine Word lists,
fixed table geometry, repeating table headers, inline image references,
meaningful alternative text, and A4 section geometry.

## Design contract

- A4 portrait (210 × 297 mm) with 0.75-inch side margins and a restrained running head.
- Cambria body text in a compact two-column scholarly layout.
- Full-width, single-column analytical plates at major evidence transitions.
- A restrained navy, teal, gold and burgundy system that remains legible in
  print and does not encode meaning by colour alone.
- Structured abstract, numbered academic sections, equation, numbered tables,
  numbered figures, primary-source bibliography, limitations and a
  reproducibility appendix.
- Inline figures with descriptive alternative text and interpretive captions.

Generated PDFs, page renders and audit JSON files are QA intermediates under
`.codex-redesign/`; the DOCX is the publication artefact.
