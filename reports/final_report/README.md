# Project report

Open the [nine-page PDF report](output/pdf/predeparture_flight_delay_report.pdf).
The [editable Markdown source](predeparture_flight_delay_report.md) contains the same prose, tables and figures.

The first seven pages focus on sources, cleaning, exploratory data analysis, weather extraction, feature creation, transformations and sampling. The last two cover the four classifiers, the objective findings, limitations and references. Six figures explain the findings alongside the preparation decisions they support.

The report distinguishes January-September 2025 EDA from final all-2025 model refits and the frozen July 2026 evaluation. Earlier Q1/Q2 results are historical benchmarks. It uses the executed project evidence; no models were refitted or selected while writing it.

Supporting files:

- [Evidence and frozen-source hashes](report_evidence.json).
- [Report verification](verification.json).
- [Individual report figures](figures/).
- [Report builder](../../scripts/build_final_report.py).

To regenerate the PDF, Markdown, figures and evidence from the existing project handoffs:

```bash
python3 scripts/build_final_report.py
```

Run from the project root. In addition to the project's analysis packages, PDF generation uses ReportLab and the macOS Arial fonts. Visual verification used PyMuPDF to render all nine pages. Regeneration reads the saved preparation handoffs and July exports; it does not run the notebooks or alter the fitted models. Changes to prose should be made in the builder because regeneration replaces the Markdown copy.
