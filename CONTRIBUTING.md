# Contributing

Thanks for helping build open, interactive teaching materials in chemical and analytical science. This repository hosts **Open Educational Resources (OER)**: narrative text and figures are **CC BY 4.0**, example code is **MIT**. By contributing, you agree to release your contributions under these licenses.

---

## Scope & Audience

* **Goal:** Short, interactive notebooks that turn core chemical and analytical concepts (e.g., chemical equilibrium, ideal gas, diffusion, van Deemter) into hands-on learning.
* **Audience:** Students and practitioners in (analytical) chemistry; no prior coding required to *use* notebooks.

---

## Code of Conduct

Be respectful and constructive. Assume positive intent. Critique ideas, not people.

---

## Repository Structure

```
/notebooks/           # Jupyter notebooks (numbered, kebab-case)
  01-van-deemter.ipynb
/templates/           # Notebook template(s), reusable assets
  notebook_template.ipynb
/env/                 # Reproducible environments
  environment.yml
  requirements.txt
/binder/              # Binder config (mirrors /env when used)
/data/                # Small, open datasets (<5 MB per file, CSV/TSV/JSON only)
README
LICENSE            # CC BY 4.0 (content), MIT (code)
```

**Naming:** use `NN-topic-name.ipynb` (e.g., `02-chemical-equilibrium.ipynb`). Keep file names short, lowercase, kebab-case.

---

## How to Set Up (Local)

1. **Clone** the repo and choose an environment route:

   * Conda: `conda env create -f env/environment.yml && conda activate chromatography-oer`
   * Pip: `python -m venv .venv && source .venv/bin/activate && pip install -r env/requirements.txt`
2. **Launch** JupyterLab: `jupyter lab` and open a notebook.
3. **Run All** cells to verify the environment. Keep total runtime for core notebooks **< 2 minutes** on a typical laptop.

**Binder/Colab:** Notebooks should run via the README badges. If you change dependencies, update `/env` and (if used) `/binder`.

---

## Adding a New Notebook

1. **Start from the template:** `templates/notebook_template.ipynb`.
2. **Keep the canonical sections:**

   * Title & subtitle
   * Overview (2–4 sentences)
   * Learning objectives (3–6 bullets)
   * Theory (brief, correct, visual-first)
   * Interactive section(s)
   * Scenarios / guided explorations
   * Check-your-understanding (5–8 items)
   * References (with DOIs where possible)
   * Reproducibility & license notes
3. **Interactivity guidelines:**

   * Prefer **ipywidgets** (sliders, dropdowns). Provide sensible defaults and **units**.
   * Add a one-sentence **interpretation** below each widget/plot.
   * Keep interactions **fast** (<200 ms per update) to support Binder/Colab.
4. **Accessibility:**

   * Do not rely on color alone; include labels/markers.
   * Provide **captions** or a brief **alt-style description** under figures.
   * Use readable font sizes in plots; axis labels with **units**.
5. **Data:** Use small, open data only. No proprietary or sensitive content. Place sample data in `/data` with a short README (source, license).

---

## Style Guide

* **Language:** English, concise and student-friendly.
* **Math:** LaTeX for symbols/variables (e.g., \$H(u)\$, \$u\_\text{opt}\$). Variables in italics; vectors/matrices bold if needed.
* **Units:** SI with spaces (e.g., `1.0 mL min⁻¹`, `25 °C`).
* **Figures:** Label axes and include units; keep file sizes small.
* **Citations:** Prefer primary literature; include DOI or stable URL in a reference list at the end of the notebook.
* **File size:** Keep notebooks **< 10 MB**. Avoid embedding large outputs or binaries.

---

## Reproducibility

* Pin versions in `/env/requirements.txt` or `/env/environment.yml`.
* Set random seeds where randomness affects outputs.
* Ensure notebooks run **top-to-bottom without errors** on a clean environment.
* Prefer **cleared heavy outputs** before commit; lightweight pedagogical images are OK if they help offline reading.

---

## Licensing & Attribution

* **Content (text/figures generated here):** CC BY 4.0 → include an attribution note where substantial reuse occurs.
* **Code (snippets, helpers):** MIT.
* If you include third-party material, ensure it is license-compatible and **credit the source** clearly in the cell/figure caption.

---

## Issues

Use clear titles and labels:

* `type:bug`, `type:enhancement`, `type:content`, `type:accessibility`, `good first issue`
  Include:
* Steps to reproduce (for bugs)
* Screenshots or cell outputs (if relevant)
* Environment info (local vs Binder/Colab)

---

## Pull Requests

**Branch naming:** `feat/<topic>`, `fix/<short>`, `docs/<short>`

**PR checklist (copy into your PR):**

* [ ] Notebook follows the template sections.
* [ ] Runs top-to-bottom without errors in a fresh env.
* [ ] Execution time acceptable (core < 2 min).
* [ ] Dependencies updated in `/env` (and `/binder` if applicable).
* [ ] Large outputs cleared; figures have captions and units.
* [ ] References include DOIs where possible.
* [ ] Licensing/attribution is correct (CC BY 4.0 for content, MIT for code).

Keep PRs focused and reasonably small. Add a short note on **pedagogical intent** (“This slider highlights the trade-off between A and B.”).

---

## Governance & Contact

Maintainer: **Christian Haas**.
Start with an **issue** for discussion before large changes. For questions about scope, pedagogy, or licensing, open an issue with label `question`.
