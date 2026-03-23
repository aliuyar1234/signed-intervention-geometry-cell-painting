# Signed Intervention Geometry in Cell Painting

[![DOI](https://zenodo.org/badge/1189558738.svg)](https://doi.org/10.5281/zenodo.19185214)
[![PDF](https://img.shields.io/badge/PDF-manuscript-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://github.com/aliuyar1234/signed-intervention-geometry-cell-painting/raw/main/paper/main.pdf)

This repository contains the code, manuscript, and compact evidence bundle for a locked reanalysis of matched ORF and CRISPR Cell Painting profiles in the JUMP `cpg0016` resource.

The central result is **structured yet fragile**: same-gene ORF/CRISPR morphology is biologically organized, but the final regime map is dominated by asymmetric and ambiguous states and does not support a strong retrieval-style positive claim.

## At a glance

- Manuscript PDF: `paper/main.pdf`
- LaTeX source: `paper/main.tex`
- Figure source data: `paper/source_data/`
- Summary evidence bundle: `paper/evidence/`
- Primary implementation: `src/siggeo/`

## Key findings

- A same-gene ORF/CRISPR morphological regime structure exists.
- The final regime landscape is dominated by `ambiguous`, `asymmetric_CRISPR`, and `asymmetric_ORF` labels.
- Biological enrichment survives, especially in asymmetric regimes.
- Robustness is mixed across preprocessing variants rather than uniformly strong.
- Retrieval is too weak to support a utility-forward or predictive claim.

## Main outputs

- Paper PDF: `paper/main.pdf`
- LaTeX source: `paper/main.tex`
- Figure source data: `paper/source_data/`
- Summary evidence files: `paper/evidence/`

## Repository layout

- `src/siggeo/`
  Python implementation of the pipeline, including preprocessing, gene-level statistics, signed geometry, retrieval, enrichment, confounds, figures, and paper/report assembly.
- `scripts/`
  Numbered entrypoints for the full phase-driven pipeline.
- `configs/`
  Official, smoke, and robustness run configurations.
- `tests/`
  Unit and integration-style tests for the pipeline.
- `paper/`
  Root-level manuscript package with PDF, LaTeX source, figures, evidence summaries, and source-data CSVs.

## What is intentionally not tracked

To keep the repository lightweight and publication-friendly, the following are excluded from git:

- downloaded raw data
- local metadata databases
- full run directories and memmaps
- logs, caches, and temporary files
- internal planning, handoff, and agent-facing markdown

The full local run bundles remain available outside the tracked GitHub-facing file set.

## Installation

1. Create a Python 3.11 environment.
2. Install the package:

```bash
python -m pip install -e .
```

## Reproducing the pipeline

Download the public assets:

```bash
python scripts/00_download_assets.py --config configs/paths.example.yaml
python scripts/01_build_metadata_db.py --config configs/paths.example.yaml
```

Run the smoke pipeline:

```bash
python -m siggeo.cli run-pipeline --config configs/analysis_smoke.yaml
```

Run the official pipeline:

```bash
python -m siggeo.cli run-pipeline --config configs/analysis_official.yaml
```

## Reproducing the manuscript

The top-level manuscript package is self-contained:

```powershell
powershell -ExecutionPolicy Bypass -File .\paper\build.ps1
```

That rebuilds `paper/main.pdf` using the local `paper/figures/` assets.

## Interpretation

The cleanest evidence-supported framing of this project is not a triumphant discovery claim. It is a constrained research result showing that:

- a same-gene ORF/CRISPR regime structure exists
- that structure is predominantly asymmetric rather than cleanly symmetric
- robustness is mixed across preprocessing variants
- biological enrichment survives
- retrieval remains too weak for a utility-forward claim

## License

This repository is released under the MIT License. See `LICENSE`.

## Citation

If you use this repository, cite the manuscript in `paper/main.pdf`.
