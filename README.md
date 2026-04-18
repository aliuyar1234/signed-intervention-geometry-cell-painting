# Structured Yet Fragile: Signed Intervention Geometry of Matched ORF and CRISPR Cell Painting Profiles

[![DOI](https://zenodo.org/badge/1189558738.svg)](https://doi.org/10.5281/zenodo.19185214)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](paper/main.pdf)
[![Manuscript Source](https://img.shields.io/badge/LaTeX-Source-1D4ED8?style=flat-square&logo=latex&logoColor=white)](paper/main.tex)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/aliuyar1234/signed-intervention-geometry-cell-painting)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Scope](https://img.shields.io/badge/Scope-Locked%20Analysis%20%2B%20Fragility%20Result-5B4B8A?style=flat-square)](#scope)

Ali Uyar
Independent Researcher

**Paper title:** *Structured Yet Fragile: Signed Intervention Geometry of Matched ORF and CRISPR Cell Painting Profiles*

This repository accompanies a locked reanalysis of matched ORF and CRISPR Cell Painting profiles in the JUMP `cpg0016` genetic resource. It asks whether same-gene gain- and loss-of-function perturbations occupy a signed regime space with interpretable axes (alignment, inversion, orthogonality, modality-specific asymmetry). The central result is deliberately non-triumphal: same-gene morphology is biologically organized, but the final regime map is dominated by asymmetric and ambiguous states and does not support a retrieval-style positive claim.

## Abstract

Matched gain- and loss-of-function perturbations are an attractive setting in which to ask whether cellular morphology occupies a signed regime space with interpretable axes such as alignment, inversion, orthogonality, and modality-specific asymmetry. We tested that question in a locked, auditable pipeline on the `cpg0016` subset of the JUMP Cell Painting genetic resource, focusing on same-gene ORF and CRISPR pairs. Across 5,332 analyzed gene pairs, 2,552 were active in both modalities and 2,724 received confident non-inactive labels after bootstrap consensus. The resulting landscape was structured but not cleanly symmetric: 48.9% of pairs were labeled ambiguous, 30.3% asymmetric_CRISPR, 20.8% asymmetric_ORF, and only 3 genes inactive; no aligned, inverse, or orthogonal labels survived the final consensus layer. Robustness was mixed rather than absent. Agreement was high under no-final-centering (0.951 primary-only agreement) and very high under the dedicated artifact-sensitivity comparison (0.994), but only moderate under the interpretable pipeline (0.758), and regime prevalence shifted strongly across mean-strength deciles (maximum absolute fraction shift 0.434). Biological signal remained detectable despite this fragility, with 536 significant Reactome terms concentrated in asymmetric regimes, whereas retrieval remained effectively null-like and did not support a useful regime-aware ranking claim. The locked evidence therefore supports a restrained but non-null conclusion: same-gene ORF and CRISPR morphology occupies a biologically meaningful regime map, yet that map is dominated by asymmetry, preprocessing dependence, and fragility rather than by a robust symmetric signed geometry.

## Main Finding

The locked run supports the existence of a same-gene morphological regime map, but not the clean signed-geometry story the project originally set out to test.

| Metric                                                 | Value        |
| ------------------------------------------------------ | ------------ |
| Analyzed same-gene ORF/CRISPR pairs                    | 5,332        |
| Pairs active in both modalities                        | 2,552        |
| Confident non-inactive final labels                    | 2,724        |
| Final ambiguous fraction                               | 48.9%        |
| Final asymmetric_CRISPR fraction                       | 30.3%        |
| Final asymmetric_ORF fraction                          | 20.8%        |
| Plate-holdout final-label agreement                    | 0.868        |
| Artifact-removal final-label agreement                 | 0.896        |
| Official vs. interpretable primary-only agreement      | 0.758        |
| Official vs. no-final-centering primary-only agreement | 0.951        |
| Official vs. artifact-sensitivity primary-only agreement | 0.994       |
| Best official retrieval MRR (primary score)            | 0.001656     |
| Significant Reactome terms                             | 536          |
| Significant CORUM terms                                | 0            |

The decisive visual outcome is not the presence of strong aligned or inverse quadrants; it is the dominance of ambiguous and asymmetric labels after bootstrap consensus. No aligned, inverse, or orthogonal labels survived the final consensus layer. Robustness is partial rather than absent: plate holdout and artifact sensitivity are reassuring, while the interpretable pipeline agreement is only moderate and regime prevalence shifts substantially with mean strength. Retrieval is effectively null-like (`MRR = 0.001656`), so the geometry cannot be presented as a useful same-gene ranking engine. Biological enrichment, however, survives and concentrates in asymmetric regimes.

## Contributions

1. A locked, auditable analysis of same-gene ORF and CRISPR Cell Painting profiles on the `cpg0016` subset of the JUMP genetic resource, with pre-registered thresholds (strength z = 2.0, replicate-consistency z = 2.0, sign z-threshold 1.96, asymmetry threshold 1.0, 200 bootstrap samples, 0.7 consensus threshold).
2. A signed intervention-geometry framework over matched gain-of-function and loss-of-function centroids, with activity-aware nulls, explicit regime taxonomy (aligned, inverse, orthogonal, asymmetric_ORF, asymmetric_CRISPR, ambiguous, inactive), and bootstrap consensus labels.
3. A robustness stack of three locked comparison runs (interpretable pipeline, no-final-centering, artifact-sensitivity) plus within-run confound checks (plate holdout, strength-decile prevalence shift, cell-count scan, magnitude residualization).
4. An honest, non-positive report: structured biology exists and enrichment concentrates in asymmetric regimes, but the map is dominated by asymmetry and ambiguity, partially fragile under preprocessing variation, and retrieval-useless.

## Scope

This release is intentionally narrow and claim-safe.

- One public image-based profiling resource: the `cpg0016` genetic subset of JUMP Cell Painting
- Same-gene ORF and CRISPR pairs only; no orthogonal perturbation types
- Locked pipeline: one official run plus three pre-specified comparison runs
- Descriptive regime taxonomy, not a mechanistic model of dosage, editing efficiency, or pathway nonlinearity
- No wet-lab validation
- Retrieval is too weak to support a utility-forward claim

The contribution is not a signed-geometry discovery. It is a locked boundary result: a defensible statement of what a pre-registered morphology pipeline can and cannot support on matched ORF/CRISPR data.

## Paper

- Compiled PDF: [`paper/main.pdf`](paper/main.pdf)
- LaTeX source: [`paper/main.tex`](paper/main.tex)
- Build script: [`paper/build.ps1`](paper/build.ps1)
- Figure source data: [`paper/source_data/`](paper/source_data/)
- Summary evidence bundle: [`paper/evidence/`](paper/evidence/)

## Repository Layout

- [`src/siggeo/`](src/siggeo/) — Python implementation of preprocessing, gene-level statistics, signed geometry, retrieval, enrichment, confounds, figures, and paper/report assembly
- [`scripts/`](scripts/) — numbered entrypoints for the full phase-driven pipeline
- [`configs/`](configs/) — official, smoke, and robustness run configurations
- [`tests/`](tests/) — unit and integration-style tests for the pipeline
- [`paper/`](paper/) — manuscript package with PDF, LaTeX source, figures, evidence summaries, and source-data CSVs
- [`data/`](data/) — lightweight processed and interim assets used by the pipeline

## Reproducibility

The top-level manuscript package is self-contained and rebuilds `paper/main.pdf` from the local `paper/figures/` assets:

```bash
python -m pip install -e .
python scripts/00_download_assets.py --config configs/paths.example.yaml
python scripts/01_build_metadata_db.py --config configs/paths.example.yaml
python -m siggeo.cli run-pipeline --config configs/analysis_official.yaml
```

```powershell
powershell -ExecutionPolicy Bypass -File .\paper\build.ps1
```

Full local run bundles (raw downloads, memmaps, log folders) remain outside the tracked repository.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

## Citation

```bibtex
@unpublished{uyar2026signedgeometry,
  author = {Uyar, Ali},
  title  = {Structured Yet Fragile: Signed Intervention Geometry of Matched {ORF} and {CRISPR} Cell Painting Profiles},
  year   = {2026},
  doi    = {10.5281/zenodo.19185214},
  note   = {Independent research}
}
```
